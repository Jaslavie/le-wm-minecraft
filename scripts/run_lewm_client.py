import pickle
import os
import struct
import torch
import socket
import time
import math
import wandb
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf

from lewm.models.lewm import LeWM
from lewm.planning.planner import Planner
from lewm.data.utils import get_cam_mean_std
from lewm.paths import repo_path
from lewm.data.utils import planner_output_to_actions

def bytes_to_image(frame):
    image = np.frombuffer(frame, dtype=np.uint8)
    image = image.reshape((64, 64, 3)) # malmo is 540, 952
    image = Image.fromarray(image)

    return image

def load_trained_lewm(cfg: DictConfig, checkpoint):
    lewm_model = LeWM(
        image_size=cfg.vit.image_size,
        patch_size=cfg.vit.patch_size,
        embedding_dim=cfg.vit.embedding_dim,
        num_channels=cfg.vit.num_channels,
        num_patches=cfg.vit.num_patches,
        vit_attention_heads=cfg.vit.attention_heads,
        vit_mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes,
        vit_transformer_blocks=cfg.vit.transformer_blocks,
        predictor_attention_heads=cfg.predictor.attention_heads,
        predictor_mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes,
        predictor_transformer_blocks=cfg.predictor.transformer_blocks,
        action_dim=cfg.action_dim,
        dropout=cfg.predictor.dropout,
        history_len=cfg.predictor.history_len,
        num_proj=cfg.sigreg.num_proj,
        factor=cfg.sigreg.factor,
        phi=cfg.sigreg.phi,
    )
    lewm_model.load_state_dict(checkpoint["model_state_dict"])
    lewm_model.eval()

    return lewm_model

def process_frame_pixels(transform, frame):
    """resizes raw malmo frame to 64x64"""

    img_t = transform(bytes_to_image(frame)).unsqueeze(0) * 255.0 # (1, 3, 64, 64)

    return img_t

def recvall(sock, nbytes):
    """Helper receives byte data from socket"""
    data = b""
    while len(data) < nbytes:
        chunk = sock.recv(nbytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def run_inference(
    model_path: str,
    env_name: str,
    goal_state: str,
    goal_path: str | Path | None = None,
    use_wandb: bool = True,
):
    """
    model_path: path to the trained LeWM model checkpoint
    env_name: name of Malmo sandbox environment to test on
    goal_state: task objective, either navigation or chopping
    goal_path: path to one goal frame pickle file
    use_wandb: whether to use wandb for logging
    
    Returns:
        path to the MSE dashboard image
        None if wandb is used
    """
    # load environment configuration
    cfg = OmegaConf.load(repo_path("config", "lewm.yaml"))
    env_cfg = cfg.env.configs[env_name]
    target_position = tuple(env_cfg.target_position)
    target_name = env_cfg.target_name

    # load paths to goal frame and checkpoint
    goal_path = goal_path or repo_path(cfg.paths.goal_frame)
    checkpoint = torch.load(Path(model_path), map_location="cpu") # Allow model selection for evals

    # load camera mean and std
    cam_mean, cam_std = get_cam_mean_std(str(repo_path(cfg.paths.dataset_h5)))
    
    transform = transforms.Compose([
        transforms.Resize((cfg.vit.image_size, cfg.vit.image_size)),
        transforms.ToTensor(),
    ])

    # Load goal frame to correct size for encoder
    goal_name = Path(goal_path).stem
    with open(str(goal_path), "rb") as file:
        goal_obs = process_frame_pixels(transform, pickle.load(file)[0])

    # Initialize models
    lewm_model = load_trained_lewm(cfg, checkpoint)
    planner = Planner(
        max_iter=cfg.planner.max_iter,
        n_samples=cfg.planner.n_samples,
        n_elites=cfg.planner.n_elites,
        planning_horizon=cfg.planner.planning_horizon,
        action_dim=cfg.action_dim,
    )

    logs_dir = repo_path(cfg.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            job_type="planning",
            config=OmegaConf.to_container(cfg, resolve=False),
            dir=str(logs_dir / "wandb_runs"),
        )
        wandb.define_metric("planning/step")
        wandb.define_metric("planning/*", step_metric="planning/step")
        wandb.define_metric("control/*", step_metric="planning/step")

    # initialize planning parameters
    step = 0
    cycle_times = []
    action_queue = []
    planning_losses = None
    frames = []
    metric_steps = []
    latent_goal_mses = []
    distance_by_step = []
    # initialize warm start for first plan
    last_distribution_params = None

    # Connect to server socket
    print("Establishing connection...", end="")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 25565))
    print("Connected!")

    # Begin loop
    failed = False
    success_threshold_met = False
    while (
        not failed
        and step < cfg.planner.max_steps # reached max steps
        and not success_threshold_met # reached success threshold
    ):
        start = time.perf_counter()
        # Receive frame
        frame = recvall(client, 64 * 64 * 3)
        if frame is None:
            print("Connection closed (mission ended or Malmo server stopped).")
            failed = True
            continue

        # Pair with lewm_integration_layer_py27: frame, then length-prefixed stats.
        stats_len = struct.unpack(">I", recvall(client, 4))[0]
        agent_stats = pickle.loads(recvall(client, stats_len))

        # Objective is to decrease euclidean distance to target
        agent_x = agent_stats.get("x")
        agent_z = agent_stats.get("z")
        distance_to_tree = float(math.dist((agent_x, agent_z), target_position))

        # Save frames for video creation
        frame_np = np.frombuffer(frame, dtype=np.uint8).reshape(64, 64, 3).copy()
        frames.append(frame_np)

        # Preprocess current observation (1, 3, 64, 64)
        obs = process_frame_pixels(transform, frame)

        # Plan when action queue is empty
        if not action_queue:
            warm_start = None
            
            # Set warm start by shifting the previous sampling distributions forward.
            if last_distribution_params is not None:
                warm_start = {
                    "mu": np.concatenate(
                        [last_distribution_params["mu"][1:], last_distribution_params["mu"][-1:]],
                        axis=0,
                    ),
                    "sigma": np.concatenate(
                        [last_distribution_params["sigma"][1:], last_distribution_params["sigma"][-1:]],
                        axis=0,
                    ),
                    "p": np.concatenate(
                        [last_distribution_params["p"][1:], last_distribution_params["p"][-1:]],
                        axis=0,
                    ),
                }
            
            # Planner embeds obs with vit in its pipeline
            action_sequence, planning_losses, distribution_params = planner.planner(
                lewm_model, obs, goal_obs, cam_mean, cam_std, cfg.sigreg.lambd, warm_start=warm_start
            )
            current_goal_mse = planning_losses["current_goal_mse"]
            # update warm start for next plan
            last_distribution_params = distribution_params

            action_queue = list(planner_output_to_actions(action_sequence, cam_mean, cam_std))
            
            print(f"finished planning: action_sequence={action_sequence.shape}, queue={len(action_queue)}")
        else:
            # Plan when action queue is not empty
            goal_latent = planning_losses["goal_latent"]
            with torch.no_grad():
                lewm_model.eval()
                z1 = lewm_model.encoder(obs.to(goal_latent.device))
            current_goal_mse = planner.objective_function(z1, goal_latent).item()

        if current_goal_mse <= cfg.planner.success_threshold:
            print(f"reached goal {goal_name} (mse={current_goal_mse:.4f})")
            success_threshold_met = True

        if success_threshold_met:
            continue

        # Execute first action
        action_to_take = action_queue.pop(0)
        action_to_take[[1, 3]] = 0.0 # TODO: currently disabling left/right movement
        if goal_state != "chopping": # TODO: currently disabling attack for navigation
            action_to_take[7] = 0.0
        print(f"Current action: {action_to_take} ({len(action_queue)} left in plan)")

        # Collect metrics
        cycle_time = time.perf_counter() - start
        cycle_times.append(cycle_time)
        step += 1
        print(f"Finished step {step} / {cfg.planner.max_steps} | Runtime: {cycle_time:.4f} seconds")

        # W&B logs
        if use_wandb:
            metrics = {
                "planning/step": step,
                "planning/env_name": env_name,
                "planning/goal_state": goal_state,
                "planning/goal_name": goal_name,
                "control/target_name": target_name,
                "control/task_success_latent_mse": current_goal_mse,

                "planning/cycle_time": cycle_time, # time to plan and execute action
                "planning/avg_cycle_time": sum(cycle_times) / len(cycle_times),
                # Videos
                "planning/rollout_video": wandb.Video(
                    np.transpose(np.array(frames, dtype=np.uint8), (0, 3, 1, 2)),
                    format="mp4",
                    caption=f"step {step}",
                ),
                "planning/frame": wandb.Image(
                    frame_np, caption=f"step {step}"
                )
            }

            metrics["control/distance_to_tree"] = distance_to_tree
            metrics["control/task_success_distance"] = float(
                distance_to_tree < cfg.planner.success_distance
            )

            # Plot planner objective and runtime.
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(current_goal_mse, label="Latent distance to goal")
            ax1.legend()
            ax2.plot(cycle_times, label="cycle")
            ax2.plot([sum(cycle_times[:i]) / i for i in range(1, len(cycle_times) + 1)], label="avg")
            ax2.legend()
            metrics["planning/dashboard"] = wandb.Image(fig)
            wandb.log(metrics, step=step)
            plt.close(fig)
        else:
            # Collect metrics for plotting
            metric_steps.append(step)
            latent_goal_mses.append(current_goal_mse)
            distance_by_step.append((step, distance_to_tree))

        # Send actions to Malmo to perform
        client.sendall(pickle.dumps(action_to_take.tolist()))

        # Navigation success is defined by the real distance to the target
        if distance_to_tree < cfg.planner.success_distance:
            print(
                f"reached {target_name}: "
                f"distance={distance_to_tree:.2f} < {cfg.planner.success_distance}"
            )
            success_threshold_met = True

    # Save rollout frames for video creation
    rollout_frames_path = repo_path(cfg.paths.fixtures_dir, "rollout_frames.pkl")
    rollout_frames_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rollout_frames_path, "wb") as file:
        pickle.dump(frames, file)
    print(f"Saved rollout frames to {rollout_frames_path}")

    client.close()

    # Close wandb or upload metrics
    if use_wandb:
        wandb.finish()
        return None
    else:
        # Return MSE and other graphs in a single image
        output_path = logs_dir / f"{model_path.stem}_mse_dashboard.png"
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        ax1.plot(metric_steps, latent_goal_mses, marker="x", label="Latent MSE to goal")
        ax1.set_xlabel("Planning step")
        ax1.set_ylabel("MSE")
        ax1.set_title("Latent distance to goal")
        ax1.legend()

        if distance_by_step:
            ax2.plot(*zip(*distance_by_step), marker="o", color="green", label="Distance to tree")
            ax2.axhline(cfg.planner.success_distance, color="red", ls="--", label="Success threshold")
        ax2.set_xlabel("Planning step")
        ax2.set_ylabel("Blocks")
        ax2.set_title("Real distance to tree")
        ax2.legend()

        ax3.plot(metric_steps, cycle_times, marker="o", label="Cycle time")
        ax3.plot(
            metric_steps,
            [sum(cycle_times[:i]) / i for i in range(1, len(cycle_times) + 1)],
            label="Average cycle time",
        )
        ax3.set_xlabel("Planning step")
        ax3.set_ylabel("Seconds")
        ax3.set_title("Planning runtime")
        ax3.legend()
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return str(output_path)

if __name__ == "__main__":
    run_inference(
        # model_path=repo_path("artifacts", "final_models", "best_model_custom_vit.pt"),
        model_path=repo_path("artifacts", "final_models", "best_model_resnet.pt"),
        use_wandb=True,
        env_name=os.environ.get("LEWM_ENV", "single_tree_navigation"),
        goal_state=os.environ.get("LEWM_GOAL_STATE", "navigation"),
    )

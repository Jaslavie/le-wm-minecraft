import pickle
import os
import struct
import torch
import torch.nn.functional as F
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
from omegaconf import OmegaConf

from lewm.planning.planner import Planner
from lewm.paths import repo_path
from lewm.utils import get_cam_mean_std, load_trained_lewm, planner_output_to_actions

def bytes_to_image(frame):
    image = np.frombuffer(frame, dtype=np.uint8)
    image = image.reshape((64, 64, 3)) # malmo is 540, 952
    image = Image.fromarray(image)

    return image

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

def print_metrics_dashboard(metric_steps, latent_goal_mses, distance_by_step, planning_times, planning_time_steps, logs_dir):
    output_path = logs_dir / f"{model_path.stem}_mse_dashboard.png"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.plot(metric_steps, latent_goal_mses, marker="x", label="Latent MSE to goal")
    ax1.set_xlabel("Planning step")
    ax1.set_ylabel("MSE")
    ax1.set_title("Latent distance to goal")
    ax1.legend()
    
    # Distance to tree
    ax2.plot(*zip(*distance_by_step), marker="o", color="green", label="Distance to tree")
    ax2.axhline(cfg.planner.success_distance, color="red", ls="--", label="Success threshold")
    ax2.set_xlabel("Planning step")
    ax2.set_ylabel("Blocks")
    ax2.set_title("Real distance to tree")
    ax2.legend()

    # Planning time
    ax3.plot(planning_time_steps, planning_times, marker="o", label="Planning time")
    ax3.plot(
        planning_time_steps,
        [sum(planning_times[:i]) / i for i in range(1, len(planning_times) + 1)],
        label="Average planning time",
    )
    ax3.set_xlabel("Planning step")
    ax3.set_ylabel("Seconds")
    ax3.set_title("Planning runtime")
    ax3.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)
        
def run_inference(
    model_path: str,
    env_name: str,
    use_wandb: bool = True,
    eval_mode: bool = False,
):
    """
    model_path: path to the trained LeWM model checkpoint
    env_name: name of Malmo sandbox environment to test on
    use_wandb: whether to use wandb for logging
    eval_mode: strip wandb/plotting/frame-saving and return per-episode status.

    Returns:
        eval_mode -> dict with status and success metrics
        use_wandb -> None
        otherwise -> path to the MSE dashboard image
    """
    # eval mode never logs to wandb
    if eval_mode:
        use_wandb = False
    # load environment configuration
    cfg = OmegaConf.load(repo_path("config", "lewm.yaml"))
    env_cfg = cfg.env.configs[env_name]
    # multi-tree environments will have multiple target positions
    if env_cfg.get("target_positions"):
        target_positions = [tuple(pos) for pos in env_cfg.target_positions]
    else:
        target_positions = [tuple(env_cfg.target_position)]
    # begin with first target position
    target_idx = 0
    target_position = target_positions[target_idx]
    target_name = env_cfg.target_name

    # load paths to checkpoint
    checkpoint = torch.load(Path(model_path), map_location="cpu") # Allow model selection for evals

    # load camera mean and std
    cam_mean, cam_std = get_cam_mean_std(str(repo_path(cfg.paths.data_dir, "mineRL_training.h5")))
    
    transform = transforms.Compose([
        transforms.Resize((cfg.vit.image_size, cfg.vit.image_size)),
        transforms.ToTensor(),
    ])

    # Load goal frames to correct size for encoder
    nav_goal_path = repo_path(cfg.paths.nav_goal_frame)
    chop_goal_path = repo_path(cfg.paths.chop_goal_frame)
    goal_name, chop_goal_name = nav_goal_path.stem, chop_goal_path.stem
    with open(str(nav_goal_path), "rb") as file:
        goal_obs = process_frame_pixels(transform, pickle.load(file)[0])
    # Initialize with the first stage: navigation
    stage = "NAV"
    with open(str(chop_goal_path), "rb") as file:
        chop_goal_obs = process_frame_pixels(transform, pickle.load(file)[0])

    # Set universal device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    # Initialize models (strict=False tolerates checkpoints without the inverse-dynamics head)
    lewm_model = load_trained_lewm(cfg, checkpoint, device, strict=False)

    @torch.no_grad()
    def enc(obs):
        return lewm_model.encoder(obs.to(device)).view(-1)
    
    planner = Planner(
        max_iter=cfg.planner.max_iter,
        n_samples=cfg.planner.n_samples,
        n_elites=cfg.planner.n_elites,
        planning_horizon=cfg.planner.planning_horizon,
        action_dim=cfg.action_dim,
        rollout_batch_size=cfg.planner.rollout_batch_size,
    )

    # Encode chop goal frame which is shared across all trees
    # We use this to determine if chopping is complete by computing the latent MSE
    with torch.no_grad():
        chop_goal_z = enc(chop_goal_obs)
        nav_goal_z = enc(goal_obs)

    # Initialize logging directory
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
    planning_times = []
    planning_time_steps = []
    last_planning_time = None
    action_queue = []
    planning_losses = None
    frames = []
    metric_steps = []
    latent_goal_mses = []
    distance_by_step = []
    # initialize warm start for first plan
    last_distribution_params = None
    nav_mse = float("inf")
    chop_done_ct = 0
    chop_stage_steps = 0
    # Neutral action so Malmo does not keep moving during long CEM replans.
    stop_action = np.zeros(cfg.action_dim, dtype=np.float64)

    # Connect to server socket
    print("Establishing connection...", end="")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 25565))
    print("Connected!")

    # Begin loop
    failed = False
    while (
        not failed
        and step < cfg.planner.max_steps # reached max steps
    ):
        # Receive frame
        frame = recvall(client, 64 * 64 * 3)
        if frame is None:
            print("Connection closed (mission ended or Malmo server stopped).")
            failed = True
            continue

        stats_len = struct.unpack(">I", recvall(client, 4))[0]
        agent_stats = pickle.loads(recvall(client, stats_len))

        # Save frames for video creation
        frame_np = np.frombuffer(frame, dtype=np.uint8).reshape(64, 64, 3).copy()
        frames.append(frame_np)

        # Preprocess current observation (1, 3, 64, 64)
        obs = process_frame_pixels(transform, frame)

        # Switch to CHOP when the frame reaches the MSE threshold
        # if stage == "NAV" and distance_to_tree < cfg.planner.success_distance:
        if stage == "NAV" and F.mse_loss(enc(obs), nav_goal_z).item() < cfg.planner.nav_done_mse:
            stage = "CHOP"
            action_queue = []
            last_distribution_params = None
            chop_done_ct = 0
            chop_stage_steps = 0
            print("[CHOP] stage switched to CHOP")
            print(f"stage-CHOP at step {step}")

        # Chop success is judged by latent MSE.
        if stage == "CHOP":
            # start first chop
            chop_stage_count += 1 if F.mse_loss(enc(obs), chop_goal_z).item() < cfg.planner.chop_done_mse else 0
            # continue chopping continuously for a minimum number of steps
            if chop_stage_count >= cfg.planner.chop_done_patience:
                stage = "SUCCESS"
        if stage == "SUCCESS":
            break

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
            active_goal_obs = chop_goal_obs if stage == "CHOP" else goal_obs
            planning_start = time.perf_counter()
            action_sequence, planning_losses, distribution_params = planner.planner(
                lewm_model, obs, active_goal_obs, cam_mean, cam_std, cfg.sigreg.lambd, warm_start=warm_start
            )
            last_planning_time = time.perf_counter() - planning_start
            planning_times.append(last_planning_time)
            planning_time_steps.append(step)
            current_goal_mse = planning_losses["current_goal_mse"]
            # update warm start for next plan
            last_distribution_params = distribution_params

            # build action queue after CEM planning runs
            action_queue = list(planner_output_to_actions(action_sequence, cam_mean, cam_std))
            # append stop action to queue to prevent movement during long replans
            action_queue.append(stop_action.copy())

            print(
                f"[{stage}] "
                f"planning_time={last_planning_time:.4f}s "
                f"selected_action={action_queue[0]} "
                f"mse_final={planning_losses['final_goal_mse']:.4f} "
            )
        else:
            current_goal_mse = planning_losses["current_goal_mse"]

        # Execute first action
        action_to_take = action_queue.pop(0)
        if stage == "CHOP":
            action_to_take[7] = 1.0

        # Collect metrics
        step += 1

        # W&B logs
        if use_wandb:
            metrics = {
                "planning/step": step,
                "planning/env_name": env_name,
                "planning/goal_state": stage,
                "planning/stage": stage,
                "planning/target_index": target_idx,
                "planning/goal_name": chop_goal_name if stage == "CHOP" else goal_name,
                "control/target_name": target_name,
                "control/task_success_latent_mse": current_goal_mse,

                "planning/planning_time": last_planning_time,
                "planning/avg_planning_time": sum(planning_times) / len(planning_times),
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

            metrics["control/nav_mse"] = nav_mse
            metrics["control/distance_to_tree"] = distance_to_tree

            # Plot planner objective and planning runtime.
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(current_goal_mse, label="Latent distance to goal")
            ax1.legend()
            ax2.plot(planning_times, label="planning")
            ax2.plot([sum(planning_times[:i]) / i for i in range(1, len(planning_times) + 1)], label="avg")
            ax2.legend()
            metrics["planning/dashboard"] = wandb.Image(fig)
            wandb.log(metrics, step=step)
            plt.close(fig)
        else:
            # Collect metrics for plotting
            metric_steps.append(step)
            latent_goal_mses.append(current_goal_mse)
            nav_mse_by_step.append((step, nav_mse))
            distance_by_step.append((step, distance_to_tree))

        # Send actions to Malmo to perform
        client.sendall(pickle.dumps(action_to_take.tolist()))

    # Save rollout frames for video creation (skipped in eval mode for speed)
    if not eval_mode:
        rollout_frames_path = repo_path(cfg.paths.fixtures_dir, "rollout_frames.pkl")
        rollout_frames_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rollout_frames_path, "wb") as file:
            pickle.dump(frames, file)
        print(f"Saved rollout frames to {rollout_frames_path}")

    client.close()
    if planning_times:
        avg_planning_time = sum(planning_times) / len(planning_times)
        print(
            f"Average planning runtime: {avg_planning_time:.4f}s "
            f"over {len(planning_times)} planning session(s)"
        )

    # eval mode: return per-episode result dict, skip dashboard/plotting
    if eval_mode:
        return {
            "status": stage,
            # success criteria
            "nav_success": stage in ("CHOP", "SUCCESS"),
            "chop_success": stage == "SUCCESS",
            "min_nav_mse": None if min_nav_mse == float("inf") else round(min_nav_mse, 3),
            "steps": step,
            "avg_planning_time": avg_planning_time if planning_times else None,
        }

    # Close wandb or upload metrics
    if use_wandb:
        wandb.finish()
        return None
    else:
        # Return MSE and other graphs in a single image
        print_metrics_dashboard(metric_steps, latent_goal_mses, distance_by_step, planning_times, planning_time_steps, logs_dir)

def run_evals(model_path, env_name, n_episodes=100):
    """
    Run n_episodes of rollouts without wandb
    """
    successes = 0
    for ep in range(n_episodes):
        success = run_inference(model_path, env_name, eval_mode=True)["nav_success"]
        successes += int(success)
        print(f"[eval] episode {ep + 1}/{n_episodes} | success={success} | running ratio={successes / (ep + 1):.3f}")
    ratio = successes / n_episodes
    print(f"[eval] final success ratio: {successes}/{n_episodes} = {ratio:.3f}")
    return ratio

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evals", action="store_true", help="Run rollouts without wandb")
    parser.add_argument("--episodes", type=int, default=10, help="Set number of rollouts to run")
    args = parser.parse_args()

    # model_path=repo_path("artifacts", "final_models", "best_model_custom_vit.pt"),
    model_path = repo_path("artifacts", "final_models", "best_model_resnet_invdyn.pt")
    env_name = os.environ.get("LEWM_ENV", "single_tree_navigation")

    if args.evals:
        run_evals(model_path, env_name, n_episodes=args.episodes)
    else:
        run_inference(
            model_path=model_path,
            use_wandb=True,
            env_name=env_name,
        )

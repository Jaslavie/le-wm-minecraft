import pickle
import torch
import socket
import time
import wandb
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf

from lewm.models.lewm import LeWM
from lewm.planning.planner import Planner
from lewm.data.utils import get_cam_mean_std, planner_output_to_actions
from lewm.paths import repo_path

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
        phi=cfg.sigreg.phi
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

def main():
    cfg = OmegaConf.load(repo_path("config", "lewm.yaml"))
    checkpoint = torch.load(repo_path("autoresearch_experiments", "checkpoints", "epoch_002.pt"), map_location="cpu")
    goal_file = repo_path("artifacts", "fixtures", "forest_goal_frame.pkl")
    cam_mean, cam_std = get_cam_mean_std(str(repo_path(cfg.paths.dataset_h5)))
    transform = transforms.Compose([
        transforms.Resize((cfg.vit.image_size, cfg.vit.image_size)),
        transforms.ToTensor(),
    ])

    # Read file values
    with open(goal_file, "rb") as file:
        goal_frame = pickle.load(file)[0]

    # Preprocess goal frame
    goal_obs = process_frame_pixels(transform, goal_frame)

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

    # initialize warm start for first plan
    last_distribution_params = None

    # Connect to server socket
    print("Establishing connection...", end="")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 25565))
    print("Connected!")

    # Begin loop
    done = False
    while not done:
        start = time.perf_counter()
        # Receive frame
        frame = recvall(client, 64 * 64 * 3)
        if frame is None:
            print("Connection closed (mission ended or Malmo server stopped).")
            done = True
            continue
        
        # Save frames for video creation
        frame_np = np.frombuffer(frame, dtype=np.uint8).reshape(64, 64, 3).copy()
        frames.append(frame_np)

        # Preprocess current observation (1, 3, 64, 64)
        obs = process_frame_pixels(transform, frame)
        print(f"finished processing: obs={obs.shape}, goal_obs={goal_obs.shape}")

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
            # update warm start for next plan
            last_distribution_params = distribution_params

            action_queue = list(action_sequence)
           
            print(f"finished planning: action_sequence={action_sequence.shape}, queue={len(action_queue)}")

        # Execute first action
        action_to_take = action_queue.pop(0)
        print(f"Current action: {action_to_take} ({len(action_queue)} left in plan)")

        # Collect metrics
        end = time.perf_counter()
        cycle_time = end - start
        cycle_times.append(cycle_time)
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        step += 1
        print(f"Runtime: {cycle_time:.4f} seconds")

        # Prepare video
        video_array = np.array(frames, dtype=np.uint8)
        video_array = np.transpose(video_array, (0, 3, 1, 2))
        
        # W&B
        #   objective latent distance: how close the real observation is to the goal in latent space
        #   dreaming gap: how close the imagined next latent is to the next real latent
        #   task success pixel mse: did the agent reach the goal in pixel space?
        pixel_goal_mse = F.mse_loss(obs.float() / 255.0, goal_obs.float() / 255.0).item()
        metrics = {
            "planning/step": step,
            "control/task_success_latent_mse": planning_losses["current_goal_mse"],
            "control/task_success_pixel_mse": pixel_goal_mse,
            
            "planning/cycle_time": cycle_time, # time to plan and execute action
            "planning/avg_cycle_time": avg_cycle_time,
            # Videos
            "planning/rollout_video": wandb.Video(
                video_array, format="mp4", caption=f"step {step}"
            ),
            "planning/frame": wandb.Image(
                frame_np, caption=f"step {step}"
            )
        }

        # Plot actual distance to goal vs imagined distance
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(pixel_goal_mse, label="Actual pixel distance to goal")
        ax1.plot(planning_losses["current_goal_mse"], label="Imagined distance to goal")
        ax1.legend()
        ax2.plot(cycle_times, label="cycle")
        ax2.plot([sum(cycle_times[:i]) / i for i in range(1, len(cycle_times) + 1)], label="avg")
        ax2.legend()
        plt.close(fig)

        # Send actions to Malmo to perform
        client.sendall(pickle.dumps(action_to_take.tolist()))

    client.close()

    wandb.finish()

if __name__ == "__main__":
    main()

import pickle
import torch
import socket
import time
import wandb
# import cv2
from pathlib import Path
# import hydra
import numpy as np
from PIL import Image
# import  stable_worldmodel as swm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf

# Project files
import predictor
from planner import Planner
import lewm
import utils

def bytes_to_image(frame):
    image = np.frombuffer(frame, dtype=np.uint8)
    image = image.reshape((64, 64, 3)) # malmo is 540, 952
    image = Image.fromarray(image)

    return image

def load_trained_lewm(cfg: DictConfig, checkpoint):
    weights = checkpoint["model_state_dict"]

    lewm_model = lewm.LeWM(
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
    lewm_model.load_state_dict(weights)
    lewm_model.eval()

    return lewm_model

def process_frame_pixels(transform, frame):
    """resizes raw malmo frame to 64x64"""

    img_t = transform(bytes_to_image(frame)).unsqueeze(0) # (1, 3, 64, 64)

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

# @hydra.main(version_base=None, config_path="./config", config_name="lewm")
def main():#cfg: DictConfig):
    # Initialize variables
    checkpoint = torch.load("best_model.pt", map_location="cpu")
    cfg = OmegaConf.load("./config/lewm.yaml")
    # action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    goal_file = "./goal_frame.pkl"
    cam_mean, cam_std = utils.get_cam_mean_std("mineRL_training.h5")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
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

    wandb.init(
        project=cfg.wandb.project,
        job_type="planning",
        config=OmegaConf.to_container(cfg, resolve=False),
        dir="logs/wandb_runs",
    )
    wandb.define_metric("planning/step")
    wandb.define_metric("planning/*", step_metric="planning/step")

    step = 0
    cycle_times = []
    current_goal_mses = []
    cem_best_costs = []
    action_queue = []
    planning_losses = None
    frames = []
    # video_path = Path("logs/planning_rollout.mp4")
    Path("logs").mkdir(exist_ok=True)

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
            # TODO: Update warm start 
            
            # Planner embeds obs with vit in its pipeline
            mu, planning_losses = planner.planner(
                lewm_model, obs, goal_obs, cam_mean, cam_std, cfg.sigreg.lambd, warm_start=None
            )
            action_queue = list(utils.planner_output_to_actions(mu, cam_mean, cam_std))
            
            current_goal_mses.append(planning_losses["current_goal_mse"])
            cem_best_costs.append(planning_losses["cem_best_cost"])
            print(f"finished planning: mu={mu.shape}, queue={len(action_queue)}")

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
        wandb.log({
            "planning/step": step,
            "planning/current_goal_mse": planning_losses["current_goal_mse"],
            "planning/cem_best_cost": planning_losses["cem_best_cost"],
            "planning/imagined_goal_mse": planning_losses["imagined_goal_mse"],
            "planning/cycle_time": cycle_time,
            "planning/avg_cycle_time": avg_cycle_time,
            "planning/rollout_video": wandb.Video(
                video_array, format="mp4", caption=f"step {step}"
            ),
        })

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(current_goal_mses, label="current_goal_mse")
        ax1.plot(cem_best_costs, label="cem_best_cost")
        ax1.legend()
        ax2.plot(cycle_times, label="cycle")
        ax2.plot([sum(cycle_times[:i]) / i for i in range(1, len(cycle_times) + 1)], label="avg")
        ax2.legend()
        wandb.log({"planning/dashboard": wandb.Image(fig)})
        plt.close(fig)

        wandb.log({"planning/frame": wandb.Image(frame_np, caption=f"step {step}")})

        # Send actions to Malmo to perform
        client.sendall(pickle.dumps(action_to_take.tolist()))

    client.close()

    wandb.finish()

if __name__ == "__main__":
    main()

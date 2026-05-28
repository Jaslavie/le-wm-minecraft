import pickle
import torch
import socket
import time
import hydra
import numpy as np
from PIL import Image
import  stable_worldmodel as swm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torchvision import transforms
from omegaconf import DictConfig

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

def process_frame_pixels(frame):
    """resizes raw malmo frame to 64x64"""
    transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    img_t = transform(bytes_to_image(frame)).unsqueeze(0) # (1, 3, 64, 64)

    return img_t

@hydra.main(version_base=None, config_path="./config", config_name="lewm")
def main(cfg: DictConfig):
    # Initialize variables
    checkpoint = torch.load("best_model.pt", map_location="cpu")
    action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    goal_file = "./goal_frame.pkl"
    # dataset = swm.data.HDF5Dataset("mineRL_training", cache_dir=".")
    cam_mean, cam_std = utils.get_cam_mean_std("mineRL_training.h5")

    # Read file values
    with open(goal_file, "rb") as file:
        goal_frame = pickle.load(file)[0]

    # Initialize models
    lewm_model = load_trained_lewm(cfg, checkpoint)
    planner = Planner(
        max_iter=cfg.planner.max_iter,
        n_samples=cfg.planner.n_samples,
        n_elites=cfg.planner.n_elites,
        planning_horizon=cfg.planner.planning_horizon,
        action_dim=cfg.action_dim,
    )

    # Connect to server socket
    print("Establishing connection...", end="")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 25565))
    print(f"Connected!")

    # Begin loop
    done = 0
    while not done:
        start = time.perf_counter()
        # Receive frame
        frame = client.recv(64 * 64 * 3)
        print("Frame received!")
        # plt.figure()
        # plt.imshow(bytes_to_image(frame))
        # plt.show()
        
        # Preprocess current observation and goal (1, 3, 64, 64)
        goal_obs = process_frame_pixels(goal_frame)
        obs = process_frame_pixels(frame)
        print(f"finished processing: obs={obs.shape}, goal_obs={goal_obs.shape}")

        # Planner embeds obs with vit in its pipeline
        mu = planner.planner(lewm_model, obs, goal_obs, cam_mean, cam_std)
        print(f"finished planning: mu={mu.shape}")
        
        action_to_take = utils.planner_output_to_actions(mu, cam_mean, cam_std)[0]
        input(f"Current action: {action_to_take} \n Press ENTER to continue...")
        
        end = time.perf_counter()
        print(f"Runtime: {end - start:.4f} seconds")

        # Send actions to Malmo to perform
        client.sendall(pickle.dumps(action_to_take.tolist()))

        done += 1
    client.close()

if __name__ == "__main__":
    main()

import pickle
import torch
import socket
import hydra
import numpy as np
from PIL import Image
import  stable_worldmodel as swm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from hydra import initialize, compose
from torchvision import transforms
from omegaconf import DictConfig

# Project files
import predictor
from planner import Planner
import lewm
import utils

def bytes_to_image(frame):
    image = np.frombuffer(frame, dtype=np.uint8)
    image = image.reshape((540, 952, 3)) # malmo is 540, 952
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

@hydra.main(version_base=None, config_path="../config", config_name="lewm")
def main():
    # Initialize variables
    checkpoint = torch.load("best_model.pt", map_location="cpu")
    action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    goal_file = "./goal_frame.pkl"
    cam_mean, cam_std = utils.get_cam_mean_std("mineRL_training.h5")
    

    # Read file values
    with open(goal_file, "rb") as file:
        goal_frame = pickle.load(file)[0]
    with initialize(config_path="./config", version_base=None):
        cfg = compose(config_name="lewm")

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
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 25565))

    # Begin loop
    done = 0
    while not done:
        # Receive frame
        frame = client.recv(952 * 540 * 3)
        print("Frame received!")
        plt.figure()
        plt.imshow(bytes_to_image(frame))
        plt.show()
        
        # Preprocess current observation and goal (1, 3, 64, 64)
        goal_obs = process_frame_pixels(goal_frame)
        obs = process_frame_pixels(frame)

        # Planner embeds obs with vit in its pipeline 
        mu = planner.planner(lewm_model, obs, goal_obs, cam_mean, cam_std)

        action = mu[0]

        input(f"Current action: {action}\n Press ENTER to continue...")

        # print(len(embeddings))

        client.sendall(pickle.dumps(action))

        done += 1
    client.close()

if __name__ == "__main__":
    main()

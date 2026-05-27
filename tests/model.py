import pickle
import torch
import socket
import hydra
import numpy as np
import  stable_worldmodel as swm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
from vit import tinyViT
from hydra import initialize, compose
from torchvision import transforms
from omegaconf import DictConfig

# Project files
from vit import tinyViT
import predictor
from planner import Planner
import lewm
import utils

def bytes_to_image(frame):
    image = np.frombuffer(frame, dtype=np.uint8)
    image = image.reshape((64, 64, 3)) # 540, 952
    image = Image.fromarray(image)

    return image

def new_vit(cfg: DictConfig):
    checkpoint = torch.load('./best_model.pt', weights_only=True, map_location="cpu")
    vit_weights = checkpoint["model_state_dict"]

    vit_weights = {k.replace("encoder.", ""): v for k, v in vit_weights.items() if k.startswith("encoder.")}

    vit = tinyViT( 
        image_size=cfg.vit.image_size,
        patch_size=cfg.vit.patch_size,
        embedding_dim=cfg.vit.embedding_dim,
        num_channels=cfg.vit.num_channels,
        num_patches=cfg.vit.num_patches,
        attention_heads=cfg.vit.attention_heads,
        mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes,
        transformer_blocks=cfg.vit.transformer_blocks
    )
    vit.load_state_dict(vit_weights, strict=False)
    vit.eval()

    return vit

def process_frame_pixels(frame, vit):
    transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    img_t = transform(bytes_to_image(frame)).unsqueeze(0)

    with torch.no_grad():
        embed = vit(img_t)

    return embed

@hydra.main(version_base=None, config_path="./config", config_name="lewm")
def main():
    # Initialize variables
    model_state = torch.load("best_model.pt", map_location="cpu")
    action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    goal_file = "./goal_frame.pkl"
    cam_mean, cam_std = utils.get_cam_mean_std("mineRL_training.h5")
    

    # Read file values
    with open(goal_file, "rb") as file:
        goal_frame = pickle.load(file)[0]
    with initialize(config_path="./config", version_base=None):
        cfg = compose(config_name="lewm")

    # Initialize mosules
    vit = new_vit(cfg)
    planner = Planner()

    # Connect to server socket
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 25565))

    # Begin loop
    done = 0
    while not done:
        # Receive frame
        frame = client.recv(16384)
        print("Frame received!")
        plt.figure()
        plt.imshow(bytes_to_image(frame))
        plt.show()
        
        # Run vit
        goal_embed = process_frame_pixels(goal_frame, vit)
        obs_embed = process_frame_pixels(frame, vit)

        mu = Planner.planner(lewm, obs_embed, goal_embed, cam_mean, cam_std)

        action = mu[0]

        input(f"Current action: {action}\n Press ENTER to continue...")

        # print(len(embeddings))

        client.sendall(pickle.dumps(action))

        done += 1
    client.close()

if __name__ == "__main__":
    main()
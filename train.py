import stable_worldmodel as swm
from utils import normalize_columns
import hydra
from omegaconf import DictConfig, OmegaConf
from lewm import LeWM
import torch
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader

@hydra.main(version_base=None, config_path="./config", config_name="lewm")
def train(cfg: DictConfig):
    """
    The primary training parameters are random projections M and 
    the regularization weight λ (how much to prioritize stability)
    """
    # Load and normalize original dataset
    ds = swm.data.HDF5Dataset("mineRL_training", cache_dir=".")

    normalizer = normalize_columns(
        ds,
        col="action",
        target_col="action",
    )

    dataset = swm.data.HDF5Dataset(
        "mineRL_training",
        cache_dir=".",
        num_steps=cfg.sub_trajectory, # batch every 4 timestamps
        transform=normalizer,
    )

    randomizer = torch.Generator().manual_seed(42)

    # training size is split based on the episode length
    num_episodes = len(dataset)
    train_size = int(0.80 * num_episodes)
    val_size = num_episodes - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=randomizer)
    print(f"training size: {train_size}, validation size: {val_size}")
    
    train = DataLoader(train_dataset, cfg.batch_size, shuffle=True, drop_last=True, generator=randomizer)
    val = DataLoader(val_dataset, cfg.batch_size, shuffle=False, drop_last=True)

    # init world  model
    lewm = LeWM(
        image_size=cfg.vit.image_size,
        patch_size=cfg.vit.patch_size,
        embedding_dim=cfg.vit.embedding_dim,
        num_channels=cfg.vit.num_channels,
        num_patches=cfg.vit.num_patches,
        attention_heads=cfg.vit.attention_heads,
        mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes,
        transformer_blocks=cfg.vit.transformer_blocks,
        action_dim=cfg.action_dim,
        dropout=cfg.predictor.dropout,
        num_proj=cfg.sigreg.num_proj,
        factor=cfg.sigreg.factor,
        phi=cfg.sigreg.phi
    )
    optimizer = AdamW(lewm.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    
    # run training over batches
    training_loss = 0
    for i, data in enumerate(train):
        # each data sample contains actions and pictures
        action = data["action"]
        pixels = data["pixels"]

        # forward pass
        optimizer.zero_grad()
        loss = lewm(pixels, action, cfg.sigreg.lambd)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        # print loss every 100 batches
        if i % 100 == 0:
            print(f"Training loss: {loss.item()}")
    
    for i, data in enumerate(val):
        # each data sample contains actions and pictures
        

        # forward pass
        loss = lewm(pixels, action, cfg.sigreg.lambd)
        validation_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Validation loss: {loss.item()}")
    
    print(f"Training loss: {training_loss / len(train)}")
    print(f"Validation loss: {validation_loss / len(val)}")



if __name__=="__main__":
    train()
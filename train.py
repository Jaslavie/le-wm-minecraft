import stable_worldmodel as swm
from utils import normalize_columns
import hydra
from lewm import LeWM, compute_loss, SIGReg
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="./config", config_name="lewm")
def train_model(cfg: DictConfig):
    """
    The primary training parameters are random projections M and 
    the regularization weight λ (how much to prioritize stability)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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

    # split training data by samples
    num_episodes = len(dataset.lengths)
    train_size = int(0.80 * num_episodes)
    
    train_indices = [
        i for i, (ep_i, _) in enumerate(dataset.clip_indices)
        if ep_i <= train_size
    ]
    val_indices = [
        i for i, (ep_i, _) in enumerate(dataset.clip_indices)
        if ep_i > train_size
    ]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train = DataLoader(train_dataset, cfg.batch_size, shuffle=True, drop_last=True, generator=randomizer)
    val = DataLoader(val_dataset, cfg.batch_size, shuffle=False, drop_last=True)

    print(f"training size: {train_size}, validation size: {num_episodes - train_size}")

    # init loss
    sigreg = SIGReg(
        num_proj=cfg.sigreg.num_proj,
        factor=cfg.sigreg.factor,
        phi=cfg.sigreg.phi
    )
    
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
    ).to(device)
    optimizer = AdamW(lewm.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    
    # run training over epochs
    for epoch in range(cfg.epochs):
        training_loss = 0
        validation_loss = 0

        # training
        for i, data in enumerate(train):
            # each data sample contains actions and pictures
            action = data["action"].to(device)
            pixels = data["pixels"].to(device)

            # forward pass
            model_out = lewm(pixels, action)
            loss = compute_loss(
                next_emb_pred=model_out[0], 
                next_emb_target=model_out[1],
                emb= model_out[2],
                sigreg= sigreg,
                lambd=cfg.sigreg.lambd,
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            # print loss every 100 batches
            if i % 100 == 0:
                print(f"Training loss for {i} / {len(train)}: {loss.item()}")

        # validation
        with torch.no_grad():
            for i, data in enumerate(val):
                action = data["action"]
                pixels = data["pixels"]
                loss = lewm(pixels, action, cfg.sigreg.lambd)
                validation_loss += loss.item()
        
        # save best after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': lewm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': validation_loss,
        }, cfg.model_path)

        print(f"Total Training loss for epoch {epoch}: {training_loss / len(train)}")
        print(f"Total Validation loss for epoch {epoch}: {validation_loss / len(val)}")



if __name__=="__main__":
    train_model()
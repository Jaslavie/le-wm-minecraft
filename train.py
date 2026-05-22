from pathlib import Path

from utils import normalize_columns
import hydra
import wandb
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from omegaconf import DictConfig, OmegaConf

from lewm import LeWM, compute_loss, SIGReg
import stable_worldmodel as swm

@hydra.main(version_base=None, config_path="./config", config_name="lewm")
def train_model(cfg: DictConfig):
    """
    The primary training parameters are random projections M and 
    the regularization weight λ (how much to prioritize stability)
    """
    # set up wandb training run for each training session
    run = wandb.init(
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"using device {device}")

    # load and normalize original dataset
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
        history_len=cfg.predictor.history_len,
        num_proj=cfg.sigreg.num_proj,
        factor=cfg.sigreg.factor,
        phi=cfg.sigreg.phi
    ).to(device)

    # optimization with warmup and cosine annealing 
    optimizer = AdamW(lewm.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.total_epochs - cfg.warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs])

    checkpoint_dir = Path(cfg.checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(cfg.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # run training over epochs
    for epoch in range(cfg.total_epochs):
        training_loss = 0
        training_pred_loss = 0
        training_sigreg_loss = 0
        validation_loss = 0
        validation_pred_loss = 0
        validation_sigreg_loss = 0

        # training
        for i, data in enumerate(train):
            # each data sample contains actions and pictures
            action = data["action"].to(device)
            pixels = data["pixels"].to(device)

            # forward pass
            model_out = lewm(pixels, action)
            loss, pred_loss, sigreg_loss = compute_loss(
                next_emb_pred=model_out[0], 
                next_emb_target=model_out[1],
                emb= model_out[2],
                sigreg= sigreg,
                lambd=cfg.sigreg.lambd,
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lewm.parameters(), max_norm=1.0)
            optimizer.step()

            training_loss += loss.item()
            training_pred_loss += pred_loss.item()
            training_sigreg_loss += sigreg_loss.item()

            # log training metrics
            wandb.log({
                "train/loss": loss.item(),
                "train/pred_loss": pred_loss.item(),
                "train/sigreg_loss": sigreg_loss.item(),
                "train/weighted_sigreg_loss": cfg.sigreg.lambd * sigreg_loss.item(),
                "epoch": epoch,
                "step": i,
            })

            # print loss every 100 batches
            if i % 100 == 0:
                print(f"Training loss for {i} / {len(train)}: {loss.item()}")

        # increment scheduler per epoch
        scheduler.step()

        # validation
        with torch.no_grad():
            for i, data in enumerate(val):
                action = data["action"].to(device)
                pixels = data["pixels"].to(device)
                model_out = lewm(pixels, action)
                loss, pred_loss, sigreg_loss = compute_loss(
                    next_emb_pred=model_out[0], 
                    next_emb_target=model_out[1],
                    emb= model_out[2],
                    sigreg= sigreg,
                    lambd=cfg.sigreg.lambd,
                )
                validation_loss += loss.item()
                validation_pred_loss += pred_loss.item()
                validation_sigreg_loss += sigreg_loss.item()

        avg_train_loss = training_loss / len(train)
        avg_val_loss = validation_loss / len(val)
        avg_train_pred_loss = training_pred_loss / len(train)
        avg_train_sigreg_loss = training_sigreg_loss / len(train)
        avg_val_pred_loss = validation_pred_loss / len(val)
        avg_val_sigreg_loss = validation_sigreg_loss / len(val)

        # save latest model weights
        torch.save({
            "epoch": epoch,
            "model_state_dict": lewm.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }, model_path)

        # collect checkpoints at the end of each epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": lewm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }, checkpoint_dir / f"epoch_{epoch:03d}.pt")

        # log epoch metrics to wandb
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "train/epoch_pred_loss": avg_train_pred_loss,
            "train/epoch_sigreg_loss": avg_train_sigreg_loss,
            "train/epoch_weighted_sigreg_loss": cfg.sigreg.lambd * avg_train_sigreg_loss,
            "val/epoch_loss": avg_val_loss,
            "val/epoch_pred_loss": avg_val_pred_loss,
            "val/epoch_sigreg_loss": avg_val_sigreg_loss,
            "val/epoch_weighted_sigreg_loss": cfg.sigreg.lambd * avg_val_sigreg_loss,
            "lr": scheduler.get_last_lr()[0],
        })

        print(f"Total Training loss for epoch {epoch}: {avg_train_loss} (pred={avg_train_pred_loss:.6f}, sigreg={avg_train_sigreg_loss:.6f})")
        print(f"Total Validation loss for epoch {epoch}: {avg_val_loss} (pred={avg_val_pred_loss:.6f}, sigreg={avg_val_sigreg_loss:.6f})")

if __name__=="__main__":
    train_model()
from lewm.data.utils import normalize_columns
from lewm.models.lewm import LeWM, compute_loss
from lewm.paths import repo_path
import hydra
import wandb
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm

@hydra.main(version_base=None, config_path="../config", config_name="lewm")
def train_model(cfg: DictConfig):
    """
    The primary training parameters are random projections M and 
    the regularization weight λ (how much to prioritize stability)
    """
    data_dir = repo_path(cfg.paths.data_dir)

    # set up wandb training run for each training session
    run = wandb.init(
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(repo_path(cfg.paths.logs_dir, "wandb_runs")),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"using device {device}")

    # load and normalize original dataset
    ds = swm.data.HDF5Dataset("mineRL_training", cache_dir=str(data_dir))

    normalizer = normalize_columns(
        ds,
        col="action",
        target_col="action",
    )

    dataset = swm.data.HDF5Dataset(
        "mineRL_training",
        cache_dir=str(data_dir),
        num_steps=cfg.sub_trajectory, # batch every 4 timestamps
        transform=normalizer,
    )

    train_randomizer = torch.Generator().manual_seed(42)
    val_randomizer = torch.Generator().manual_seed(43)

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
    
    train = DataLoader(
        train_dataset, cfg.batch_size, shuffle=True, drop_last=True, generator=train_randomizer
    )
    val = DataLoader(
        val_dataset, cfg.batch_size, shuffle=True, drop_last=True, generator=val_randomizer
    )

    print(f"training size: {train_size}, validation size: {num_episodes - train_size}")

    # init world  model
    lewm = LeWM(
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
    ).to(device)


    # optimization with warmup and cosine annealing
    optimizer = AdamW(lewm.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.total_epochs - cfg.warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs])

    checkpoint_dir = repo_path(cfg.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = repo_path(cfg.paths.latest_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_model_path = repo_path(cfg.paths.best_model)
    best_val_pred_loss = float("inf")
    train_epoch_pred_losses = []
    val_epoch_pred_losses = []

    # run training over epochs
    for epoch in range(cfg.total_epochs):
        training_loss = 0
        training_pred_loss = 0
        training_sigreg_loss = 0
        validation_loss = 0
        validation_pred_loss = 0
        validation_sigreg_loss = 0

        # training
        lewm.train()
        for i, data in enumerate(train):
            # each data sample contains actions and pictures
            action = data["action"].to(device)
            pixels = data["pixels"].to(device)

            # forward pass
            model_out = lewm(pixels, action)
            loss, pred_loss, sigreg_loss, inv_loss = compute_loss(
                next_emb_pred=model_out[0],
                next_emb_target=model_out[1],
                emb= model_out[2],
                sigreg=lewm.sigreg,
                lambd=cfg.sigreg.lambd,
                action_pred=model_out[3],
                action_target=action[:, :-1],
                lambd_inv=cfg.inverse_dynamics.lambd,
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
                "train/inv_loss": inv_loss.item(),
                "epoch": epoch,
                "step": i,
            })

            # print loss every 100 batches
            if i % 100 == 0:
                print(f"Training loss for {i} / {len(train)}: {loss.item()}")

        # increment scheduler per epoch
        scheduler.step()
        # validation
        lewm.eval()
        with torch.no_grad():
            for i, data in enumerate(val):
                action = data["action"].to(device)
                pixels = data["pixels"].to(device)
                model_out = lewm(pixels, action)
                loss, pred_loss, sigreg_loss, inv_loss = compute_loss(
                    next_emb_pred=model_out[0],
                    next_emb_target=model_out[1],
                    emb= model_out[2],
                    sigreg=lewm.sigreg,
                    lambd=cfg.sigreg.lambd,
                    action_pred=model_out[3], # predicted action
                    action_target=action[:, :-1], # most recent action (not predicted)
                    lambd_inv=cfg.inverse_dynamics.lambd,
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

        train_epoch_pred_losses.append(avg_train_pred_loss)
        val_epoch_pred_losses.append(avg_val_pred_loss)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": lewm.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_pred_loss": avg_train_pred_loss,
            "val_pred_loss": avg_val_pred_loss,
        }

        # save latest model weights
        torch.save(checkpoint_payload, model_path)

        # save best model by prediction loss
        if avg_val_pred_loss < best_val_pred_loss:
            best_val_pred_loss = avg_val_pred_loss
            torch.save(checkpoint_payload, best_model_path)

        # collect checkpoints at the end of each epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": lewm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_pred_loss": avg_train_pred_loss,
            "val_pred_loss": avg_val_pred_loss,
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
            "epoch": epoch,
        })

        epochs = list(range(len(train_epoch_pred_losses)))
        fig, ax = plt.subplots()
        ax.plot(epochs, train_epoch_pred_losses, marker="o", label="train")
        ax.plot(epochs, val_epoch_pred_losses, marker="o", label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("pred loss")
        ax.set_title("Train vs validation prediction loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        wandb.log({"train_val/epoch_pred_loss": wandb.Image(fig), "epoch": epoch})
        plt.close(fig)

        print(f"Total Training loss for epoch {epoch}: {avg_train_loss} (pred={avg_train_pred_loss:.6f}, sigreg={avg_train_sigreg_loss:.6f})")
        print(f"Total Validation loss for epoch {epoch}: {avg_val_loss} (pred={avg_val_pred_loss:.6f}, sigreg={avg_val_sigreg_loss:.6f})")

if __name__=="__main__":
    train_model()

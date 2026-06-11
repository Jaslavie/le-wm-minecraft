"""
Probes measure if the latent representation contains relevant information about the environment.
https://carpentries-incubator.github.io/fair-explainable-ml/5c-probes.html

Probe continuous observations on physics space (from ObservationFromFullStats). i.e. we will back-predict
xyz position, distance to goal, velocity, etc. based on the latent representation.

Things to measure
    Distance to tree
    Wether the tree is broken or intact
    Is the tree centered in the frame
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pickle
import numpy as np
import wandb
from scipy import stats

from omegaconf import OmegaConf
from lewm.paths import repo_path
from lewm.utils import load_trained_lewm


class MalmoProbeDataset(torch.utils.data.Dataset):
    """
    Loads random Malmo rollout samples collected by malmo_mission_runner_py27.py and
    encodes each frame once with the frozen encoder

    Each sample returns:
        latent: (D,) cached encoder embedding of the frame
        stats:  raw continuous physics values from ObservationFromFullStats
        target: (T,) regression target, set externally via `targets` (see evals.py)
    """
    def __init__(self, dataset_path, encoder, device, batch_size=128):
        # load video sample pickle file
        with open(dataset_path, "rb") as file:
            samples = pickle.load(file)

        self.stats = [sample["stats"] for sample in samples]
        self.actions = [torch.tensor(sample["action"], dtype=torch.float32) for sample in samples]
        self.steps = [sample["step"] for sample in samples]
        # decode every frame into a pixel tensor (N, 3, 64, 64) in 0-255 range
        pixels = torch.stack([
            torch.from_numpy(
                np.frombuffer(sample["frame"], dtype=np.uint8).reshape(64, 64, 3).copy()
            ).permute(2, 0, 1).float()
            for sample in samples
        ])

        # run frozen encoder on all frames
        encoder.eval()
        latents = []
        with torch.no_grad():
            for start in range(0, len(pixels), batch_size):
                chunk = pixels[start:start + batch_size].to(device)
                latents.append(encoder(chunk))
        self.latents = torch.cat(latents, dim=0).float()

        # regression targets are built separately from `stats` and assigned in evals.py
        self.targets = None
        self.class_targets = None

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        sample = {
            "latent": self.latents[idx],
            "target": self.targets[idx],
        }
        if self.class_targets is not None:
            sample["class_target"] = self.class_targets[idx]
        return sample

# =================
# Probe definitions
# =================
class LinearClassificationProbe(nn.Module):
    def __init__(self, hidden_dim: int, class_size: int)  -> None:
        '''
        Classifies whether we are at the goal or not.
            :param hidden_dim: The dimensionality of the hidden layer of the probe.
            :param class_size: The number of classes in the probe. (2 for yes/no are we at goal)
            :return: None
        '''
        super().__init__()
        # The probe is a simple linear classifier, with a hidden layer and an output layer.
        # The input to the probe is the embeddings from the model, and the output is the predicted class.
        self.probe = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, class_size),
            torch.nn.Sigmoid() # convert to probability distribution over the classes
        )
    def forward(self, x: torch.Tensor):
        return self.probe(x)
    
class LinearRegressionProbe(nn.Module):
    """
    Predicts continuous observations on physics space.
    """
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.probe = torch.nn.Linear(hidden_dim, output_dim)
    def forward(self, x: torch.Tensor):
        return self.probe(x)

# =================
# Probe training
# =================
def train_probe(
    probe, # regression or classification probe
    train_loader,
    device,
    criterion,
    target_key,
    val_loader,
    epochs,
    lr=1e-3,
):
    """
    Trains a probes:
        LinearRegressionProbe: predicts continuous observations on physics space given encoded frames.
        LinearClassificationProbe: predicts whether we are at the goal or not given encoded frames.
    """
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    # train class probe
    for epoch in range(epochs):
        probe.train()
        train_loss = 0.0
        for batch in train_loader:
            z = batch["latent"].to(device)
            target = batch[target_key].to(device)

            pred = probe(z)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * z.size(0)

        train_loss /= len(train_loader.dataset)

        probe.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                z = batch["latent"].to(device)
                target = batch[target_key].to(device)
                pred = probe(z)
                loss = criterion(pred, target)
                val_loss += loss.item() * z.size(0)

        val_loss /= len(val_loader.dataset)

        if wandb.run is not None:
            wandb.log({
                "probe/train_loss": train_loss,
                "probe/val_loss": val_loss,
                "epoch": epoch,
            })

    return probe

def run_probe_training(
    cfg,
    encoder,
    device,
    epochs,
    batch_size,
    target_names=("x", "y", "z", "yaw", "pitch"),
):
    # load physics data from random malmo rollout video
    probe_dataset = MalmoProbeDataset(
        repo_path(cfg.paths.fixtures_dir, "random_malmo_001.pkl"),
        encoder=encoder,
        device=device,
    )

    # Load data
    # build regression targets separately from the raw physics stats (i.e. ground truth)
    targets = torch.stack([
        torch.tensor([float(stat[key]) for key in target_names], dtype=torch.float32)
        for stat in probe_dataset.stats
    ])

    num_samples = len(probe_dataset)
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(42))
    train_end = int(0.70 * num_samples)
    val_end = int(0.85 * num_samples)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # normalize targets using train-split statistics, then attach to the dataset
    target_mean = targets[train_idx].mean(dim=0, keepdim=True)
    target_std = targets[train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)
    probe_dataset.targets = (targets - target_mean) / target_std
    target_position = cfg.env.configs.multi_tree_navigation.target_position
    distances = torch.sqrt((targets[:, 0] - target_position[0]) ** 2 + (targets[:, 2] - target_position[1]) ** 2)
    probe_dataset.class_targets = (distances <= cfg.planner.success_distance).float().unsqueeze(1)

    train_loader = DataLoader(Subset(probe_dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(probe_dataset, val_idx.tolist()), batch_size=batch_size, shuffle=False)

    regression_probe = LinearRegressionProbe(
        hidden_dim=probe_dataset.latents.shape[-1],
        output_dim=len(target_names),
    ).to(device)
    trained_probe = train_probe(
        probe=regression_probe,
        train_loader=train_loader,
        device=device,
        criterion=nn.MSELoss(),
        target_key="target",
        val_loader=val_loader,
        epochs=epochs,
    )
    probe_path = repo_path(cfg.paths.final_models_dir, "linear_regression_probe.pt")
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_probe.state_dict(), probe_path)

    class_probe = LinearClassificationProbe(
        hidden_dim=probe_dataset.latents.shape[-1],
        class_size=1,
    ).to(device)
    trained_class_probe = train_probe(
        probe=class_probe,
        train_loader=train_loader,
        device=device,
        criterion=nn.BCELoss(),
        target_key="class_target",
        val_loader=val_loader,
        epochs=epochs,
    )
    class_probe_path = repo_path(cfg.paths.final_models_dir, "linear_classification_probe.pt")
    torch.save(trained_class_probe.state_dict(), class_probe_path)

    return trained_probe, trained_class_probe

# =================
# Probe evaluation
# =================
def evaluate_probe(
    probe: LinearRegressionProbe | LinearClassificationProbe,
    data_loader,
    device: torch.device,
    target_names,
    target_mean=None,
    target_std=None,
):
    """
    Evaluate based on pearson's correlation coefficient and MSE.
    """
    probe.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            z = batch["latent"].to(device)
            target = batch["target"].to(device)
            
            preds.append(probe(z).cpu())
            targets.append(target.cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    if target_mean is not None and target_std is not None:
        preds = preds * target_std + target_mean
        targets = targets * target_std + target_mean

    metrics = {}
    for idx, name in enumerate(target_names):
        pred_col = preds[:, idx]
        target_col = targets[:, idx]
        err = pred_col - target_col

        # Compute "predicted and ground truth quantities"
        pred_centered = pred_col - pred_col.mean()
        target_centered = target_col - target_col.mean()

        # Pearson's correlation coefficient measures the strength of the relationship
        # between the predicted continuous number and the true number.
        pearson = stats.pearsonr(pred_centered, target_centered)
        mse = err.square().mean().item()
        metrics[name] = {"mse": mse, "pearson_r": float(pearson.statistic)}

    return metrics

if __name__ == "__main__":
    cfg = OmegaConf.load("config/lewm.yaml")
    
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    checkpoint = torch.load(repo_path(cfg.paths.best_model), map_location="cpu")
    lewm_model = load_trained_lewm(cfg, checkpoint, device)
    encoder = lewm_model.encoder
    
    run_probe_training(
        cfg=cfg,
        encoder=encoder,
        device=device,
        epochs=cfg.probe.epochs,
        batch_size=cfg.probe.batch_size,
    )


import json

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from probes import (
    evaluate_probe,
    LinearClassificationProbe,
    LinearRegressionProbe,
    MalmoProbeDataset,
)
from lewm.paths import repo_path
from lewm.utils import load_trained_lewm


def plot_probe_predictions(eval_targets, preds, target_names, class_targets, class_preds, images_dir):
    """
    Generates 2 types of graphs: 
        Scatter plots of each target vs prediction for the regression probe
        Scatter plot of all regression predictions vs ground truth targets

    TODO: classification
    """
    plot_data = [
        (name, eval_targets[:, idx], preds[:, idx], images_dir / f"regression_{name}_scatter.png")
        for idx, name in enumerate(target_names)
    ]
    plot_data.append(("at_goal", class_targets[:, 0], class_preds[:, 0], images_dir / "classification_at_goal_scatter.png"))

    for name, target, pred, output_path in plot_data:
        fig, ax = plt.subplots()
        ax.scatter(target, pred, s=8, alpha=0.6)
        ax.set_xlabel("target")
        ax.set_ylabel("prediction")
        ax.set_title(f"Probe prediction: {name}")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (name, target, pred, _) in zip(axes.flatten(), plot_data):
        ax.scatter(target, pred, s=8, alpha=0.6)
        ax.set_xlabel("target")
        ax.set_ylabel("prediction")
        ax.set_title(f"Probe prediction: {name}")

    fig.tight_layout()
    fig.savefig(images_dir / "probe_combined_scatter.png")
    plt.close(fig)


@hydra.main(version_base=None, config_path="../config", config_name="lewm")
def run_evals(cfg: DictConfig):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint = torch.load(repo_path(cfg.paths.best_model), map_location="cpu")
    lewm_model = load_trained_lewm(cfg, checkpoint, device)

    # Load probe dataset
    target_names = ("x", "y", "z", "yaw", "pitch")
    probe_dataset = MalmoProbeDataset(
        repo_path(cfg.paths.fixtures_dir, "random_malmo_001.pkl"),
        encoder=lewm_model.encoder,
        device=device,
        batch_size=cfg.probe.batch_size,
    )

    # Build target values from ground truth physics stats
    targets = torch.stack([
        torch.tensor([float(stat[key]) for key in target_names], dtype=torch.float32)
        for stat in probe_dataset.stats
    ])
    
    # Split dataset
    num_samples = len(probe_dataset)
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(42))
    train_end = int(0.70 * num_samples)
    val_end = int(0.85 * num_samples)
    train_idx = indices[:train_end]
    val_idx = indices[val_end:]
    
    # pass in normalized targets to the probe
    # targets must be normalized at eval time since the probe was trained on normalized targets
    target_mean = targets[train_idx].mean(dim=0, keepdim=True)
    target_std = targets[train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)
    regression_targets = (targets - target_mean) / target_std
    regression_eval_data = {
        "latents": probe_dataset.latents[val_idx],
        "targets": regression_targets[val_idx],
    }

    # Load trained regression probe
    regression_probe = LinearRegressionProbe(hidden_dim=cfg.vit.embedding_dim, output_dim=len(target_names)).to(device)
    regression_probe.load_state_dict(torch.load(
        repo_path(cfg.paths.final_models_dir, "linear_regression_probe.pt"),
    ))

    # Evaluate regression probe
    regression_probe.eval()
    with torch.no_grad():
        regression_preds = regression_probe(regression_eval_data["latents"].to(device))
    metrics = {
        "regression": evaluate_probe(
            regression_probe,
            regression_eval_data,
            device,
            target_names,
            target_mean,
            target_std,
        ),
        # "classification": evaluate_probe(
        #     class_probe,
        #     [{"latent": class_eval_data["latents"], "target": class_eval_data["targets"]}],
        #     device,
        #     ("at_goal",),
        # ),
    }
    
    # TODO: classification probe
    # class_probe = LinearClassificationProbe(hidden_dim=cfg.vit.embedding_dim, class_size=1).to(device)
    # class_probe.load_state_dict(torch.load(
    #     repo_path(cfg.paths.final_models_dir, "linear_classification_probe.pt"),
    # ))

    # class_probe.eval()
    # with torch.no_grad():
    #     class_preds = class_probe(class_eval_data["latents"].to(device)).cpu()
    # class_targets = class_eval_data["targets"]

    # Save scatter plots
    images_dir = repo_path(cfg.paths.evals_dir, "probes")
    images_dir.mkdir(parents=True, exist_ok=True)
    regression_preds = regression_preds * target_std + target_mean
    plot_probe_predictions(regression_eval_data["targets"], regression_preds, target_names, images_dir)

    metrics_path = repo_path(cfg.paths.evals_dir, "probes", "probe_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as file:
        json.dump({"metrics": metrics}, file, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run_evals()
#!/usr/bin/env python3
"""
VoE (Violation of Expectation) Surprise Metrics Script
Computes prediction error metrics for the REAL LeWM model on test data.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import json
import argparse
import h5py
from dataclasses import dataclass
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lewm.paths import repo_path
from lewm.models.lewm import LeWM

try:
    from omegaconf import OmegaConf
except ImportError:
    print("Warning: omegaconf not available, using hardcoded defaults")
    OmegaConf = None

try:
    import stable_worldmodel as swm
    STABLE_WM_AVAILABLE = True
except ImportError:
    STABLE_WM_AVAILABLE = False
    print("Warning: stable_worldmodel not available, using mock data")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

@dataclass
class VoEMetrics:
    mean_prediction_error: float
    median_prediction_error: float
    std_prediction_error: float
    max_prediction_error: float
    percentile_95_error: float
    surprise_score: float
    total_frames_analyzed: int
    prediction_variance: float

def load_real_lewm_model(device: str) -> nn.Module:
    """
    Load the real LeWM model trained by train.py.
    """
    print("🔧 Initializing real LeWM model...")
    
    # Load hyperparameters from config/lewm.yaml
    try:
        if OmegaConf is not None:
            cfg_path = repo_path("config/lewm.yaml")
            cfg = OmegaConf.load(cfg_path)
            sigreg_cfg = getattr(cfg, "sigreg", {})
            num_proj = sigreg_cfg.get("num_proj", 1)
            factor = sigreg_cfg.get("factor", 1.0)
            phi = sigreg_cfg.get("phi", 1.0)
        else:
            raise FileNotFoundError
    except Exception:
        print("Could not load config/lewm.yaml, using default hyperparameters.")
        # Fallback hardcoded values based on train.py
        class Cfg: pass
        cfg = Cfg()
        cfg.vit = Cfg(); cfg.vit.image_size=64; cfg.vit.patch_size=8; cfg.vit.embedding_dim=192
        cfg.vit.num_channels=3; cfg.vit.num_patches=64; cfg.vit.attention_heads=3
        cfg.vit.mlp_hidden_nodes=768; cfg.vit.transformer_blocks=12
        cfg.predictor = Cfg(); cfg.predictor.attention_heads=16; cfg.predictor.transformer_blocks=6
        cfg.predictor.dropout=0.1; cfg.predictor.history_len=8
        cfg.action_dim = 10
        num_proj, factor, phi = 1, 1.0, 1.0

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
        num_proj=num_proj,
        factor=factor,
        phi=phi
    ).to(device)

    # Load weights from best_model.pt
    model_path = repo_path("artifacts/checkpoints/best_model.pt")
    if model_path.exists():
        print(f"✅ Loading trained weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        lewm.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        print(f"⚠️ Warning: Could not find {model_path}. Using untrained LeWM model.")
    
    lewm.eval()
    return lewm

def collect_trajectory(env_type: str, seed: int, num_samples: int, seq_len: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect a trajectory of pixels and actions.
    """
    if env_type == "dataset":
        print(f"📂 Loading trajectory from mineRL_training dataset...")
        
        # Use the exact path you confirmed exists
        h5_path = repo_path("data/mineRL_training.h5")
        if not h5_path.exists():
            raise FileNotFoundError(f"Could not find dataset at {h5_path}")
            
        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            print(f"   -> Found HDF5 keys: {keys}")
            
            # Robustly find the observation and action keys regardless of exact naming
            obs_key = next((k for k in keys if "obs" in k.lower() or "pixel" in k.lower()), keys[0])
            act_key = next((k for k in keys if "act" in k.lower()), keys[1] if len(keys) > 1 else keys[0])
            
            print(f"   -> Using keys: observations='{obs_key}', actions='{act_key}'")
            
            obs_data = f[obs_key]
            act_data = f[act_key]
            
            # Handle both flat datasets (N, C, H, W) and pre-chunked sequences (N, seq_len, C, H, W)
            if obs_data.ndim == 5: 
                # Data is already chunked into sequences
                indices = np.random.choice(obs_data.shape[0], min(num_samples, obs_data.shape[0]), replace=False)
                batch_images = torch.tensor(np.array(obs_data[indices]))
                batch_actions = torch.tensor(np.array(act_data[indices]))
            else: 
                # Data is flat, we need to extract subsequences of length seq_len
                total_frames = obs_data.shape[0]
                max_start = max(1, total_frames - seq_len)
                start_indices = np.random.randint(0, max_start, min(num_samples, max_start))
                batch_images, batch_actions = [], []
                
                for start in start_indices:
                    batch_images.append(torch.tensor(np.array(obs_data[start:start+seq_len])))
                    batch_actions.append(torch.tensor(np.array(act_data[start:start+seq_len])))
                    
                batch_images = torch.stack(batch_images)
                batch_actions = torch.stack(batch_actions)
                
        print(f"✅ Successfully loaded {batch_images.shape[0]} trajectories of length {seq_len}.")
        return batch_images, batch_actions
        
    else:
        print(f"🎲 Generating simulated trajectory for '{env_type}' (Seed: {seed})...")
        # Simulate data shapes: (num_samples, seq_len, C, H, W)
        images = torch.randn(num_samples, seq_len, 3, 64, 64)
        actions = torch.randn(num_samples, seq_len, 10)
        return images, actions

def compute_voe_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> VoEMetrics:
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    gt_flat = ground_truth.reshape(-1, ground_truth.shape[-1])
    errors = np.linalg.norm(pred_flat - gt_flat, axis=1)

    return VoEMetrics(
        mean_prediction_error=float(np.mean(errors)),
        median_prediction_error=float(np.median(errors)),
        std_prediction_error=float(np.std(errors)),
        max_prediction_error=float(np.max(errors)),
        percentile_95_error=float(np.percentile(errors, 95)),
        surprise_score=float(np.mean(errors)), # Standard VoE definition
        total_frames_analyzed=len(errors),
        prediction_variance=float(np.var(errors)),
    )

def run_voe_analysis(env_type: str, seed: int, num_samples: int = 100, batch_size: int = 32, device: str = "cpu", use_wandb: bool = False) -> Dict:
    print(f"\n🎮 VoE Surprise Metrics Analysis (Real Model)")
    print(f"Environment: {env_type} | Seed: {seed}")
    print(f"Device: {device}")
    print("-" * 60)

    lewm = load_real_lewm_model(device)
    images, actions = collect_trajectory(env_type, seed, num_samples, seq_len=10)
    
    all_predictions, all_ground_truth = [], []
    total_samples = images.shape[0]
    
    with torch.no_grad():
        for batch_idx in range(0, total_samples, batch_size):
            batch_end = min(batch_idx + batch_size, total_samples)
            batch_images = images[batch_idx:batch_end].to(device)
            batch_actions = actions[batch_idx:batch_end].to(device)
            
            # Forward pass through the REAL LeWM model
            model_out = lewm(batch_images, batch_actions)
            
            # model_out[0] is predicted next embedding, model_out[1] is actual next embedding
            all_predictions.append(model_out[0].cpu().numpy())
            all_ground_truth.append(model_out[1].cpu().numpy())
            
    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)
    
    print(f"\n🔍 Analyzed {predictions.shape[0] * predictions.shape[1]} frame transitions")
    metrics = compute_voe_metrics(predictions, ground_truth)
    
    results = {
        "env_type": env_type, "seed": str(seed),
        "voe_metrics": {
            "mean_prediction_error": metrics.mean_prediction_error,
            "median_prediction_error": metrics.median_prediction_error,
            "std_prediction_error": metrics.std_prediction_error,
            "max_prediction_error": metrics.max_prediction_error,
            "percentile_95_error": metrics.percentile_95_error,
            "surprise_score": metrics.surprise_score,
            "prediction_variance": metrics.prediction_variance,
            "total_frames_analyzed": metrics.total_frames_analyzed,
        }
    }
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project="le-wm-voe-metrics", name=f"VoE_{env_type}_seed_{seed}", config={"env_type": env_type, "seed": seed})
        wandb.log(results["voe_metrics"])
        print(f"\n[✅] WandB Run Complete")
        wandb.finish()
    else:
        print("\n📊 VoE SURPRISE METRICS RESULTS:")
        print(json.dumps(results, indent=4))
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VoE with real LeWM model")
    parser.add_argument("--env_type", type=str, default="forest", choices=["forest", "superflat", "dataset"])
    parser.add_argument("--seed", type=int, default=-2744534680298546054)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = run_voe_analysis(
        env_type=args.env_type, seed=args.seed, num_samples=args.num_samples,
        batch_size=32, device=device, use_wandb=args.use_wandb
    )
    
    output_path = repo_path("outputs") / f"voe_metrics_{args.env_type}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
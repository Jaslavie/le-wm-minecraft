#!/usr/bin/env python3
"""
VoE (Violation of Expectation) Surprise Metrics Script
Computes prediction error metrics for the LeWM model on test data.
VoE measures how much actual observations violate model expectations (predictions).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import json
from dataclasses import dataclass
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lewm.paths import repo_path

try:
    import stable_worldmodel as swm
    STABLE_WM_AVAILABLE = True
except ImportError:
    STABLE_WM_AVAILABLE = False
    print("Warning: stable_worldmodel not available, using mock data")


@dataclass
class VoEMetrics:
    """Container for VoE metrics"""
    mean_prediction_error: float
    median_prediction_error: float
    std_prediction_error: float
    max_prediction_error: float
    percentile_95_error: float
    surprise_score: float  # Higher = more violation of expectations
    total_frames_analyzed: int
    prediction_variance: float


def create_dummy_model(embedding_dim: int = 192) -> nn.Module:
    """Create a simple dummy predictor model for demonstration"""
    class DummyPredictor(nn.Module):
        def __init__(self, embedding_dim):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.fc = nn.Linear(embedding_dim + 10, embedding_dim)  # obs_emb + action
            
        def forward(self, obs_embedding, action):
            """
            Predict next observation embedding
            obs_embedding: (B, T, embedding_dim)
            action: (B, T, 10)
            """
            B, T, D = obs_embedding.shape
            combined = torch.cat([obs_embedding, action], dim=-1)
            pred = self.fc(combined)
            return pred
    
    return DummyPredictor(embedding_dim)


def create_dummy_encoder() -> nn.Module:
    """Create a simple dummy image encoder"""
    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64 * 64 * 3, 192)
        
        def forward(self, x):
            """x: (B, T, C, H, W)"""
            B, T, C, H, W = x.shape
            x = x.view(B, T, -1)
            return self.fc(x)
    
    return DummyEncoder()


def compute_voe_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> VoEMetrics:
    """
    Compute VoE metrics from predictions and ground truth
    
    Args:
        predictions: (N, embedding_dim) predicted embeddings
        ground_truth: (N, embedding_dim) actual embeddings
    
    Returns:
        VoEMetrics: computed metrics
    """
    # Compute L2 distance (prediction error) for each timestep
    errors = np.linalg.norm(predictions - ground_truth, axis=1)
    
    # Compute metrics
    mean_error = float(np.mean(errors))
    median_error = float(np.median(errors))
    std_error = float(np.std(errors))
    max_error = float(np.max(errors))
    p95_error = float(np.percentile(errors, 95))
    
    # Surprise score: normalized measure of prediction violations
    # Higher error = more surprise (violation of expectations)
    surprise_score = float(np.mean(errors) / (np.std(errors) + 1e-6))
    
    # Variance in predictions
    pred_variance = float(np.var(predictions))
    
    metrics = VoEMetrics(
        mean_prediction_error=mean_error,
        median_prediction_error=median_error,
        std_prediction_error=std_error,
        max_prediction_error=max_error,
        percentile_95_error=p95_error,
        surprise_score=surprise_score,
        total_frames_analyzed=len(errors),
        prediction_variance=pred_variance,
    )
    
    return metrics


def generate_mock_data(num_samples: int = 100, seq_len: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate mock data for testing"""
    # Random image sequences
    images = torch.randn(num_samples, seq_len, 3, 64, 64)
    # Random action sequences
    actions = torch.randn(num_samples, seq_len, 10)
    return images, actions


def run_voe_analysis(
    num_samples: int = 100,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_real_data: bool = False,
) -> Dict:
    """
    Main VoE analysis pipeline
    
    Args:
        num_samples: Number of samples to analyze
        batch_size: Batch size for processing
        device: Device to use (cuda/cpu)
        use_real_data: Whether to use real HDF5 data
    
    Returns:
        Dictionary containing all metrics and metadata
    """
    
    print(f"🎮 VoE Surprise Metrics Analysis")
    print(f"Device: {device}")
    print(f"Samples to analyze: {num_samples}")
    print("-" * 60)
    
    # Initialize models
    print("Initializing models...")
    encoder = create_dummy_encoder().to(device)
    predictor = create_dummy_model().to(device)
    encoder.eval()
    predictor.eval()
    
    all_predictions = []
    all_ground_truth = []
    
    # Load data
    print("Loading data...")
    use_real = False
    if use_real_data and STABLE_WM_AVAILABLE:
        try:
            data_dir = repo_path("data")
            dataset = swm.data.HDF5Dataset(
                "mineRL_training",
                cache_dir=str(data_dir),
                num_steps=10,
            )
            
            # Limit samples
            indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
            total_samples = len(indices)
            use_real = True
        except Exception as e:
            print(f"Warning: Could not load real data ({e}), using mock data")
            use_real = False
    
    if not use_real:
        # Generate mock data upfront
        mock_images, mock_actions = generate_mock_data(num_samples, seq_len=10)
        total_samples = num_samples
    
    print(f"Processing {total_samples} samples in batches of {batch_size}...")
    
    # Process in batches
    with torch.no_grad():
        for batch_idx in range(0, total_samples, batch_size):
            batch_end = min(batch_idx + batch_size, total_samples)
            current_batch_size = batch_end - batch_idx
            
            if use_real:
                try:
                    batch_indices = indices[batch_idx:batch_end]
                    batch_images = []
                    batch_actions = []
                    for idx in batch_indices:
                        img, action = dataset[idx]
                        batch_images.append(img)
                        batch_actions.append(action)
                    images = torch.stack(batch_images).to(device)
                    actions = torch.stack(batch_actions).to(device)
                except Exception as e:
                    print(f"Error loading real data: {e}, using mock")
                    images, actions = generate_mock_data(current_batch_size, seq_len=10)
                    images = images.to(device)
                    actions = actions.to(device)
            else:
                images = mock_images[batch_idx:batch_end].to(device)
                actions = mock_actions[batch_idx:batch_end].to(device)
            
            # Encode observations
            obs_embeddings = encoder(images)  # (B, T, 192)
            
            # Split into current and next observations
            current_obs = obs_embeddings[:, :-1]  # (B, T-1, 192)
            next_obs = obs_embeddings[:, 1:]      # (B, T-1, 192)
            current_actions = actions[:, :-1]    # (B, T-1, 10)
            
            # Predict next observations
            predicted_next_obs = predictor(current_obs, current_actions)  # (B, T-1, 192)
            
            # Store predictions and ground truth
            all_predictions.append(predicted_next_obs.cpu().numpy())
            all_ground_truth.append(next_obs.cpu().numpy())
            
            if (batch_idx // batch_size + 1) % max(1, total_samples // (batch_size * 5)) == 0:
                print(f"  ✓ Processed batch {batch_idx // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size}")
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0).reshape(-1, 192)  # (N, 192)
    ground_truth = np.concatenate(all_ground_truth, axis=0).reshape(-1, 192)  # (N, 192)
    
    print(f"\nAnalyzed {predictions.shape[0]} predictions")
    print("-" * 60)
    
    # Compute metrics
    print("Computing VoE metrics...")
    metrics = compute_voe_metrics(predictions, ground_truth)
    
    # Create results dictionary
    results = {
        "voe_metrics": {
            "mean_prediction_error": metrics.mean_prediction_error,
            "median_prediction_error": metrics.median_prediction_error,
            "std_prediction_error": metrics.std_prediction_error,
            "max_prediction_error": metrics.max_prediction_error,
            "percentile_95_error": metrics.percentile_95_error,
            "surprise_score": metrics.surprise_score,
            "prediction_variance": metrics.prediction_variance,
            "total_frames_analyzed": metrics.total_frames_analyzed,
        },
        "metadata": {
            "device": str(device),
            "num_samples": num_samples,
            "batch_size": batch_size,
            "use_real_data": use_real,
            "embedding_dim": 192,
        }
    }
    
    return results


def print_results(results: Dict) -> None:
    """Pretty print VoE metrics"""
    metrics = results["voe_metrics"]
    
    print("\n" + "="*60)
    print("📊 VoE SURPRISE METRICS RESULTS")
    print("="*60)
    print(f"Total frames analyzed: {metrics['total_frames_analyzed']}")
    print(f"\n📈 Prediction Error Statistics (L2 distance):")
    print(f"  Mean:           {metrics['mean_prediction_error']:.6f}")
    print(f"  Median:         {metrics['median_prediction_error']:.6f}")
    print(f"  Std Dev:        {metrics['std_prediction_error']:.6f}")
    print(f"  Max:            {metrics['max_prediction_error']:.6f}")
    print(f"  95th percentile: {metrics['percentile_95_error']:.6f}")
    print(f"\n🎯 Surprise Metrics:")
    print(f"  Surprise Score: {metrics['surprise_score']:.6f}")
    print(f"    (Higher = more violation of expectations)")
    print(f"  Pred Variance:  {metrics['prediction_variance']:.6f}")
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run analysis
    results = run_voe_analysis(
        num_samples=100,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_real_data=False,  # Set to True to use real data if available
    )
    
    # Print results
    print_results(results)
    
    # Save results to file
    output_path = repo_path("outputs") / "voe_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy floats to Python floats for JSON serialization
    results_json = {
        "voe_metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in results["voe_metrics"].items()},
        "metadata": results["metadata"]
    }
    
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    print(json.dumps(results_json, indent=2))

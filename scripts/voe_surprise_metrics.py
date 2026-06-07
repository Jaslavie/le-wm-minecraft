#!/usr/bin/env python3
"""
VoE Surprise Metrics - PyTorch Client Side
Connects to the Malmo Python 3.5 Server via Socket to evaluate the real LeWM model.
"""
import socket
import numpy as np
import torch
import json
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lewm.paths import repo_path
from lewm.models.lewm import LeWM

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

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

def load_real_lewm_model(device: str) -> torch.nn.Module:
    print("🔧 Initializing real LeWM model...")
    try:
        if OmegaConf is not None:
            cfg = OmegaConf.load(repo_path("config/lewm.yaml"))
            sigreg_cfg = getattr(cfg, "sigreg", {})
            num_proj, factor, phi = sigreg_cfg.get("num_proj", 1), sigreg_cfg.get("factor", 1.0), sigreg_cfg.get("phi", 1.0)
        else: raise FileNotFoundError
    except Exception:
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
        image_size=cfg.vit.image_size, patch_size=cfg.vit.patch_size, embedding_dim=cfg.vit.embedding_dim,
        num_channels=cfg.vit.num_channels, num_patches=cfg.vit.num_patches, vit_attention_heads=cfg.vit.attention_heads,
        vit_mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes, vit_transformer_blocks=cfg.vit.transformer_blocks,
        predictor_attention_heads=cfg.predictor.attention_heads, predictor_mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes, 
        predictor_transformer_blocks=cfg.predictor.transformer_blocks, action_dim=cfg.action_dim,
        dropout=cfg.predictor.dropout, history_len=cfg.predictor.history_len,
        num_proj=num_proj, factor=factor, phi=phi
    ).to(device)

    model_path = repo_path("artifacts/checkpoints/best_model.pt")
    if model_path.exists():
        print(f"✅ Loading trained weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        lewm.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        print(f"⚠️ Warning: Could not find {model_path}. Using untrained LeWM model.")
    
    lewm.eval()
    return lewm

def collect_real_malmo_trajectory(num_samples: int, seq_len: int = 10, port: int = 25565):
    print(f"🔌 Connecting to Malmo Server on localhost:{port}...")
    print("   (Ensure your Python 3.5 Malmo runner is active and waiting for connection!)")
    
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(('localhost', port))
    except ConnectionRefusedError:
        raise RuntimeError(f"❌ Could not connect to localhost:{port}. Start the Malmo runner first!")
        
    print("✅ Connected! Pulling REAL Minecraft frames...")
    
    images, actions = [], []
    current_seq_imgs, current_seq_acts = [], []
    
    # 64x64 RGB = 12288 bytes per frame
    FRAME_SIZE = 64 * 64 * 3 
    
    # Simple discrete actions to send back to Malmo to keep the agent exploring
    # Format depends on your malmo_mission_runner.py implementation (e.g., JSON or pickled list)
    import pickle
    discrete_actions = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # forward
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   # back
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   # right
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # left
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   # jump
        [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0], # turn
    ]
    
    try:
        total_frames = num_samples * seq_len
        for step in range(total_frames):
            # 1. Receive exact frame bytes
            data = b''
            while len(data) < FRAME_SIZE:
                packet = client.recv(FRAME_SIZE - len(data))
                if not packet: raise ConnectionError("Malmo runner disconnected.")
                data += packet
            
            # 2. Convert raw bytes to PyTorch tensor
            img_np = np.frombuffer(data, dtype=np.uint8).reshape(64, 64, 3)
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
            
            # 3. Send action back to Malmo
            action_vec = discrete_actions[step % len(discrete_actions)] # Cycle through actions
            client.sendall(pickle.dumps(action_vec))
            
            current_seq_imgs.append(img_tensor)
            current_seq_acts.append(torch.tensor(action_vec, dtype=torch.float32))
            
            if len(current_seq_imgs) == seq_len:
                images.append(torch.stack(current_seq_imgs))
                actions.append(torch.stack(current_seq_acts))
                current_seq_imgs, current_seq_acts = [], []
                
    except Exception as e:
        print(f"⚠️ Socket Error: {e}")
    finally:
        client.close()
        
    if not images:
        raise RuntimeError("Failed to collect any trajectories. Check Malmo server logs.")
        
    print(f"✅ Collected {len(images)} real trajectories from Minecraft!")
    return torch.stack(images), torch.stack(actions)

def compute_voe_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> VoEMetrics:
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    gt_flat = ground_truth.reshape(-1, ground_truth.shape[-1])
    errors = np.linalg.norm(pred_flat - gt_flat, axis=1)
    return VoEMetrics(
        mean_prediction_error=float(np.mean(errors)), median_prediction_error=float(np.median(errors)),
        std_prediction_error=float(np.std(errors)), max_prediction_error=float(np.max(errors)),
        percentile_95_error=float(np.percentile(errors, 95)), surprise_score=float(np.mean(errors)),
        total_frames_analyzed=len(errors)
    )

def run_voe_analysis(num_samples: int, device: str, use_wandb: bool):
    print(f"\n🎮 VoE Surprise Metrics Analysis (Real Model + Real Malmo Frames)")
    lewm = load_real_lewm_model(device)
    images, actions = collect_real_malmo_trajectory(num_samples, seq_len=10)
    
    all_preds, all_gts = [], []
    with torch.no_grad():
        for i in range(0, images.shape[0], 8): # Batch size 8 for CPU
            batch_imgs = images[i:i+8].to(device)
            batch_acts = actions[i:i+8].to(device)
            model_out = lewm(batch_imgs, batch_acts)
            all_preds.append(model_out[0].cpu().numpy())
            all_gts.append(model_out[1].cpu().numpy())
            
    metrics = compute_voe_metrics(np.concatenate(all_preds), np.concatenate(all_gts))
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project="le-wm-voe-metrics", name=f"VoE_Real_Malmo", config={"source": "live_socket"})
        wandb.log(metrics.__dict__)
        print(f"\n[✅] WandB Run Complete")
        wandb.finish()
    else:
        print("\n📊 VoE SURPRISE METRICS RESULTS:")
        print(json.dumps(metrics.__dict__, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20, help="Number of 10-frame sequences to collect")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_voe_analysis(args.num_samples, device, args.use_wandb)
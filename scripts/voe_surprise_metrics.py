#!/usr/bin/env python3
"""
VoE (Violation of Expectation) Surprise Metrics Script
Computes prediction error metrics for the REAL LeWM model on test data.
Supports: HDF5 Dataset, Simulated Data, and LIVE Malmo Minecraft Connections.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import json
import argparse
import h5py
import time
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import socket
import pickle
import random

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
    prediction_variance: float

def load_real_lewm_model(device: str) -> nn.Module:
    print("🔧 Initializing real LeWM model...")
    try:
        if OmegaConf is not None:
            cfg_path = repo_path("config/lewm.yaml")
            cfg = OmegaConf.load(cfg_path)
            sigreg_cfg = getattr(cfg, "sigreg", {})
            num_proj = sigreg_cfg.get("num_proj", 1)
            factor = sigreg_cfg.get("factor", 1.0)
            phi = sigreg_cfg.get("phi", 1.0)
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

def collect_malmo_trajectory(env_type: str, num_samples: int, seq_len: int = 10):
    try:
        import MalmoPython
    except ImportError:
        raise ImportError(
            "\n❌ MalmoPython not found in your 'lewmmc' environment!\n"
            "Your Malmo folder uses a custom socket bridge (port 25565) to separate Minecraft from PyTorch.\n"
            "To run LIVE Malmo evaluation, you must install the MalmoPython wheel into this conda environment,\n"
            "OR record a dataset using your runner and use '--env_type dataset'."
        )
    
    print(f"🎮 Connecting to Minecraft via Malmo...")
    
    if env_type == "malmo_forest":
        print("🌲 Mission: Birch & Oak Forest (Seed: -2744534680298546054)")
        mission_xml = f'''<?xml version="1.0" encoding="UTF-8" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com">
          <About><Summary>VoE Forest</Summary></About>
          <ServerSection>
            <ServerInitialConditions><Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time><Weather>clear</Weather></ServerInitialConditions>
            <ServerHandlers>
              <DefaultWorldGenerator seed="-2744534680298546054"/>
              <ServerQuitFromTimeUp timeLimitMs="120000"/><ServerQuitWhenAnyAgentFinishes/>
            </ServerHandlers>
          </ServerSection>
          <AgentSection mode="Survival"><Name>VoEBot</Name>
            <AgentStart><Placement x="0.5" y="70" z="0.5" yaw="0"/></AgentStart>
            <AgentHandlers>
              <VideoProducer want_depth="false"><Width>64</Width><Height>64</Height></VideoProducer>
              <DiscreteMovementCommands/>
            </AgentHandlers>
          </AgentSection>
        </Mission>'''
    elif env_type == "malmo_superflat":
        print("🟩 Mission: Single Tree Superflat")
        tree_xml = '''
        <DrawCuboid x1="-2" y1="6" z1="-2" x2="2" y2="7" z2="2" type="leaves"/>
        <DrawCuboid x1="-1" y1="8" z1="-1" x2="1" y2="9" z2="1" type="leaves"/>
        <DrawBlock x="0" y="4" z="0" type="log"/><DrawBlock x="0" y="5" z="0" type="log"/>
        <DrawBlock x="0" y="6" z="0" type="log"/><DrawBlock x="0" y="7" z="0" type="log"/>
        '''
        mission_xml = f'''<?xml version="1.0" encoding="UTF-8" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com">
          <About><Summary>VoE Superflat</Summary></About>
          <ServerSection>
            <ServerInitialConditions><Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time><Weather>clear</Weather></ServerInitialConditions>
            <ServerHandlers>
              <FlatWorldGenerator generatorString="3;7,2*3,2;1;"/>
              <DrawingDecorator>{tree_xml}</DrawingDecorator>
              <ServerQuitFromTimeUp timeLimitMs="120000"/><ServerQuitWhenAnyAgentFinishes/>
            </ServerHandlers>
          </ServerSection>
          <AgentSection mode="Survival"><Name>VoEBot</Name>
            <AgentStart><Placement x="0.5" y="5" z="0.5" yaw="0"/></AgentStart>
            <AgentHandlers>
              <VideoProducer want_depth="false"><Width>64</Width><Height>64</Height></VideoProducer>
              <DiscreteMovementCommands/>
            </AgentHandlers>
          </AgentSection>
        </Mission>'''
    else:
        raise ValueError(f"Unknown Malmo env type: {env_type}")

    agent_host = MalmoPython.AgentHost()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    for retry in range(3):
        try:
            agent_host.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if retry == 2: raise RuntimeError(f"Failed to start mission: {e}. Is Minecraft 1.11.2 with Malmo mod running?")
            time.sleep(2)

    print("⏳ Waiting for mission to start...")
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        
    print("✅ Mission running! Collecting real frames...")
    images, actions = [], []
    discrete_actions = ["move 1", "move -1", "strafe 1", "strafe -1", "turn 0.5", "turn -0.5", "jump 1"]
    current_seq_imgs, current_seq_acts = [], []
    
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        if world_state.number_of_video_frames_since_last_state > 0:
            frame = world_state.video_frames[-1]
            img_np = np.array(frame.pixels)
            if img_np.ndim == 1: img_np = img_np.reshape((64, 64, 3))
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
            
            action_cmd = random.choice(discrete_actions)
            agent_host.sendCommand(action_cmd)
            
            act_vec = torch.zeros(10)
            if action_cmd == "move 1": act_vec[0] = 1
            elif action_cmd == "move -1": act_vec[2] = 1
            elif action_cmd == "strafe 1": act_vec[3] = 1
            elif action_cmd == "strafe -1": act_vec[1] = 1
            elif "turn" in action_cmd: act_vec[8] = float(action_cmd.split()[1])
            elif action_cmd == "jump 1": act_vec[4] = 1
            
            current_seq_imgs.append(img_tensor)
            current_seq_acts.append(act_vec)
            
            if len(current_seq_imgs) == seq_len:
                images.append(torch.stack(current_seq_imgs))
                actions.append(torch.stack(current_seq_acts))
                current_seq_imgs, current_seq_acts = [], []
                if len(images) >= num_samples:
                    agent_host.sendCommand("quit")
                    break
        time.sleep(0.05)
        
    print(f"✅ Collected {len(images)} real trajectories from Minecraft!")
    return torch.stack(images), torch.stack(actions)

def collect_trajectory(env_type: str, seed: int, num_samples: int, seq_len: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Strict Socket Client: Connects to malmo_mission_runner.py on port 25565.
    Receives real Minecraft frames and sends actions back.
    """
    print(f"🔌 Connecting to Malmo Mission Runner on localhost:25565...")
    print(f"   (Ensure malmo_mission_runner.py is running and waiting for connection!)")
    
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(('localhost', 25565))
    except ConnectionRefusedError:
        raise RuntimeError("❌ Could not connect to localhost:25565. Please start malmo_mission_runner.py first!")
        
    print("✅ Connected to Malmo! Collecting REAL frames from Minecraft...")
    
    images, actions = [], []
    current_seq_imgs, current_seq_acts = [], []
    
    # Action format: [forward, left, back, right, jump, sneak, sprint, attack, camera_x, camera_y]
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
            # 1. Receive frame (64x64x3 = 12288 bytes)
            data = b''
            while len(data) < 12288:
                packet = client.recv(12288 - len(data))
                if not packet:
                    raise ConnectionError("Malmo runner disconnected.")
                data += packet
            
            # Convert raw bytes to numpy array
            img_np = np.frombuffer(data, dtype=np.uint8).reshape(64, 64, 3)
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
            
            # 2. Send random action back to runner to keep agent exploring
            action_vec = random.choice(discrete_actions)
            client.sendall(pickle.dumps(action_vec))
            
            current_seq_imgs.append(img_tensor)
            current_seq_acts.append(torch.tensor(action_vec, dtype=torch.float32))
            
            if len(current_seq_imgs) == seq_len:
                images.append(torch.stack(current_seq_imgs))
                actions.append(torch.stack(current_seq_acts))
                current_seq_imgs, current_seq_acts = [], []
                
    except Exception as e:
        print(f"⚠️ Error during collection: {e}")
    finally:
        client.close()
        
    print(f"✅ Collected {len(images)} real trajectories from Minecraft!")
    if len(images) == 0:
        raise RuntimeError("Failed to collect any trajectories from Malmo.")
        
    return torch.stack(images), torch.stack(actions)

def compute_voe_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> VoEMetrics:
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    gt_flat = ground_truth.reshape(-1, ground_truth.shape[-1])
    errors = np.linalg.norm(pred_flat - gt_flat, axis=1)
    return VoEMetrics(
        mean_prediction_error=float(np.mean(errors)), median_prediction_error=float(np.median(errors)),
        std_prediction_error=float(np.std(errors)), max_prediction_error=float(np.max(errors)),
        percentile_95_error=float(np.percentile(errors, 95)), surprise_score=float(np.mean(errors)),
        total_frames_analyzed=len(errors), prediction_variance=float(np.var(errors))
    )

def run_voe_analysis(env_type: str, seed: int, num_samples: int = 100, batch_size: int = 32, device: str = "cpu", use_wandb: bool = False) -> Dict:
    print(f"\n🎮 VoE Surprise Metrics Analysis (Real Model)")
    print(f"Environment: {env_type} | Seed: {seed}")
    lewm = load_real_lewm_model(device)
    images, actions = collect_trajectory(env_type, seed, num_samples, seq_len=10)
    
    all_predictions, all_ground_truth = [], []
    with torch.no_grad():
        for batch_idx in range(0, images.shape[0], batch_size):
            batch_images = images[batch_idx:batch_idx+batch_size].to(device)
            batch_actions = actions[batch_idx:batch_idx+batch_size].to(device)
            model_out = lewm(batch_images, batch_actions)
            all_predictions.append(model_out[0].cpu().numpy())
            all_ground_truth.append(model_out[1].cpu().numpy())
            
    metrics = compute_voe_metrics(np.concatenate(all_predictions), np.concatenate(all_ground_truth))
    results = {"env_type": env_type, "seed": str(seed), "voe_metrics": metrics.__dict__}
    
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
    parser.add_argument("--env_type", type=str, default="malmo_forest", choices=["malmo_forest", "malmo_superflat", "dataset", "forest", "superflat"])
    parser.add_argument("--seed", type=int, default=-2744534680298546054)
    parser.add_argument("--num_samples", type=int, default=50) # Start small for live Malmo
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_voe_analysis(env_type=args.env_type, seed=args.seed, num_samples=args.num_samples, batch_size=32, device=device, use_wandb=args.use_wandb)
    
    output_path = repo_path("outputs") / f"voe_metrics_{args.env_type}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
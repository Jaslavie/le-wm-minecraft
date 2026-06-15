"""
TreeChop benchmark on baselines and the trained LeWM model.

Stages
    CHOP: chop goal image (tree trunk) reached within MSE threshold
    NAV: nav goal image (tree) reached within MSE threshold
    SUCCESS: CHOP and NAV stages reached

Run:
    python evals/benchmark.py
"""
import sys
import json
import math
import pickle
import socket
import struct
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

try:
    import wandb
except ImportError:
    wandb = None

from lewm.paths import repo_path
from lewm.planning.planner import Planner
from lewm.utils import (
    get_cam_mean_std,
    load_trained_lewm,
    make_transform,
    planner_output_to_actions,
    process_frame_pixels,
    recvall,
)

sys.path.insert(0, str(repo_path("scripts")))
from run_lewm_client import run_evals

# models we will benchmark against each other
AGENT_REGISTRY = {
    "random": dict(kind="random", display="Random"),
    "lewm_no_invdyn": dict(kind="lewm", checkpoint="no_invdyn", display="LeWM (no invdyn)"),
    "lewm_invdyn": dict(kind="lewm", checkpoint="invdyn", display="LeWM (invdyn)"),
}

def connect_with_retry():
    """Wait for the Malmo mission server to run and connect socket to it."""
    deadline = time.time() + 120 # 2 minutes
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", 25565))
            return s
        except OSError:
            time.sleep(2)
    return None


def run_random_episode(conn, cfg, transform, enc, nav_goal_z, chop_goal_z, seed, device):
    """Random policy as the baseline test"""
    rng = np.random.default_rng(seed)
    stage = "NAV"
    chop_stage_count = 0
    step = 0
    while step < cfg.planner.max_steps:
        frame = recvall(conn, 64 * 64 * 3)
        if frame is None:
            break

        _ = recvall(conn, struct.unpack(">I", recvall(conn, 4))[0])
        obs = process_frame_pixels(transform, frame)
        
        # match success criteria of run_lewm_client
        # if stage == "NAV" and distance_to_tree < cfg.planner.success_distance:
        if stage == "NAV" and F.mse_loss(enc(obs.to(device)), nav_goal_z).item() < cfg.planner.nav_done_mse:
            stage = "CHOP"
        if stage == "CHOP":
            # start first chop
            chop_stage_count += 1 if F.mse_loss(enc(obs.to(device)), chop_goal_z).item() < cfg.planner.chop_done_mse else 0
            # continue chopping continuously for a minimum number of steps
            if chop_stage_count >= cfg.planner.chop_done_patience:
                stage = "SUCCESS"
        if stage == "SUCCESS":
            break
        
        # select random actions to move around
        action = np.zeros(cfg.action_dim)
        action[:8] = rng.integers(0, 2, 8)
        print(f"action selected for step {step}: {action}")
        conn.sendall(pickle.dumps(action.tolist()))
        step += 1
    
    return {
        "status": stage,
        "nav_success": stage in ("CHOP", "SUCCESS"),
        "chop_success": stage == "SUCCESS",
        "steps": step,
    }

# =============
# Plotting functions
# =============
def plot_scoreboard(summary, out_path):
    agents = list(summary)
    x = np.arange(len(agents))
    w = 0.38 # width of the bars
    nav = [summary[a]["nav_rate"] * 100 for a in agents]
    chop = [summary[a]["chop_rate"] * 100 for a in agents]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, nav, w, label="Navigation", color=C_NAV)
    ax.bar(x + w / 2, chop, w, label="Chop", color=C_CHOP)
    for i in range(len(agents)):
        ax.text(x[i] - w / 2, nav[i] + 1, f"{nav[i]:.0f}%", ha="center", fontsize=9)
        ax.text(x[i] + w / 2, chop[i] + 1, f"{chop[i]:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([summary[a]["display"] for a in agents])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"TreeChop success by agent (N={summary[agents[0]]['n']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_metrics_table(summary, out_path):
    agents = list(summary)
    headers = ["Metric"] + [summary[a]["display"] for a in agents]
    rows = [
        ["Nav success (%)"] + [f"{summary[a]['nav_rate'] * 100:.0f}" for a in agents],
        ["Chop success (%)"] + [f"{summary[a]['chop_rate'] * 100:.0f}" for a in agents],
        ["Min nav-MSE (latent)"] + [f"{summary[a]['nav_mse']:.3f}" for a in agents],
        ["N (episodes)"] + [str(summary[a]["n"]) for a in agents],
    ]
    fig, ax = plt.subplots(figsize=(2 + 2.2 * len(headers), 2.2))
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8f0e0")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)



def main():
    cfg = OmegaConf.load(repo_path("config", "lewm.yaml"))
    bm = cfg.benchmark
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Models
    lewm_model = load_trained_lewm(cfg, torch.load(repo_path(bm.models["invdyn"]), map_location="cpu"), device, strict=False)
    @torch.no_grad()
    def enc(obs):
        return lewm_model.encoder(obs.to(device)).view(-1)

    # Load and process goal frames
    transform = make_transform(cfg.vit.image_size)
    nav_goal = process_frame_pixels(transform, pickle.load(open(repo_path(cfg.paths.nav_goal_frame), "rb"))[0])
    chop_goal = process_frame_pixels(transform, pickle.load(open(repo_path(cfg.paths.chop_goal_frame), "rb"))[0])
    nav_goal_z = enc(nav_goal)
    chop_goal_z = enc(chop_goal)

    print(f"ENV: {bm.env} | EPISODES: {bm.episodes}")
    results = {}
    # process results for each model
    for name in bm.agents:
        print(f"======== Running {name} ==========")
        spec = AGENT_REGISTRY[name]
        eps = []
        for ep in range(bm.episodes):
            # Random policy rollout in malmo environment
            if spec["kind"] == "random":
                conn = connect_with_retry()
                if conn is None:
                    break
                try:
                    r = run_random_episode(conn, cfg, transform, enc, nav_goal_z, chop_goal_z, seed=ep, device=device)
                finally:
                    conn.close()
            # LeWM model rollout
            else:
                r = run_evals(repo_path(bm.models[spec["checkpoint"]]), bm.env, n_episodes=bm.episodes)
            
            # Add status to result dict
            r["nav_success"] = r["status"] in ("CHOP", "SUCCESS")
            r["chop_success"] = r["status"] == "SUCCESS"
            eps.append(r)
            
            print(
                f"  {name} | episode {ep + 1} | stage={r['status']} "
                f"nav_success={r['nav_success']} chop_success={r['chop_success']} nav_mse={r['nav_mse']}"
            )

        results[name] = eps

    # Report summary of results from each episode
    # Results is a list of success statuses from each episode
    summary = {}
    for name, eps in results.items():
        n = len(eps)
        nav_mses = [e["nav_mse"] for e in eps if e.get("nav_mse") is not None]
        summary[name] = {
            "n": n,
            # Success rates for each stage
            "nav_rate": sum(e.get("status") in ("CHOP", "SUCCESS") for e in eps) / n if n else 0.0,
            "chop_rate": sum(e.get("status") == "SUCCESS" for e in eps) / n if n else 0.0,
            "nav_mse": float(np.mean(nav_mses)) if nav_mses else float("nan"),
            "display": AGENT_REGISTRY[name]["display"],
        }

    # Store and plot results
    out_dir = repo_path(cfg.paths.evals_dir, "benchmark")
    with open(out_dir / "benchmark_results.json", "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    plot_scoreboard(summary, out_dir / "scoreboard.png")
    plot_metrics_table(summary, out_dir / "metrics_table.png")
    
if __name__ == "__main__":
    main()

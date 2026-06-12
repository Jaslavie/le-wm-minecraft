"""
TreeChop benchmark on baselines and the trained LeWM model.

Per episode (judged from the final stage):
    SUCCESS: reached the tree AND hit the chop completion criterion -> nav + chop success
    CHOP:    reached the tree, no block cut              -> nav success only
    NAV:     never reached the tree                      -> neither

LeWM agents reuse the real planner rollout (run_lewm_client.run_inference, wandb off);
random is its own short loop. Outputs to cfg.paths.evals_dir/benchmark/:
scoreboard.png, metrics_table.png, action_ablation.png, benchmark_results.json.

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
from run_lewm_client import run_inference  # the planner rollout we reuse (no reimplementation)

ACTION_DIM = 10  # [fwd,left,back,right,jump,sneak,sprint,attack,yaw,pitch]
MAX_EPISODES = 5
C_NAV, C_CHOP, C_WARN = "#9bbf6a", "#2f7d32", "#cc7a2f"

AGENT_REGISTRY = {
    "random": dict(kind="random", display="Random"),
    "lewm_no_invdyn": dict(kind="lewm", checkpoint="no_invdyn", display="LeWM (no invdyn)"),
    "lewm_invdyn": dict(kind="lewm", checkpoint="invdyn", display="LeWM (invdyn)"),
}


def connect_with_retry(host, port, wait_timeout):
    """Wait for the Malmo mission server, then return a connected socket (or None)."""
    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            return s
        except OSError:
            time.sleep(2)
    return None


def run_random_episode(conn, cfg, target, transform, seed):
    """Random policy rollout"""
    rng = np.random.default_rng(seed)
    stage = "NAV"
    min_dist = None
    step = 0
    while step < cfg.planner.max_steps:
        frame = recvall(conn, 64 * 64 * 3)
        if frame is None:
            break
        stats = pickle.loads(recvall(conn, struct.unpack(">I", recvall(conn, 4))[0]))
        x, z = stats.get("x"), stats.get("z")
        dist = math.dist((x, z), target) if x is not None and z is not None else float("inf")
        min_dist = dist if min_dist is None else min(min_dist, dist)
        if stage == "NAV" and dist < cfg.planner.success_distance:
            stage = "CHOP"
        action = np.zeros(ACTION_DIM)
        action[:8] = rng.integers(0, 2, 8)
        conn.sendall(pickle.dumps(action.tolist()))
        step += 1
    return {
        "status": stage,
        "nav_success": stage in ("CHOP", "SUCCESS"),
        "chop_success": stage == "SUCCESS",
        "min_distance": None if min_dist is None else round(min_dist, 3),
        "steps": step,
    }


def plot_scoreboard(summary, out_path):
    agents = list(summary)
    x = np.arange(len(agents))
    w = 0.38
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
        ["Min dist (blocks)"] + [f"{summary[a]['min_dist']:.1f}" for a in agents],
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


def run_action_ablation(cfg, model_path, target, transform, nav_goal, cam_mean, cam_std, device, out_path, episodes=2):
    """SINGLE self-contained ablation: enabling left/right strafe or yaw/pitch camera on top of
    forward-only planning degrades navigation. Runs its own planner rollout (so it can mask the
    executed action), `episodes` per variant, and bar-plots the mean closest distance reached."""
    model = load_trained_lewm(cfg, torch.load(model_path, map_location="cpu"), device, strict=False)
    planner = Planner(
        max_iter=cfg.planner.max_iter, n_samples=cfg.planner.n_samples, n_elites=cfg.planner.n_elites,
        planning_horizon=cfg.planner.planning_horizon, action_dim=cfg.action_dim,
        rollout_batch_size=cfg.planner.rollout_batch_size,
    )
    cm, cs = np.asarray(cam_mean).ravel(), np.asarray(cam_std).ravel()
    bm = cfg.benchmark
    variants = ["forward only", "+ strafe", "+ yaw/pitch"]
    closest = {v: [] for v in variants}
    reached = {v: 0 for v in variants}

    for v in variants:
        for ep in range(episodes):
            conn = connect_with_retry(bm.host, bm.port, bm.connect_timeout)
            if conn is None:
                break
            queue, params, min_d, step = [], None, float("inf"), 0
            try:
                while step < cfg.planner.max_steps:
                    frame = recvall(conn, 64 * 64 * 3)
                    if frame is None:
                        break
                    stats = pickle.loads(recvall(conn, struct.unpack(">I", recvall(conn, 4))[0]))
                    x, z = stats.get("x"), stats.get("z")
                    dist = math.dist((x, z), target) if x is not None and z is not None else float("inf")
                    min_d = min(min_d, dist)
                    if dist < cfg.planner.success_distance:
                        reached[v] += 1
                        break
                    obs = process_frame_pixels(transform, frame)
                    if not queue:
                        warm = None if params is None else {
                            k: np.concatenate([params[k][1:], params[k][-1:]], axis=0) for k in ("mu", "sigma", "p")
                        }
                        seq, _losses, params = planner.planner(model, obs, nav_goal, cam_mean, cam_std, cfg.sigreg.lambd, warm_start=warm)
                        raw = np.asarray(seq, dtype=np.float64).reshape(-1, ACTION_DIM)
                        for j, a in enumerate(planner_output_to_actions(seq, cam_mean, cam_std)):
                            a = np.asarray(a, dtype=np.float64).copy()  # camera already zeroed here
                            if v == "forward only":
                                a[1] = a[3] = 0.0                       # drop strafe
                            elif v == "+ yaw/pitch":
                                a[1] = a[3] = 0.0                       # drop strafe, re-enable camera
                                a[8] = raw[j, 8] * cs[0] + cm[0]
                                a[9] = raw[j, 9] * cs[1] + cm[1]
                            queue.append(a)                            # "+ strafe" keeps planner default
                        queue.append(np.zeros(ACTION_DIM))
                    conn.sendall(pickle.dumps(queue.pop(0).tolist()))
                    step += 1
            finally:
                conn.close()
            closest[v].append(None if min_d == float("inf") else min_d)

    means = [float(np.mean([d for d in closest[v] if d is not None])) if any(d is not None for d in closest[v]) else 0.0 for v in variants]
    rates = [reached[v] / episodes * 100 for v in variants]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(np.arange(len(variants)), means, color=[C_CHOP, C_NAV, C_WARN])
    for i, v in enumerate(variants):
        ax.text(i, means[i] + 0.1, f"{means[i]:.1f} blk\n{rates[i]:.0f}% reach", ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_ylabel("Mean closest distance to tree (blocks · lower = better)")
    ax.set_title(f"Action-space ablation — LeWM planner (N={episodes}/variant)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

def main():
    cfg = OmegaConf.load(repo_path("config", "lewm.yaml"))
    bm = cfg.benchmark
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    episodes = min(int(bm.episodes), MAX_EPISODES)
    env_cfg = cfg.env.configs[bm.env]
    target = tuple(env_cfg.get("target_position") or env_cfg.target_positions[0])
    transform = make_transform(cfg.vit.image_size)
    nav_goal = process_frame_pixels(transform, pickle.load(open(repo_path(cfg.paths.nav_goal_frame), "rb"))[0])
    cam_mean, cam_std = get_cam_mean_std(str(repo_path(cfg.paths.data_dir, "mineRL_training.h5")))

    print(f"ENV: {bm.env} | EPISODES: {episodes} | TARGET LOCATION: {target}")
    results = {}
    for name in bm.agents:
        print(f"======== Running {name} ==========")
        spec = AGENT_REGISTRY[name]
        eps = []
        for ep in range(episodes):
            # Random policy rollout in malmo environment
            if spec["kind"] == "random":
                conn = connect_with_retry(bm.host, bm.port, bm.connect_timeout)
                if conn is None:
                    break
                try:
                    r = run_random_episode(conn, cfg, target, transform, seed=ep)
                finally:
                    conn.close()
            # LeWM model rollout
            else:
                r = run_inference(repo_path(bm.models[spec["checkpoint"]]), bm.env, use_wandb=False, eval_mode=True)
            r["nav_success"] = r["status"] in ("CHOP", "SUCCESS")
            r["chop_success"] = r["status"] == "SUCCESS"
            eps.append(r)
            print(
                f"  {name} | episode {ep + 1} | stage={r['status']} "
                f"nav_success={r['nav_success']} chop_success={r['chop_success']} min_dist={r['min_distance']}"
            )
        results[name] = eps

    # Report summary of results from each episode
    summary = {}
    for name, eps in results.items():
        n = len(eps)
        dists = [e["min_distance"] for e in eps if e.get("min_distance") is not None]
        summary[name] = {
            "n": n,
            "nav_rate": sum(e.get("status") in ("CHOP", "SUCCESS") for e in eps) / n if n else 0.0,
            "chop_rate": sum(e.get("status") == "SUCCESS" for e in eps) / n if n else 0.0,
            "min_dist": float(np.mean(dists)) if dists else float("nan"),
            "display": AGENT_REGISTRY[name]["display"],
        }

    out_dir = repo_path(cfg.paths.evals_dir, "benchmark")
    with open(out_dir / "benchmark_results.json", "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    plot_scoreboard(summary, out_dir / "scoreboard.png")
    plot_metrics_table(summary, out_dir / "metrics_table.png")
    run_action_ablation(cfg, repo_path(bm.models["invdyn"]), target, transform, nav_goal,
                        cam_mean, cam_std, device, out_dir / "action_ablation.png")


if __name__ == "__main__":
    main()

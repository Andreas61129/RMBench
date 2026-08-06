#!/usr/bin/env python3
"""Open-loop joint-trajectory plot for RMBench's official Diffusion Policy checkpoint.

Sibling of ../../temporal_scene_graphs/plot_rollout_rmbench.py (the SIR image-only baseline's
open-loop script) -- run this to get a directly comparable RMSE number for DP on the same episode.
Uses the exact same inference path as deploy_policy.py (DP class + DPRunner sliding window), just
teacher-forced against a recorded demo_clean episode instead of a live TASK_ENV, so no simulator is
needed and nothing predicted is ever fed back as an observation.

Must run in the RMBench conda env (needs diffusers + dill + hydra 1.2, not installed in sir_env):
    /home/andreas/miniconda3/envs/RMBench/bin/python plot_rollout_dp.py --task swap_T --episode 0

Note the checkpoint's own obs contract (embedded in the .ckpt, see shape_meta) differs from SIR's:
head_camera only (no wrist cams), raw 14-D joint_action/vector as agent_pos (no joint/gripper
split or reordering), n_obs_steps=3, n_action_steps=6.
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import dill
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy.workspace.robotworkspace import RobotWorkspace  # noqa: E402

DATA_ROOT = Path("/mnt/projects/Temporal_Scene_Graphs/sir_rmbench/RMBench/data/data")

# Raw joint_action/vector layout -- DP's agent_pos IS this vector, unreordered.
JOINT_LABELS = ([f"left_j{i}" for i in range(1, 7)] + ["left_gripper"]
                + [f"right_j{i}" for i in range(1, 7)] + ["right_gripper"])


def decode_jpeg(raw) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("failed to decode RMBench JPEG frame")
    return bgr[..., ::-1]


CAM_KEY_TO_H5 = {"head_cam": "head_camera", "front_cam": "front_camera",
                 "left_cam": "left_camera", "right_cam": "right_camera"}


def load_policy(ckpt_file: Path, device="cuda:0"):
    """Mirrors DP.get_policy in dp_model.py, but also hands back cfg so the caller can read the
    checkpoint's own shape_meta / n_obs_steps / n_action_steps instead of assuming a fixed camera
    set -- see module docstring: the shipped checkpoint's shape_meta only has head_cam, while
    dp_runner.py's get_action() hardcodes head+left+right and KeyErrors on this checkpoint."""
    import hydra
    payload = torch.load(open(ckpt_file, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


def encode_obs(cam_frames: dict, agent_pos: np.ndarray) -> dict:
    """cam_frames: {cam_key: (H,W,3) uint8 RGB}. Matches deploy_policy.py's encode_obs, generalized
    to whichever camera keys the checkpoint's shape_meta actually declares."""
    obs = {k: (np.moveaxis(v, -1, 0) / 255.0).astype(np.float32) for k, v in cam_frames.items()}
    obs["agent_pos"] = agent_pos.astype(np.float32)
    return obs


def load_episode(task: str, episode: int, cam_keys):
    task_dir = DATA_ROOT / task / "demo_clean"
    ep_path = task_dir / "data" / f"episode{episode}.hdf5"
    f = h5py.File(ep_path, "r")
    actions = np.asarray(f["joint_action"]["vector"][()], dtype=np.float32)
    cam_dsets = {k: f["observation"][CAM_KEY_TO_H5[k]]["rgb"] for k in cam_keys}

    instr_path = task_dir / "instructions" / f"episode{episode}.json"
    instruction = ""
    if instr_path.exists():
        raw = json.load(open(instr_path))
        variants = raw.get("seen") or raw.get("unseen") or []
        instruction = variants[0] if variants else ""
    return actions, cam_dsets, instruction, f


def stack_last_n(hist: deque, n: int, key: str) -> np.ndarray:
    """Same padding rule as DPRunner.stack_last_n_obs: right-aligned, repeat the oldest kept
    frame to fill the window if history is shorter than n (true only for the first few steps)."""
    vals = [h[key] for h in hist]
    out = np.zeros((n,) + vals[-1].shape, dtype=vals[-1].dtype)
    start = -min(n, len(vals))
    out[start:] = np.array(vals[start:])
    if n > len(vals):
        out[:start] = out[start]
    return out


def run_open_loop(policy, cfg, actions: np.ndarray, cam_dsets: dict, stride: int, device: str):
    n_obs, n_act = cfg.n_obs_steps, cfg.n_action_steps
    cam_keys = list(cam_dsets.keys())
    num_steps = actions.shape[0]
    hist = deque(maxlen=n_obs + 1)

    steps, chunks = [], []
    for t in range(num_steps):
        frame = {k: decode_jpeg(dset[t]) for k, dset in cam_dsets.items()}
        hist.append(encode_obs(frame, actions[t]))
        if t % stride:
            continue
        if t % 50 == 0:
            print(f"step {t}/{num_steps}", flush=True)

        np_obs = {k: stack_last_n(hist, n_obs, k) for k in cam_keys + ["agent_pos"]}
        obs_dict = dict_apply(np_obs, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        with torch.no_grad():
            action = policy.predict_action(obs_dict)["action"]
        chunks.append(action.squeeze(0).detach().cpu().numpy().astype(np.float32))
        steps.append(t)

    max_len = max(c.shape[0] for c in chunks)
    chunks = [c if c.shape[0] == max_len else np.vstack([c, np.repeat(c[-1:], max_len - c.shape[0], axis=0)])
              for c in chunks]
    return np.array(steps), np.stack(chunks), n_act


def plot(steps, chunks, actions, task, episode, tag, chunk_fan, save_path):
    pred = chunks[:, 0, :]
    gt = actions[steps]
    T = actions.shape[0]

    fig, axes = plt.subplots(7, 2, figsize=(16, 20), sharex=True)
    axes = axes.T.reshape(-1)

    for j in range(14):
        ax = axes[j]
        ax.plot(np.arange(T), actions[:, j], color="#1f77b4", lw=1.6, label="ground truth")
        ax.plot(steps, pred[:, j], color="#2ca02c", lw=1.4, ls="--", label="DP predicted (chunk[0])")
        if chunk_fan:
            for k in range(0, len(steps), chunk_fan):
                horizon = np.arange(steps[k], steps[k] + chunks.shape[1])
                ax.plot(horizon, chunks[k, :, j], color="#98df8a", lw=0.9, alpha=0.7)
        rmse = float(np.sqrt(((pred[:, j] - gt[:, j]) ** 2).mean()))
        ax.set_ylabel(f"{JOINT_LABELS[j]}\nRMSE {rmse:.3f}", fontsize=9)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[6].set_xlabel("timestep")
    axes[13].set_xlabel("timestep")
    total_rmse = float(np.sqrt(((pred - gt) ** 2).mean()))
    fig.suptitle(f"DP open-loop rollout | {task} episode {episode} | ckpt {tag} | "
                 f"overall RMSE {total_rmse:.4f} rad", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"saved plot to {save_path}")
    return total_rmse


def write_summary(steps, chunks, actions, save_path, meta):
    pred, gt = chunks[:, 0, :], actions[steps]
    per_joint = {}
    for j, name in enumerate(JOINT_LABELS):
        d = pred[:, j] - gt[:, j]
        per_joint[name] = {"rmse": float(np.sqrt((d ** 2).mean())), "mae": float(np.abs(d).mean()),
                            "max_abs_error": float(np.abs(d).max()), "bias": float(d.mean())}
    horizon = []
    for k in range(chunks.shape[1]):
        idx = np.clip(steps + k, 0, actions.shape[0] - 1)
        horizon.append(float(np.sqrt(((chunks[:, k, :] - actions[idx]) ** 2).mean())))
    report = {**meta, "num_predicted_steps": int(len(steps)),
              "overall_rmse": float(np.sqrt(((pred - gt) ** 2).mean())),
              "overall_mae": float(np.abs(pred - gt).mean()),
              "per_joint": per_joint, "rmse_by_chunk_offset": horizon}
    json.dump(report, open(save_path, "w"), indent=2)
    print(f"saved summary to {save_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="swap_T")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--checkpoint-num", type=int, default=600)
    ap.add_argument("--setting", default="demo_clean")
    ap.add_argument("--expert-data-num", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--chunk-fan", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=HERE.parents[2] / "temporal_scene_graphs"
                    / "visualizations" / "rmbench_rollouts_dp")
    args = ap.parse_args()

    ckpt_dir = HERE / "checkpoints" / f"{args.task}-{args.setting}-{args.expert_data_num}-{args.seed}"
    ckpt_file = ckpt_dir / f"{args.checkpoint_num}.ckpt"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"{ckpt_file} not found; available: {list(ckpt_dir.glob('*.ckpt')) if ckpt_dir.exists() else 'no such dir'}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    policy, cfg = load_policy(ckpt_file, device)
    cam_keys = [k for k, v in cfg.task.shape_meta.obs.items() if v.get("type") == "rgb"]
    print(f"loaded {ckpt_file}: cameras={cam_keys}, n_obs_steps={cfg.n_obs_steps}, "
          f"n_action_steps={cfg.n_action_steps}, agent_pos_dim={cfg.task.shape_meta.obs.agent_pos.shape}")

    actions, cam_dsets, instruction, h5f = load_episode(args.task, args.episode, cam_keys)
    print(f"episode has {actions.shape[0]} steps; instruction: {instruction!r}")
    try:
        steps, chunks, n_action_steps = run_open_loop(policy, cfg, actions, cam_dsets,
                                                       max(1, args.stride), device)
    finally:
        h5f.close()

    tag = f"{args.checkpoint_num}"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"openloop_dp_{args.task}_ep{args.episode}_ckpt{tag}"

    total_rmse = plot(steps, chunks, actions, args.task, args.episode, tag, args.chunk_fan,
                      out_dir / f"{stem}.png")
    np.savez_compressed(out_dir / f"{stem}.npz", steps=steps, chunks=chunks, gt_actions=actions)
    report = write_summary(steps, chunks, actions, out_dir / f"{stem}.json", {
        "task": args.task, "episode": args.episode, "checkpoint": str(ckpt_file),
        "instruction": instruction, "cameras": [CAM_KEY_TO_H5[k] for k in cam_keys],
        "n_obs_steps": cfg.n_obs_steps, "n_action_steps": n_action_steps, "stride": max(1, args.stride),
    })
    print(f"\noverall RMSE: {total_rmse:.4f} rad   MAE: {report['overall_mae']:.4f}")
    print("RMSE by chunk offset:", " ".join(f"{v:.3f}" for v in report["rmse_by_chunk_offset"]))
    worst = sorted(report["per_joint"].items(), key=lambda kv: -kv[1]["rmse"])[:5]
    print("worst joints:", ", ".join(f"{k} {v['rmse']:.3f}" for k, v in worst))


if __name__ == "__main__":
    main()

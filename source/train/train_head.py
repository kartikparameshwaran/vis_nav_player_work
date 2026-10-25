"""
Train a small projection head on frozen DINOv2 descriptors so that frames of the SAME PLACE
(true position within 0.4 m and heading within 20 deg) are close and different places are far.
Labels come from the simulator's true poses (see gen_data.py); the head is what you ship.

    uv run source/train/train_head.py --data data_train --out models/head.npy --epochs 30
    uv run source/eval_local.py --descriptor dino --head models/head.npy ...      (see features.py)

Why a head and not the whole backbone: ~50k frames, 2-3 days, and the head trains in minutes
on a GPU from cached features; fine-tuning the last transformer blocks is the next step if
the held-out recall here shows the head helps.
Held-out check: mazes whose seed % 5 == 0 are validation; recall@1 = the nearest frame from
a different drive is a true neighbour pose.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # source/, for features, vlad

import cv2
import numpy as np
import torch
from features import DinoDescriptor

POS_M, POS_DEG, NEG_M, NEG_DEG = 0.4, 20.0, 1.5, 60.0


def load_maze(path: Path):
    imgs, poses, traj = [], [], []
    for t in sorted(path.glob("traj_*")):
        for r in json.loads((t / "data_info.json").read_text()):
            imgs.append(t / r["image"])
            poses.append(r["pose"])
            traj.append(int(t.name.split("_")[1]))
    return imgs, np.array(poses, np.float32), np.array(traj)


def pair_masks(poses: np.ndarray):
    d = np.linalg.norm(poses[:, None, :2] - poses[None, :, :2], axis=-1)
    dyaw = np.abs((poses[:, None, 2] - poses[None, :, 2] + math.pi) % (2 * math.pi) - math.pi)
    dyaw = np.degrees(dyaw)
    pos = (d < POS_M) & (dyaw < POS_DEG)
    neg = (d > NEG_M) | (dyaw > NEG_DEG)
    return pos, neg


def embed_maze(desc: DinoDescriptor, imgs, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    out = []
    for i in range(0, len(imgs), 128):
        out.append(desc.embed([cv2.imread(str(p)) for p in imgs[i : i + 128]]))
    emb = np.concatenate(out)
    np.save(cache, emb)
    return emb


def recall_at_1(emb: np.ndarray, poses: np.ndarray, traj: np.ndarray, proj=None) -> float:
    x = emb if proj is None else torch.nn.functional.normalize(torch.from_numpy(emb) @ proj, dim=1).numpy()
    pos, _ = pair_masks(poses)
    hits, n = 0, 0
    for q in range(len(x)):
        other = traj != traj[q]
        if not (pos[q] & other).any():
            continue
        s = np.where(other, x @ x[q], -2)
        hits += bool(pos[q, int(np.argmax(s))])
        n += 1
    return hits / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data_train"))
    ap.add_argument("--out", type=Path, default=Path("models/head.npy"))
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    desc = DinoDescriptor(device=dev, cache_dir=args.data / ".emb")
    desc.cache_dir.mkdir(parents=True, exist_ok=True)
    mazes = {}
    for m in sorted(args.data.glob("maze-*")):
        imgs, poses, traj = load_maze(m)
        mazes[m.name] = (embed_maze(desc, imgs, desc.cache_dir / f"{m.name}.npy"), poses, traj)
        print(m.name, len(imgs), "frames", flush=True)
    val = [k for k in mazes if int(k.split("-")[1]) % 5 == 0]
    train = [k for k in mazes if k not in val]
    d_in = next(iter(mazes.values()))[0].shape[1]
    proj = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(d_in, args.dim)).to(dev))
    opt = torch.optim.AdamW([proj], lr=args.lr, weight_decay=1e-4)

    def report(tag):
        with torch.no_grad():
            p = proj.detach().cpu()
            base = np.mean([recall_at_1(mazes[k][0], mazes[k][1], mazes[k][2]) for k in val]) if val else float("nan")
            ft = np.mean([recall_at_1(mazes[k][0], mazes[k][1], mazes[k][2], p) for k in val]) if val else float("nan")
        print(f"{tag}: held-out recall@1  raw DINO {base:.3f}   head {ft:.3f}", flush=True)

    report("init")
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        for _ in range(len(train) * 4):
            k = train[int(rng.integers(len(train)))]
            emb, poses, _ = mazes[k]
            idx = rng.choice(len(emb), min(args.batch, len(emb)), replace=False)
            pos, neg = pair_masks(poses[idx])
            x = torch.nn.functional.normalize(torch.from_numpy(emb[idx]).to(dev) @ proj, dim=1)
            sim_ = x @ x.T / args.temp
            eye = torch.eye(len(idx), dtype=torch.bool, device=dev)
            pos_t = torch.from_numpy(pos).to(dev) & ~eye
            neg_t = torch.from_numpy(neg).to(dev)
            valid = pos_t.any(1)
            if not valid.any():
                continue
            logits = sim_.masked_fill(~(pos_t | neg_t), -1e9)
            lse_all = torch.logsumexp(logits, 1)
            lse_pos = torch.logsumexp(sim_.masked_fill(~pos_t, -1e9), 1)
            loss = (lse_all - lse_pos)[valid].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 4 or ep == args.epochs - 1:
            report(f"epoch {ep + 1} loss {loss.item():.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, proj.detach().cpu().numpy())
    print("saved", args.out)


if __name__ == "__main__":
    main()

"""
Train SPTM's two networks on pose-labelled drives from the LOCAL simulator.

    # 1) data: random drives with true poses (once; many mazes)
    uv run source/train/gen_data.py --mazes 100-139 --out data_train --ticks 3000

    # 2) retrieval network R  ("is this the same place?")
    uv run source/sptm_train.py r --data data_train --out models/sptm_r.pt --iters 30000
    # 3) locomotion network L  ("which move gets me toward that frame?")
    uv run source/sptm_train.py l --data data_train --out models/sptm_l.pt --iters 30000

Differences from the paper, on purpose: labels come from true poses (so frames from DIFFERENT
drives count as the same place, which is exactly what the agent needs), ResNet-18 starts from
ImageNet weights, and L's label is the move a pose-aware expert would make, not the random
walker's noisy action. Mazes whose seed % 5 == 0 are held out for validation.
Poses are training labels only; the agent never sees them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # source/

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sptm_models import ACTIONS, LNet, RNet, pick_device, prep, to_tensor

POS_M, POS_DEG = 0.35, 30.0  # same place: within this distance and heading
NEG_M, NEG_DEG = 1.0, 75.0  # different place: farther than this, or facing well away
LOOKAHEAD = 8  # L: target frame is 1..LOOKAHEAD frames ahead in the same drive
TURN_RAD = math.radians(10.0)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Maze:
    def __init__(self, path: Path):
        self.name = path.name
        cache, meta = path / "frames160.npy", path / "meta160.npz"
        if not cache.exists():
            imgs, poses, traj, act = [], [], [], []
            for t in sorted(path.glob("traj_*"), key=lambda d: int(d.name.split("_")[1])):
                for r in json.loads((t / "data_info.json").read_text()):
                    imgs.append(cv2.imread(str(t / r["image"])))
                    poses.append(r["pose"])
                    traj.append(int(t.name.split("_")[1]))
                    act.append(r["action"][0])
            np.save(cache, prep(imgs))
            np.savez(meta, poses=np.array(poses, np.float64), traj=np.array(traj), act=np.array(act))
        m = np.load(meta)
        self.imgs = np.load(cache, mmap_mode="r")
        self.poses, self.traj, self.act = m["poses"], m["traj"], m["act"]
        self.n = len(self.poses)
        self.idx = np.arange(self.n)

    def pose_rel(self, a: int):
        d = np.linalg.norm(self.poses[:, :2] - self.poses[a, :2], axis=1)
        dyaw = np.degrees(np.abs(wrap(self.poses[:, 2] - self.poses[a, 2])))
        return d, dyaw


def turn_sign(mazes: list[Maze]) -> float:
    """+1 if a LEFT action raises yaw (counter-clockwise), -1 otherwise, measured from the data."""
    vals = []
    for m in mazes[:5]:
        ok = (m.act[:-1] == "LEFT") & (m.traj[:-1] == m.traj[1:])
        vals += list(wrap(m.poses[1:, 2] - m.poses[:-1, 2])[ok])
    s = float(np.sign(np.mean(vals))) if vals else 1.0
    print(f"turn sign {s:+.0f} (mean yaw change on LEFT {np.mean(vals) if vals else float('nan'):+.3f} rad)")
    return s or 1.0


def expert_action(pa, pb, sign: float) -> int:
    """The move a pose-aware expert would make to get from pose a toward pose b."""
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    ang = wrap(math.atan2(dy, dx) - pa[2]) if math.hypot(dx, dy) > 0.06 else wrap(pb[2] - pa[2])
    if abs(ang) > TURN_RAD:
        return ACTIONS.index("LEFT") if ang * sign > 0 else ACTIONS.index("RIGHT")
    return ACTIONS.index("FORWARD")


def load_all(data: Path, max_mazes: int | None):
    mazes = [Maze(p) for p in sorted(data.glob("maze-*"))[:max_mazes]]
    if not mazes:
        sys.exit(f"no maze-*/ under {data}; run source/train/gen_data.py first")
    val = [m for m in mazes if int(m.name.split("-")[1]) % 5 == 0]
    train = [m for m in mazes if m not in val] or mazes
    print(f"{len(train)} train mazes, {len(val)} validation mazes")
    return train, val


# ---- R

def sample_r(m: Maze, B: int, rng: np.random.Generator):
    a = rng.integers(m.n, size=B)
    y = rng.random(B) < 0.5
    b = np.empty(B, np.int64)
    for k in range(B):
        d, dyaw = m.pose_rel(a[k])
        if y[k]:
            cand = np.flatnonzero((d < POS_M) & (dyaw < POS_DEG) & (m.idx != a[k]))
            if len(cand):
                b[k] = rng.choice(cand)
                continue
            y[k] = False
        cand = np.flatnonzero((d > NEG_M) | (dyaw > NEG_DEG))
        b[k] = rng.choice(cand) if len(cand) else (a[k] + m.n // 2) % m.n
    return a, b, y.astype(np.float32)


@torch.no_grad()
def eval_r(net: RNet, val: list[Maze], dev, rng) -> str:
    if not val:
        return "no validation mazes"
    net.eval()
    correct = total = 0
    hits = queries = 0
    for m in val:
        a, b, y = sample_r(m, 400, rng)
        logit = net(to_tensor(m.imgs[a], dev), to_tensor(m.imgs[b], dev)).cpu().numpy()
        correct += int(((logit > 0) == (y > 0.5)).sum())
        total += len(y)
        emb = torch.cat([net.embed(to_tensor(m.imgs[i : i + 128], dev)) for i in range(0, m.n, 128)])
        for q in rng.choice(m.n, min(150, m.n), replace=False):
            d, dyaw = m.pose_rel(q)
            other = m.traj != m.traj[q]
            good = (d < 0.3) & (dyaw < 25) & other
            if not good.any():
                continue
            lg = net.pair_logit(emb[q : q + 1].expand(m.n, -1), emb).cpu().numpy()
            lg[~other] = -1e9
            hits += bool(good[int(np.argmax(lg))])
            queries += 1
    net.train()
    return f"pair acc {correct / max(total, 1):.3f}  recall@1 (other drives) {hits / max(queries, 1):.3f} over {queries} queries"


def train_r(args, train, val, dev):
    net = RNet(pretrained=not args.no_pretrained).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        m = train[int(rng.integers(len(train)))]
        a, b, y = sample_r(m, args.batch, rng)
        logit = net(to_tensor(m.imgs[a], dev), to_tensor(m.imgs[b], dev))
        loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(y).to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0:
            print(f"it {it}/{args.iters} loss {loss.item():.4f}  {time.time() - t0:.0f}s", flush=True)
        if it % 2000 == 0 or it == args.iters:
            print("  val:", eval_r(net, val, dev, rng), flush=True)
            save(net, args)
    return net


# ---- L

def sample_l(m: Maze, B: int, rng: np.random.Generator, sign: float):
    a = rng.integers(m.n - LOOKAHEAD - 1, size=B)
    b = a + rng.integers(1, LOOKAHEAD + 1, size=B)
    keep = m.traj[a] == m.traj[b]
    a, b = a[keep], b[keep]
    y = np.array([expert_action(m.poses[i], m.poses[j], sign) for i, j in zip(a, b, strict=True)])
    return a, b, y


@torch.no_grad()
def eval_l(net: LNet, val: list[Maze], dev, rng, sign) -> str:
    if not val:
        return "no validation mazes"
    net.eval()
    ok = tot = 0
    counts = np.zeros(len(ACTIONS))
    for m in val:
        a, b, y = sample_l(m, 500, rng, sign)
        if not len(a):
            continue
        pred = net(to_tensor(m.imgs[a], dev), to_tensor(m.imgs[b], dev)).argmax(1).cpu().numpy()
        ok += int((pred == y).sum())
        tot += len(y)
        counts += np.bincount(y, minlength=len(ACTIONS))
    net.train()
    return f"action acc {ok / max(tot, 1):.3f} (always-majority {counts.max() / max(counts.sum(), 1):.3f})"


def train_l(args, train, val, dev):
    sign = turn_sign(train)
    args.turn_sign = sign
    net = LNet(pretrained=not args.no_pretrained).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(1)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        m = train[int(rng.integers(len(train)))]
        a, b, y = sample_l(m, args.batch, rng, sign)
        if not len(a):
            continue
        logits = net(to_tensor(m.imgs[a], dev), to_tensor(m.imgs[b], dev))
        loss = F.cross_entropy(logits, torch.from_numpy(y).to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0:
            print(f"it {it}/{args.iters} loss {loss.item():.4f}  {time.time() - t0:.0f}s", flush=True)
        if it % 2000 == 0 or it == args.iters:
            print("  val:", eval_l(net, val, dev, rng, sign), flush=True)
            save(net, args)
    return net


def save(net, args) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "args": {k: str(v) for k, v in vars(args).items()}}, args.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("net", choices=["r", "l"])
    ap.add_argument("--data", type=Path, default=Path("data_train"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-mazes", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (no internet)")
    args = ap.parse_args()
    args.out = args.out or Path(f"models/sptm_{args.net}.pt")
    dev = pick_device(args.device)
    print("device", dev)
    train, val = load_all(args.data, args.max_mazes)
    (train_r if args.net == "r" else train_l)(args, train, val, dev)
    print("saved", args.out)


if __name__ == "__main__":
    main()

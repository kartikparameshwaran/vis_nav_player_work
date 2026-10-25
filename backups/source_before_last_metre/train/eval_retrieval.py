"""
How good is place recognition, in numbers? Uses pose-labelled drives from gen_data.py.

    uv run source/train/gen_data.py --mazes 100-109 --out data_train --ticks 3000
    uv run source/train/eval_retrieval.py --data data_train --descriptors vlad dino
    uv run source/train/eval_retrieval.py --data data_train --descriptors vlad dino --head models/head.npy

For every maze: the database is all frames, the query is each frame, the answer must come from
a DIFFERENT drive (so we test "have I been here before", not "which frame is next to me").
A hit = the retrieved frame's true pose is within --pos-m metres and --pos-deg degrees.
Reports recall@1 and recall@5, per maze and mean. Poses are labels only; nothing here is
available at run time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # source/, for features, vlad

import numpy as np
from vlad import Frame


def load_maze(path: Path):
    frames, poses = [], []
    for t in sorted(path.glob("traj_*"), key=lambda d: int(d.name.split("_")[1])):
        ti = int(t.name.split("_")[1])
        for r in json.loads((t / "data_info.json").read_text()):
            if "pose" not in r:
                continue
            frames.append(Frame(t / r["image"], r["action"][0], ti))
            poses.append(r["pose"])
    return frames, np.array(poses, np.float64)


def recalls(emb: np.ndarray, poses: np.ndarray, traj: np.ndarray, pos_m: float, pos_deg: float):
    d = np.linalg.norm(poses[:, None, :2] - poses[None, :, :2], axis=-1)
    dyaw = np.degrees(np.abs((poses[:, None, 2] - poses[None, :, 2] + math.pi) % (2 * math.pi) - math.pi))
    pos = (d < pos_m) & (dyaw < pos_deg)
    r1 = r5 = n = 0
    sims = emb @ emb.T
    for q in range(len(emb)):
        other = traj != traj[q]
        if not (pos[q] & other).any():
            continue  # nothing to find from the other drives: not a fair query
        s = np.where(other, sims[q], -2.0)
        top = np.argsort(-s)[:5]
        r1 += bool(pos[q, top[0]])
        r5 += bool(pos[q, top].any())
        n += 1
    return (r1 / max(n, 1), r5 / max(n, 1), n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data_train"))
    ap.add_argument("--descriptors", nargs="+", default=["vlad"], choices=["vlad", "dino", "both"])
    ap.add_argument("--head", type=Path, default=None)
    ap.add_argument("--pos-m", type=float, default=0.3)
    ap.add_argument("--pos-deg", type=float, default=25.0)
    ap.add_argument("--clusters", type=int, default=128)
    ap.add_argument("--max-mazes", type=int, default=10)
    args = ap.parse_args()

    import features

    mazes = sorted(args.data.glob("maze-*"))[: args.max_mazes]
    if not mazes:
        sys.exit(f"no maze-*/ folders under {args.data}; run gen_data.py first")
    table: dict[str, list] = {k: [] for k in args.descriptors}
    for m in mazes:
        frames, poses = load_maze(m)
        traj = np.array([f.trajectory for f in frames])
        row = [m.name, f"{len(frames)} frames"]
        for kind in args.descriptors:
            cache = Path("cache") / "eval" / m.name / kind
            desc = features.make(kind, cache, args.clusters, args.head if kind != "vlad" else None)
            emb = np.asarray(desc.database(frames), np.float32)
            r1, r5, n = recalls(emb, poses, traj, args.pos_m, args.pos_deg)
            table[kind].append((r1, r5))
            row.append(f"{kind}: r@1 {r1:.2f} r@5 {r5:.2f} (n={n})")
        print("  ".join(row), flush=True)
    print("\nMEAN over", len(mazes), "mazes")
    for kind, v in table.items():
        a = np.array(v)
        print(f"  {kind:5s} recall@1 {a[:, 0].mean():.3f}   recall@5 {a[:, 1].mean():.3f}")


if __name__ == "__main__":
    main()

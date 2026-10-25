"""
Generate pose-labelled drives from the LOCAL simulator, for training place recognition.

    uv run source/train/gen_data.py --mazes 100-199 --out data_train --ticks 3000

For each maze, three random drives (forward-biased random walk, turn when blocked), a frame
every 5 ticks, the TRUE pose saved alongside. Output matches the exploration-data layout:
    <out>/maze-<seed>/traj_<i>/{<k>.jpg, data_info.json}   with  {"image","action","pose":[x,y,yaw]}
Poses only exist locally; they are used for training labels and never at run time.
Needs the course simulator (`uv sync`) and its texture pack, so run it where that works
(login node or your laptop), not inside a job without internet the first time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim


def parse(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def drive(world, out: Path, ticks: int, every: int, rng: np.random.Generator) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame, recs, stale, turning = world.initial_frame(), [], 0, 0
    last = None
    for k in range(ticks // every):
        small = cv2.resize(frame, (32, 24)).astype(np.float32)
        stale = stale + 1 if last is not None and np.abs(small - last).mean() < 1.0 else 0
        last = small
        if stale >= 2 and not turning:
            turning, direction = int(rng.integers(3, 14)), (sim.LEFT if rng.random() < 0.5 else sim.RIGHT)
        if turning:
            action, name, turning = direction, "LEFT" if direction == sim.LEFT else "RIGHT", turning - 1
        elif rng.random() < 0.04:
            direction, turning = (sim.LEFT if rng.random() < 0.5 else sim.RIGHT), int(rng.integers(1, 4))
            action, name = direction, "LEFT" if direction == sim.LEFT else "RIGHT"
        else:
            action, name = sim.FORWARD, "FORWARD"
        x, y, yaw = world.pose()
        cv2.imwrite(str(out / f"{k}.jpg"), frame)
        recs.append({"step": k, "image": f"{k}.jpg", "action": [name], "pose": [x, y, yaw]})
        frame = world.step(action, every)[-1]
    (out / "data_info.json").write_text(json.dumps(recs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mazes", default="100-119")
    p.add_argument("--out", type=Path, default=Path("data_train"))
    p.add_argument("--ticks", type=int, default=3000)
    p.add_argument("--drives", type=int, default=3)
    p.add_argument("--every", type=int, default=5)
    args = p.parse_args()
    textures = sim.Textures.find()
    for seed in parse(args.mazes):
        world = sim.Simulator(textures, seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed)
        for d in range(args.drives):
            rng = np.random.default_rng(seed * 100 + d)
            world.reset(seed=seed, motion_seed=seed * 100 + d)
            drive(world, args.out / f"maze-{seed}" / f"traj_{d}", args.ticks, args.every, rng)
        print("maze", seed, "done", flush=True)


if __name__ == "__main__":
    main()

"""
Score an agent on several local mazes (free, headless) and write one CSV row per maze.

    uv run source/eval_local.py --seeds 7-16 --descriptor vlad --out results/vlad.csv
    uv run source/eval_local.py --seeds 7,9,12 --descriptor dino --burst 4 --out results/dino.csv
    uv run source/summarize.py results/*.csv

This is the only number that matters before you spend a server attempt: mean points over
many mazes (same rubric as the course default), plus how often the gate was passed.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cli  # noqa: E402
from bayes_agent import BayesAgent  # noqa: E402
from vis_nav_sdk import run  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="7-16")
    p.add_argument("--descriptor", default="vlad", choices=["vlad", "dino", "both"])
    p.add_argument("--subsample", type=int, default=1)
    p.add_argument("--head", type=Path, default=None)
    p.add_argument("--burst", type=int, default=1)
    p.add_argument("--budget", type=int, default=2000, help="agent's own step cap (local sim allows 60,000)")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--near-hops", type=int, default=6)
    p.add_argument("--out", type=Path, default=Path("results/eval.csv"))
    p.add_argument("--log-dir", type=Path, default=None)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "descriptor", "points", "spl", "distance_m", "heading_deg", "steps", "gate", "seconds", "error"]
    new = not args.out.exists()
    with args.out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        if new:
            w.writeheader()
        for seed in parse_seeds(args.seeds):
            row = dict.fromkeys(fields, "")
            row.update(seed=seed, descriptor=args.descriptor)
            t0 = time.time()
            try:
                data = cli.record_exploration_data(seed)
                agent = BayesAgent(
                    data, descriptor=args.descriptor, subsample=args.subsample, head=args.head, burst=args.burst,
                    tau=args.tau, near_hops=args.near_hops, cache_dir=Path("cache") / f"local-{seed}",
                    log_dir=args.log_dir, budget=args.budget,
                )
                res = run(
                    agent, f"local:{seed}", viewer=False, check=False, confirm=False,
                    browser=False, fps=None, quiet=True,
                )
                if res is not None:
                    row.update(
                        points=res.points, spl=res.spl, steps=res.nav_steps,
                        distance_m=getattr(res.arrival, "distance_m", ""),
                        heading_deg=getattr(res.arrival, "heading_deg", ""),
                        gate=int(res.points > 0),
                    )
                else:
                    row.update(points=0, gate=0, error="unscored (quit / budget)")
            except Exception:  # keep sweeping; the traceback is the useful part
                row.update(points=0, gate=0, error=traceback.format_exc().splitlines()[-1])
                traceback.print_exc()
            row["seconds"] = round(time.time() - t0)
            w.writerow(row)
            fh.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main()

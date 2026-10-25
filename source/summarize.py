"""Summarise eval CSVs:  uv run source/summarize.py results/*.csv"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict


def main(paths: list[str]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                groups[f"{path}:{row['descriptor']}"].append(row)
    print(f"{'run':44} {'n':>3} {'points':>7} {'gate%':>6} {'spl':>5} {'dist':>6} {'head':>6} {'errs':>4}")
    for name, rows in sorted(groups.items()):
        n = len(rows)
        gate = [r for r in rows if r["gate"] == "1"]

        def mean(key, rs):
            vals = [float(r[key]) for r in rs if r[key] not in ("", None)]
            return sum(vals) / len(vals) if vals else float("nan")

        print(
            f"{name:44} {n:3d} {mean('points', rows):7.2f} {100 * len(gate) / n:6.0f} "
            f"{mean('spl', gate):5.2f} {mean('distance_m', gate):6.2f} {mean('heading_deg', gate):6.1f} "
            f"{sum(1 for r in rows if r['error']):4d}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])

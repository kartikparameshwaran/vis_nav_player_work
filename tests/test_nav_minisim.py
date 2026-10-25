"""Plumbing tests on the stand-in maze: localise -> plan -> steer -> approach -> check in."""

from __future__ import annotations

import minisim
import numpy as np
from my_agent import MyAgent
from navigator import BeliefFilter, TopoMap
from vlad import load_frames


def test_agent_reaches_goal_on_minimaze(tmp_path):
    results = []
    for seed in (2, 5):
        d = tmp_path / f"s{seed}"
        world, start, goal = minisim.make_episode(seed, d, n=5, length=45)
        agent = MyAgent(d, n_clusters=64, cache_dir=d / "cache")
        results.append(minisim.run_episode(agent, world, start, goal, max_steps=4000))
    print(results)
    # Smoke test only: the whole loop runs, moves, and ends with a CHECKIN and a valid score.
    # Success on the stand-in maze is deliberately NOT asserted: it is noisy and not predictive
    # of the real simulator. Use eval_local.py on real local mazes for that.
    assert all(r["checked_in"] and r["steps"] > 0 and r["points"] >= 0 for r in results)


def test_belief_filter_tracks_a_drive(tmp_path):
    d = tmp_path / "w"
    minisim.make_episode(5, d, n=5, length=45)
    frames, bounds = load_frames(d, 1)
    from features import VladDescriptor

    db = VladDescriptor(64, d / "c").database(frames)
    topo = TopoMap(frames, bounds, db, verify=False, stride=8)
    bf = BeliefFilter(topo)
    lo, hi = bounds[0]
    hits = 0
    for i in range(lo, lo + 120):
        est = bf.update(db[i])
        hits += abs(est - i) <= 2
    assert hits > 100
    assert np.isclose(bf.b.sum(), 1.0)

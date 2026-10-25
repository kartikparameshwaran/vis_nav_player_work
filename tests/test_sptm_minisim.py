"""SPTM agent plumbing on the stand-in maze, with a fake scorer (VLAD cosine as 'R', a
shift-based steer as 'L'). Torch is not needed; the learned networks are tested elsewhere."""

from __future__ import annotations

import minisim
import numpy as np
from features import VladDescriptor
from geometry import Matcher, shift_deg
from sptm_agent import SptmAgent
from vlad import load_frames


class FakeScorer:
    def __init__(self, data_dir):
        frames, _ = load_frames(data_dir, 1)
        self.desc = VladDescriptor(64, data_dir / "cache")
        self.desc.database(frames)  # fits the codebook
        self.m = Matcher()

    def embed(self, images):
        return np.stack([self.desc.extract(im) for im in images]).astype(np.float32)

    def prob(self, a, b):
        cos = np.sum(a * b, axis=1)
        return np.clip((cos - 0.08) / 0.2, 0.0, 1.0)

    def locomote(self, cur, wp, node=-1):
        m = self.m.match(self.m.features(wp), self.m.features(cur))
        left = shift_deg(m, 92.0) if m.n >= 8 else 0.0
        return "RIGHT" if left > 8 else "LEFT" if left < -8 else "FORWARD"


def test_sptm_agent_runs_end_to_end(tmp_path):
    world, start, goal = minisim.make_episode(5, tmp_path, n=5, length=45)
    agent = SptmAgent(
        tmp_path, FakeScorer(tmp_path), s_local=0.4, s_reach=0.5, s_shortcut=0.6, window=2,
    )
    res = minisim.run_episode(agent, world, start, goal, max_steps=2500)
    print(res, "shortcuts", len(agent.shortcuts))
    assert res["steps"] > 0 and (res["checked_in"] or res["steps"] >= 2400)
    assert agent.paths and agent.reach.any()

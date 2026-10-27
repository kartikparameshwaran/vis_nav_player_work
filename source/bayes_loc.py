"""
Bayes-filter localisation over the SPTM map (lecture 8: SLAM as probabilistic estimation).

Instead of trusting one best-matching map photo and patching it with jump rules, keep a probability
for every map photo (node) and update it every step:

    predict : after a move, spread the belief to neighbouring photos along each drive, and across
              the verified shortcuts and the spawn join (the robot moved a little, we do not know
              exactly how far)
    update  : multiply by how well each photo matches the live view -- the place-recognition score
              for all photos (cheap), then SuperGlue inliers for the few most likely photos
              (lecture 8's two-step pipeline: shortlist, then re-rank by geometric verification)
    decide  : the photos within ``PLAUSIBLE`` of the most likely one are all plausible; plan from the
              one with the shortest route to the goal. Two drives through the same place give two
              plausible photos -- choosing the more useful one avoids the plan flips that a single
              best match causes (c33_m8: 225 -> 444 hops at one spot).

A little probability is always spread over the whole map, so a wrong belief can recover.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

SIGMA = 1.5  # spread along a drive per step, in photos
STAY = 0.4  # probability of being at the same photo after one step
JUMP = 0.3  # relative weight of crossing a shortcut / the spawn join in one step
TAU = 0.15  # likelihood temperature on the 0..1 place-recognition score
FLOOR = 1e-3  # likelihood floor: one bad frame cannot wipe out the true place
UNIFORM = 1e-4  # mass spread over the whole map each step, so a wrong belief can recover
TOP_K = 5  # photos re-ranked with SuperGlue each step
SG_MID, SG_SCALE = 20.0, 10.0  # SuperGlue inliers -> likelihood ratio exp((n - 20) / 10), clipped
PLAUSIBLE = 0.5  # photos with at least this fraction of the top probability are plausible


class BayesLocaliser:
    def __init__(self, n: int, bounds: list[tuple[int, int]], links: list[tuple[int, int]]):
        rows, cols, vals = [], [], []
        reach = int(np.ceil(3 * SIGMA))
        for s, e in bounds:
            for i in range(s, e):
                for d in range(-reach, reach + 1):
                    j = i + d
                    if s <= j < e:
                        rows.append(i), cols.append(j)
                        vals.append(STAY if d == 0 else (1 - STAY) * np.exp(-0.5 * (d / SIGMA) ** 2))
        for i, j in links:  # verified shortcuts and the spawn join: being at i makes j plausible
            for a, b in ((i, j), (j, i)):
                rows.append(a), cols.append(b), vals.append(JUMP)
        k = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        k = sp.diags(1 / np.asarray(k.sum(1)).ravel()) @ k
        self.kt = k.T.tocsr()  # b_new = K^T b
        self.n = n
        self.b = np.full(n, 1.0 / n)
        self.inliers: dict[int, int] = {}  # SuperGlue inliers of the last re-ranked photos (for the log)

    def reset(self, start: list[int]) -> None:
        """All probability on ``start`` (the drives' first photos: every run begins at the spawn)."""
        self.b = np.zeros(self.n)
        self.b[start] = 1.0 / len(start)

    def step(self, p: np.ndarray, sg_inliers=None) -> np.ndarray:
        """One predict + update. ``p``: place-recognition score (0..1) of every photo against the
        live view. ``sg_inliers(node) -> int``: SuperGlue inliers of a photo against the live view,
        or None to skip re-ranking. Returns the posterior."""
        b = self.kt @ self.b
        b = (1 - UNIFORM) * b + UNIFORM / self.n
        b *= FLOOR + np.exp((np.asarray(p, float) - 1.0) / TAU)
        b /= b.sum()
        self.inliers = {}
        if sg_inliers is not None:
            for i in np.argsort(-b)[:TOP_K]:
                n_in = sg_inliers(int(i))
                self.inliers[int(i)] = n_in
                b[i] *= float(np.clip(np.exp((n_in - SG_MID) / SG_SCALE), 0.05, 20.0))
            b /= b.sum()
        self.b = b
        return b

    def choose(self, cost_to_goal) -> tuple[int, float]:
        """The plausible photo with the cheapest route to the goal, and the probability of the most
        likely photo (a confidence for the log). ``cost_to_goal(node)`` returns route length."""
        top = float(self.b.max())
        cand = np.flatnonzero(self.b >= PLAUSIBLE * top)
        return int(min(cand, key=cost_to_goal)), top

"""
SPTM agent (Savinov, Dosovitskiy, Koltun, ICLR 2018) rebuilt for this challenge.

    uv run source/sptm_agent.py --local 7 --budget 2000 --verbose
    uv run source/sptm_agent.py --challenge <id>

Memory graph : one node per exploration frame; edges between consecutive frames, plus
               "visual shortcuts" between far-apart frames that a learned network R says are
               the same place for a whole WINDOW of neighbouring frames (not one lucky pair).
Localisation : R scores the camera frame against every node. Search near the previous node
               first; fall back to a global search only when nothing nearby is similar enough.
Planning     : shortest path (unit edges) from the estimated node to the goal node.
Waypoint     : the furthest node along that path that R still says is "the same place" as
               what we see now (so it is close and reachable).
Locomotion   : a learned network L maps (current frame, waypoint frame) -> FORWARD/LEFT/RIGHT.
Last metre   : same RANSAC servo on the four goal photos as my_agent.py, then CHECKIN.

The scorer (R and L) is injected: ``sptm_models.TorchScorer`` for real runs, anything with
``embed / prob / locomote`` for tests. This file needs no torch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from geometry import Matcher, scale_ratio, shift_deg
from my_agent import TURN_DEG, VIEW_OFFSET, _intrinsic_focal
from vis_nav_sdk import Action, Agent, Observation, Result, SessionInfo, run
from vlad import load_frames

SIGN = +1  # LEFT x6 shifted the picture by +26 / +23 deg on the real simulator (two mazes)


class SptmAgent(Agent):
    def __init__(
        self,
        data_dir: Path,
        scorer,
        *,
        subsample: int = 1,
        h_min: int = 1,
        h_max: int = 7,
        s_local: float = 0.7,
        s_reach: float = 0.8,
        s_shortcut: float = 0.9,
        local_hops: int = 12,
        shortcut_gap: int = 10,
        shortcut_k: int = 8,
        window: int = 3,
        near_hops: int = 3,  #3 originally
        burst: int = 1,
        scale_tol: float = 0.04,
        budget: int | None = None,
        log_dir: Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.scorer = scorer
        self.frames, self.bounds = load_frames(Path(data_dir), subsample)
        print(f"{len(self.frames)} frames from {len(self.bounds)} trajectories")
        self.n = len(self.frames)
        self.traj = np.concatenate([[t] * (e - s) for t, (s, e) in enumerate(self.bounds)])
        imgs = [cv2.imread(str(f.path)) for f in self.frames]
        self.emb = np.asarray(scorer.embed(imgs), np.float32)
        del imgs
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))
        for s, e in self.bounds:
            self.G.add_edges_from((i, i + 1) for i in range(s, e - 1))
        self.shortcuts = self._shortcuts(shortcut_gap, shortcut_k, window, s_shortcut)
        self.G.add_edges_from(self.shortcuts)
        print(f"graph: {self.G.number_of_edges()} edges, {len(self.shortcuts)} verified shortcuts")
        self.matcher = Matcher()
        self.h_min, self.h_max, self.s_local, self.s_reach = h_min, h_max, s_local, s_reach
        self.local_hops, self.near_hops, self.burst = local_hops, near_hops, burst
        self.scale_tol, self.budget = scale_tol, budget
        self.log_dir, self.verbose = log_dir, verbose
        self._local: dict[int, np.ndarray] = {}
        self._img: dict[int, np.ndarray] = {}
        self._reset()

    # ---- graph

    def _prob_pairs(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(self.scorer.prob(a, b), np.float32)

    def _prob_to_all(self, q: np.ndarray) -> np.ndarray:
        return self._prob_pairs(np.broadcast_to(q, self.emb.shape), self.emb)

    def _shortcuts(self, gap: int, k: int, window: int, thr: float) -> list[tuple[int, int]]:
        """Candidates by embedding cosine (cheap), then verified by R over a window of frames:
        the MEDIAN of R(i+d, j+d), d in [-window, window], must beat ``thr`` (paper eq. 2)."""
        e = self.emb / (np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-9)
        idx = np.arange(self.n)
        pairs: set[tuple[int, int]] = set()
        for i in range(self.n):
            sim = e @ e[i]
            sim[(self.traj == self.traj[i]) & (np.abs(idx - i) <= gap)] = -2
            for j in np.argsort(-sim)[:k]:
                pairs.add((min(i, int(j)), max(i, int(j))))
        lo = {t: s for t, (s, _) in enumerate(self.bounds)}
        hi = {t: e_ for t, (_, e_) in enumerate(self.bounds)}
        a_idx, b_idx, owner = [], [], []
        for p, (i, j) in enumerate(sorted(pairs)):
            for d in range(-window, window + 1):
                ii, jj = i + d, j + d
                if lo[self.traj[i]] <= ii < hi[self.traj[i]] and lo[self.traj[j]] <= jj < hi[self.traj[j]]:
                    a_idx.append(ii), b_idx.append(jj), owner.append(p)
        if not a_idx:
            return []
        pr = self._prob_pairs(self.emb[a_idx], self.emb[b_idx])
        owner = np.array(owner)
        plist = sorted(pairs)
        out = []
        for p, (i, j) in enumerate(plist):
            if np.median(pr[owner == p]) > thr:
                out.append((i, j))
        return out

    def _local_nodes(self, node: int) -> np.ndarray:
        if node not in self._local:
            reach = nx.single_source_shortest_path_length(self.G, node, cutoff=self.local_hops)
            self._local[node] = np.fromiter(reach.keys(), dtype=np.int64)
        return self._local[node]

    # ---- session

    def _reset(self) -> None:
        self.t, self.node, self.hops, self.state, self.log = 0, -1, -1, "init", []
        self.prev_small, self.prev_act, self.stuck, self.queue, self.n_stuck = None, None, 0, [], 0
        self.appr_steps = 0
    #Change 1 Add memory for the last turn and flip count:
        self.last_turn, self.flips = None, 0
    def setup(self, info: SessionInfo) -> None:
        self._reset()
        self.focal = _intrinsic_focal(info)
        self.targets = info.targets
        self.tfeat = [self.matcher.features(t) for t in self.targets]
        tembs = np.asarray(self.scorer.embed(list(self.targets)), np.float32)
        best = (-1, -1, 0)  # (inliers, node, view)
        fallback = (-1.0, -1)
        for k in range(4):
            pk = self._prob_to_all(tembs[k])
            if pk.max() > fallback[0]:
                fallback = (float(pk.max()), int(np.argmax(pk)))
            for i in np.argsort(-pk)[:6]:
                n = self.matcher.match(self.tfeat[k], self._feat(int(i))).n
                if n > best[0]:
                    best = (n, int(i), k)
        self.goal = best[1] if best[0] >= 12 else fallback[1]
        self.paths = nx.single_source_shortest_path(self.G, self.goal)  # goal -> node
        self.reach = np.zeros(self.n, bool)
        self.reach[list(self.paths.keys())] = True
        print(f"goal: node {self.goal} (view {best[2]}, {best[0]} inliers; R-best {fallback[1]})")

    def finish(self, result: Result | None) -> None:
        print("result:", result)
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            out = self.log_dir / f"sptm_{int(time.time())}.json"
            out.write_text(json.dumps({"log": self.log, "result": repr(result)}, default=float))

    def _frame_img(self, i: int) -> np.ndarray:
        if i not in self._img:
            self._img[i] = cv2.imread(str(self.frames[i].path))
        return self._img[i]

    def _feat(self, i: int):
        return self.matcher.features(self._frame_img(i))

    # ---- decisions

    def act(self, obs: Observation):
        self.t += 1
        left = obs.steps_left if self.budget is None else min(obs.steps_left, self.budget - obs.step)
        if left <= 12:
            return self._out(Action.CHECKIN, "budget")
        small = cv2.resize(cv2.cvtColor(obs.image, cv2.COLOR_BGR2GRAY), (40, 30)).astype(np.float32)
        if self.prev_act in ("FORWARD", "BACKWARD"):
            blocked = self.prev_small is not None and float(np.abs(small - self.prev_small).mean()) < 1.0
            self.stuck = self.stuck + 1 if blocked else 0
        self.prev_small = small

        if self.queue:
            return self._out(self.queue.pop(0), "recover")
        if self.stuck >= 2:
            self.stuck, self.n_stuck = 0, self.n_stuck + 1
            turn = Action.RIGHT if self.n_stuck % 2 else Action.LEFT
            self.queue = [(turn, min(8 + 3 * (self.n_stuck % 4), 20))]
            self.node = -1  # forget where we thought we were
            return self._out((Action.BACKWARD, 4), "stuck")

        q = np.asarray(self.scorer.embed([obs.image]), np.float32)[0]
        p = self._prob_to_all(q)
        node = -1
        if self.node >= 0:  # temporally consistent: look near the last estimate first
            cand = self._local_nodes(self.node)
            cand = cand[self.reach[cand]]
            if len(cand):
                j = int(cand[np.argmax(p[cand])])
                if p[j] >= self.s_local:
                    node = j
        if node < 0:  # global search
            node = int(np.argmax(np.where(self.reach, p, -1.0)))
        self.node = node
        path = self.paths[node][::-1]  # node ... goal
        self.hops = len(path) - 1

        if self.hops <= self.near_hops:
            r = self._approach(obs)
            if r is not None:
                return r
        self.appr_steps = 0

        top = min(self.h_max, self.hops)
        l = self.h_min if self.hops >= self.h_min else 0
        for k in range(top, self.h_min - 1, -1):  # furthest waypoint still "reachable"
            if p[path[k]] >= self.s_reach:
                l = k
                break
        wp = path[l]
        name = self.scorer.locomote(obs.image, self._frame_img(wp), wp)
        self.state = f"wp +{l} p={p[wp]:.2f} best={p.max():.2f}"
        n = self.burst if name in ("FORWARD", "BACKWARD") else 1
        return self._out((Action[name], n), "follow")

        # #Change 2: Anti - wobble rule change 
        # name = self.scorer.locomote(obs.image, self._frame_img(wp), wp)
        # # anti-wobble: LEFT, RIGHT, LEFT (each turn undoing the last) -> go forward instead
        # if name in ("LEFT", "RIGHT"):
        #     self.flips = self.flips + 1 if self.last_turn not in (None, name) else 0
        #     self.last_turn = name
        #     if self.flips >= 2:
        #         name, self.flips, self.last_turn = "FORWARD", 0, None
        # else:
        #     self.flips, self.last_turn = 0, None
        # self.state = f"wp +{l} p={p[wp]:.2f} best={p.max():.2f}"
        # n = self.burst if name in ("FORWARD", "BACKWARD") else 1
        # return self._out((Action[name], n), "follow")

    def _approach(self, obs: Observation):
        feat = self.matcher.features(obs.image)
        best = None
        for k, tf in enumerate(self.tfeat):
            m = self.matcher.match(tf, feat)
            if best is None or m.n > best[1].n:
                best = (k, m)
        k, m = best
        if m.n < 12: #Originally 12
            return None
        self.appr_steps += 1
        s = shift_deg(m, self.focal)
        left = SIGN * s + VIEW_OFFSET[k]
        r = scale_ratio(m)
        self.state = f"approach view{k} n={m.n} left={left:+.1f} scale={r:.2f}"
        turn = Action.RIGHT if left > 0 else Action.LEFT
        if k != 0 or abs(left) > 4:
            if k != 0 and abs(left) > 20:
                return self._out((turn, int(np.clip(round(abs(left) / TURN_DEG), 1, 8))), "approach-turn")
            if abs(left) > 4:
                return self._out((turn, int(np.clip(round(abs(left) / TURN_DEG), 1, 3))), "approach-turn")
        tol = self.scale_tol if self.appr_steps <= 25 else 2 * self.scale_tol
        if r < 1 - tol:
            return self._out((Action.FORWARD, int(np.clip(round((1 / r - 1) * 12), 1, 3))), "approach-fwd")
        if r > 1 + tol:
            return self._out((Action.BACKWARD, int(np.clip(round((r - 1) * 12), 1, 2))), "approach-back")
        if abs(left) <= 4 and (m.n >= 25 or self.appr_steps > 25):
            return self._out(Action.CHECKIN, "arrived")
        return self._out((Action.FORWARD, 1), "approach-nudge")

    def _out(self, a, why: str):
        act = a[0] if isinstance(a, tuple) else a
        self.prev_act = act.name if act in (Action.FORWARD, Action.BACKWARD) else None
        if self.verbose or why in ("arrived", "budget"):
            print(f"[{self.t}] node {self.node} hops {self.hops} {why} {a} {self.state}")
        self.log.append(dict(t=self.t, node=self.node, hops=self.hops, why=why, act=str(a), state=self.state))
        return a

    def hud(self) -> list[str]:
        return [f"SPTM node {self.node} goal {getattr(self, 'goal', -1)} hops {self.hops}", self.state]


def main() -> None:
    import cli
    from sptm_models import TorchScorer

    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data directory")
    parser.add_argument("--r", type=Path, default=Path("models/sptm_r.pt"))
    parser.add_argument("--l", type=Path, default=Path("models/sptm_l.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--burst", type=int, default=1)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--s-local", type=float, default=0.7)
    parser.add_argument("--s-reach", type=float, default=0.8)
    parser.add_argument("--s-shortcut", type=float, default=0.9)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()
    data = cli.exploration_data(args, args.data)
    scorer = TorchScorer(args.r, args.l, args.device)
    agent = SptmAgent(
        data, scorer, burst=args.burst, budget=args.budget, s_local=args.s_local,
        s_reach=args.s_reach, s_shortcut=args.s_shortcut, log_dir=args.log_dir, verbose=args.verbose,
    )
    run(agent, cli.challenge(args), **cli.run_options(args))


if __name__ == "__main__":
    main()

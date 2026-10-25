"""
SPTM agent (Savinov, Dosovitskiy, Koltun, ICLR 2018) rebuilt for this challenge.

    uv run source/sptm_agent.py --local 7 --budget 2000 --verbose
    uv run source/sptm_agent.py --challenge <id>

Memory graph : one node per exploration frame; edges between consecutive frames, plus
               "visual shortcuts" between far-apart frames that a learned network R says are
               the same place for a whole WINDOW of neighbouring frames (not one lucky pair).
Localisation : R scores the camera frame against every node. Search near the previous node
               first; fall back to a global search only when nothing nearby is similar enough.
Planning     : shortest path from the estimated node to the goal node (Change 20: weighted
               Dijkstra -- driven edges cost 1, weakly confirmed shortcuts cost more).
Place recog. : the learned R network, or (Change 19, --place anyloc) an AnyLoc-style DINOv2+VLAD
               descriptor with a margin check against look-alike places (source/anyloc.py).
Waypoint     : the furthest node along that path that R still says is "the same place" as
               what we see now (so it is close and reachable).
Locomotion   : a learned network L maps (current frame, waypoint frame) -> FORWARD/LEFT/RIGHT.
Last metre   : same RANSAC servo on the four goal photos as my_agent.py, then CHECKIN.
Exploration  : (Change 9) when navigation goes nowhere -- stuck in a small region, or a full
               look-around at the "goal" finds no goal photo -- drop the graph for a while and
               drive the maze directly from the image: forward while the floor ahead is open,
               a committed turn toward the more open side at walls, goal photos still checked.

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
from geometry import Matcher, bearing_from_e, scale_ratio, shift_deg, yaw_from_e
from my_agent import TURN_DEG, VIEW_OFFSET, _intrinsic_focal
from vis_nav_sdk import Action, Agent, Observation, Result, SessionInfo, run
from vlad import load_frames

SIGN = +1  # LEFT x6 shifted the picture by +26 / +23 deg on the real simulator (two mazes)

# Change 9: autonomous exploration settings
EXPLORE_STEPS = 300  # length of one exploration spell, in agent steps
STUCK_WINDOW, STUCK_UNIQUE = 200, 30  # "going nowhere": < 30 different nodes in the last 200 steps
OPEN = 0.5  # floor depth straight ahead above which the way counts as open (~0.8 m of floor)
# Change 12: steadier exploration in busy-texture areas
OPEN_STOP = 0.3  # once driving forward, only stop below this (hysteresis: no fwd/turn flicker)
SIDE_OPEN = 0.6  # a side whose floor reaches this far is a real opening (side corridor / junction)
BRANCH_P, BRANCH_COOLDOWN = 0.35, 15  # take a seen opening with this chance, at most every 15 steps
SIDE_EXTRA = 0  # Change 23: was 5. Side photos now only drive small, self-correcting moves
# (align / step / then one turn), not an immediate 90-180 deg turn, so a 20-inlier side match is
# safe to use -- and it is what the robot actually sees on the goal node (sptm7_r2: 17-22).

# Change 10: geometric check of shortcuts (ORB-SLAM2-style loop-closure verification)
FOCAL_GUESS = 92.0  # px; the true focal is only known in setup(), after the graph is built (slides: f = 92)

# Change 15b: drive to the goal spot. All four goal photos were taken from the SAME spot, so a
# moderate match with any of them gives the direction to that spot (essential-matrix translation).
GOTO_MIN = 30  # Change 22: was 18. sptm7_r showed the bearing is noise at 18-24 inliers (consecutive
# steps: +142, -43, -81 deg), which made the robot spin; compare_bearing.py's good result came from
# pairs with 40-70 inliers. Only steer by translation when the match is that strong.

# Change 17: keep hold of the front photo once it is being tracked (ORB-SLAM: tracking accepts
# weaker evidence than relocalisation, because the previous frame already said where we are)
TRACK_RELAX = 6  # while tracking the front photo, it only needs goal_min - 6 = 14 inliers
CHECKIN_MIN = 18  # front-photo inliers needed to check in (the near-miss had 20-22)

# Change 21: check-in becomes more willing as the run goes on. Scoring gate: <= 0.2 m and <= 20 deg
# (3 heading points at <= 10 deg). Each stage: (from this fraction of the budget, min front-photo
# inliers, max heading error in deg, max |scale - 1| as a distance proxy). The first stage that
# matches wins. Early: near-perfect only; late: anything plausibly inside the gate.
CHECKIN_STAGES = [  # Change 22: looser and earlier, after sptm7_r step 1565 (front photo n=14,
    (0.0, 16, 8.0, 0.10),  # heading -8.2 deg, scale 1.02) was NOT taken because stage 0 then asked
    (0.3, 14, 10.0, 0.12),  # for n >= 18 and <= 6 deg. All stages stay inside the 20 deg / 0.2 m gate.
    (0.7, 14, 15.0, 0.18),
]

# Change 24: steering toward the waypoint photo by measured angle (teach-and-repeat style)
STEER_MIN = 20  # SuperGlue inliers with the waypoint photo needed to trust the measured angle
STEER_DEG = 10.0  # turn when the measured angle exceeds this; otherwise drive forward

# Change 19 (fix 1A): margin check for relocalisation (VPR tutorial). The best match must beat the
# best match elsewhere in the maze by at least 10% of the typical same-vs-different similarity gap.
MARGIN = 0.10


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
        local_hops: int = 4, #12 originally
        shortcut_gap: int = 10,
        shortcut_k: int = 8,
        window: int = 3,
        near_hops: int = 3,  #3 originally
        burst: int = 1,
        scale_tol: float = 0.04,
        budget: int | None = None,
        log_dir: Path | None = None,
        verbose: bool = False,
        ##########################
        #SuperGlue weights - Change 3a
        goal_matcher = None,
        goal_min: int = 12,
        ##########################
        #Change 10: min SuperGlue inliers for a shortcut to count as a real place match
        geo_min: int = 20,
        #Change 18: exploration mode (Change 9) is OFF unless asked for -- let SPTM navigate
        explore: bool = False,
    ) -> None:
        self.explore_on = explore
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
            self.G.add_edges_from(((i, i + 1) for i in range(s, e - 1)), weight=1.0)  # Change 20: driven = cost 1
        #Change 10: the matchers (and the image cache) are now created BEFORE the shortcuts,
        # because the shortcut check below needs them.
        self.matcher = Matcher()
        ##########################
        #SuperGlue weights - Change 3b
        self.gm = goal_matcher or self.matcher  # matcher for the goal photos (SuperGlue if given)
        self.sg_on = goal_matcher is not None
        self.goal_min = goal_min
        ##########################
        self._img: dict[int, np.ndarray] = {}
        self.shortcuts = self._shortcuts(shortcut_gap, shortcut_k, window, s_shortcut)
        #Change 10: R alone proposes shortcuts, and it confuses look-alike walls (e.g. node 1437,
        # a purple-wall corridor, was joined to the goal area, so the robot thought it was 1 hop
        # from the goal while far away). Like ORB-SLAM2's loop closing: appearance proposes,
        # geometry confirms -- keep a shortcut only if SuperGlue + RANSAC agree. Only with --superglue.
        if self.sg_on:
            self.shortcuts = self._verify_shortcuts(self.shortcuts, geo_min)
        #Change 15b: check the sign of bearing_from_e on our own data before trusting it
        self.bear_flip = self._calibrate_bearing() if self.sg_on else None
        self.bear_ok = self.bear_flip is not None
        #Change 20 (fix 2B, practical substitute for metric planning): MASt3R-SLAM is not usable
        # here (no poses in the exploration data, and its CUDA extensions need Linux), so instead the
        # planner is told HOW SURE each edge is. Edges the robot actually drove cost 1; a shortcut
        # costs 1 + (40 - inliers)/20 if SuperGlue confirmed it with fewer than 40 inliers (so 20
        # inliers -> 2.0, 40+ -> 1.0). Dijkstra then prefers driven and strongly confirmed routes.
        self.G.add_edges_from(self.shortcuts, weight=1.0)
        for (i, j), n_in in getattr(self, "sc_inliers", {}).items():
            self.G[i][j]["weight"] = 1.0 + max(0, 40 - n_in) / 20.0
        print(f"graph: {self.G.number_of_edges()} edges, {len(self.shortcuts)} verified shortcuts")
        self.h_min, self.h_max, self.s_local, self.s_reach = h_min, h_max, s_local, s_reach
        self.local_hops, self.near_hops, self.burst = local_hops, near_hops, burst
        self.scale_tol, self.budget = scale_tol, budget
        self.log_dir, self.verbose = log_dir, verbose
        self._local: dict[int, np.ndarray] = {}
        self._wpf: dict = {}  # Change 24: SuperGlue features of waypoint frames (cache)
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

    def _verify_shortcuts(self, pairs: list[tuple[int, int]], min_n: int) -> list[tuple[int, int]]:
        """Change 10: ORB-SLAM2-style loop-closure check. Keep a shortcut (i, j) only if
        SuperPoint+SuperGlue matches between frames i and j, filtered by essential-matrix
        RANSAC (known focal), leave at least ``min_n`` inliers. In our test on maze-7 frames,
        the same place gave 25-60 inliers and different places at most 16, so 20 separates them.
        Features are computed once per frame and reused; this adds ~1-2 min at start-up."""
        feats: dict = {}

        def feat(i: int):
            if i not in feats:
                feats[i] = self.gm.features(self._frame_img(i))
            return feats[i]

        kept = []
        self.sc_inliers: dict[tuple[int, int], int] = {}  # Change 20: inliers per kept shortcut
        for c, (i, j) in enumerate(pairs, 1):
            n_in = self.gm.match(feat(i), feat(j), FOCAL_GUESS).n
            if n_in >= min_n:
                kept.append((i, j))
                self.sc_inliers[(i, j)] = n_in
            if c % 1000 == 0:
                print(f"  geometric check: {c}/{len(pairs)} shortcuts tested, {len(kept)} kept so far")
        print(f"geometric check (SuperGlue): kept {len(kept)} of {len(pairs)} shortcuts, "
              f"rejected {len(pairs) - len(kept)} look-alikes")
        return kept

    def _calibrate_bearing(self) -> bool | None:
        """Change 15b: self-check of bearing_from_e on exploration frames where the robot drove
        FORWARD. Frame i+1 was taken ahead of frame i, so the bearing from i to i+1 must point
        ahead (cos > 0). Returns False = sign correct, True = sign flipped (add 180 deg),
        None = inconsistent or too little data -> the drive-to-goal step stays switched off."""
        votes: list[bool] = []
        for i in range(self.n - 1):
            if self.traj[i] != self.traj[i + 1] or self.frames[i].action != "FORWARD":
                continue
            ahead = self.gm.features(self._frame_img(i + 1))  # A: the spot further ahead
            here = self.gm.features(self._frame_img(i))  # B: where the camera is
            b = bearing_from_e(self.gm.match(ahead, here, FOCAL_GUESS), FOCAL_GUESS)
            if b is not None:
                votes.append(bool(np.cos(np.radians(b)) > 0))
            if len(votes) >= 20:
                break
        n_ahead = sum(votes)
        if len(votes) < 8:
            verdict = None
        elif n_ahead >= 0.7 * len(votes):
            verdict = False
        elif n_ahead <= 0.3 * len(votes):
            verdict = True
        else:
            verdict = None
        label = {None: "INCONSISTENT -> drive-to-goal OFF", False: "sign OK", True: "sign flipped (fixed)"}[verdict]
        print(f"bearing check: {n_ahead}/{len(votes)} forward pairs point ahead -> {label}")
        return verdict

    def _ambiguous(self, q: np.ndarray, g: int) -> bool:
        """Change 19 (fix 1A): margin check. Compare the best node ``g`` with the best node OUTSIDE
        g's neighbourhood (a genuinely different part of the maze). If their similarities differ by
        less than MARGIN of the same-vs-different gap, the match is ambiguous. Needs an unsaturated
        similarity (the AnyLoc scorer's ``sim``); R's probabilities sit at ~1.00 for many nodes, so
        with R the check is skipped and the 3-frame confirmation (Change 3) does the work alone."""
        if not hasattr(self.scorer, "sim"):
            return False
        s = np.asarray(self.scorer.sim(np.broadcast_to(q, self.emb.shape), self.emb), np.float32)
        other = self.reach.copy()
        other[self._local_nodes(g)] = False
        if not other.any():
            return False
        margin = (s[g] - s[other].max()) / self.scorer.sim_scale
        return bool(margin < MARGIN)

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
    #Change 3: Confirmation before a big jump:
        self.jump_cand, self.jump_n = -1, 0
        #################################################
        #Change 5a:
        self.lock, self.last_appr_turn, self.search_left = 0, Action.RIGHT, 0
        self.scan = 0
        #Change 9: autonomous exploration state (see _explore)
        self.explore = 0  # steps left in the current exploration spell (0 = normal SPTM navigation)
        self.recent: list[int] = []  # last STUCK_WINDOW localised nodes, to notice going nowhere
        self.ex_turn, self.ex_turning, self.ex_force = Action.LEFT, 0, 0  # committed turn side / counters
        self.ex_moving, self.ex_since = False, 0  # Change 12: driving-forward flag / steps since last branch
        self.ex_fwd, self.ex_pref = 0, Action.RIGHT  # Change 13: straight-run length / preferred turn side
        self.rng = np.random.default_rng(0)  # fixed seed: the same run replays the same choices
        #Change 11: forward "creep" steps left after a side-photo turn (see act / _approach)
        self.creep = 0
        #Change 14 (diagnostic only): inliers of the 4 goal photos (front/left/back/right) at the
        # last _approach() call, shown in the goal-scan log lines
        self.goal_ns = [0, 0, 0, 0]
        #Change 17: which goal photo the last approach step used (0 = front, -1 = none) and how
        # many re-acquire sweeps have been tried since it was last seen
        self.last_k, self.reacq = -1, 0
        #################################################


    def setup(self, info: SessionInfo) -> None:
        self._reset()
        self.focal = _intrinsic_focal(info)
        self.targets = info.targets
        # self.tfeat = [self.matcher.features(t) for t in self.targets]
        ##############################################################
        #Change 3c-1: 
        self.tfeat = [self.gm.features(t) for t in self.targets]
        gf = self.focal if self.sg_on else None
        ##############################################################
    
        tembs = np.asarray(self.scorer.embed(list(self.targets)), np.float32)
        best = (-1, -1, 0)  # (inliers, node, view)
        fallback = (-1.0, -1)
        for k in range(4):
            pk = self._prob_to_all(tembs[k])
            if pk.max() > fallback[0]:
                fallback = (float(pk.max()), int(np.argmax(pk)))
            for i in np.argsort(-pk)[:6]:
                # n = self.matcher.match(self.tfeat[k], self._feat(int(i))).n
                ################################################################################
                #Change 3c-2:
                n = self.gm.match(self.tfeat[k], self.gm.features(self._frame_img(int(i))), gf).n
                ################################################################################
                if n > best[0]:
                    best = (n, int(i), k)
        # self.goal = best[1] if best[0] >= 12 else fallback[1]
        #################################################################
        #Change 3c-3: 
        self.goal = best[1] if best[0] >= self.goal_min else fallback[1]
        #################################################################
    
        #Change 20: weighted Dijkstra (was unweighted shortest path): routes prefer driven edges and
        # strongly confirmed shortcuts. `hops` stays the number of edges on the chosen route.
        self.paths = nx.single_source_dijkstra_path(self.G, self.goal, weight="weight")  # goal -> node
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
        #Change 21: how much of the run is used up (0..1); the check-in rule gets more willing late
        total = self.budget if self.budget is not None else obs.step + obs.steps_left
        self.frac = obs.step / max(total, 1)
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

        #Change 9: while exploring, skip SPTM localisation/planning and drive from the image
        if self.explore > 0:
            return self._explore(obs)

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
        # if node < 0:  # global search
        #     node = int(np.argmax(np.where(self.reach, p, -1.0)))
        # self.node = node
        ##################################################################      
        #Change 4: Big Jump part2 
        if node < 0:  # global search
            g = int(np.argmax(np.where(self.reach, p, -1.0)))
            if self.node < 0:
                node = g  # no previous estimate (start, or after "stuck"): accept
            elif self._ambiguous(q, g) and p[self.node] >= 0.5:
                #Change 19b: only hold on to the previous estimate while it still matches the camera
                # (p >= 0.5). sptm7_anyloc froze at node 691 for hundreds of steps with p = 0.00:
                # the robot had moved on, but every new match looked ambiguous, so it never updated.
                #Change 19 (fix 1A, margin check -- VPR tutorial): the best match is barely better
                # than the best match somewhere ELSE in the maze, so it may be a look-alike wall.
                # Don't jump on it; keep the previous estimate and restart the 3-frame confirmation.
                node, self.jump_n, self.jump_cand = self.node, 0, -1
            else:
                # far jump: only accept once 3 frames in a row point to the same area
                near = self.jump_cand >= 0 and bool(np.isin(g, self._local_nodes(self.jump_cand)))
                self.jump_n = self.jump_n + 1 if near else 1
                self.jump_cand = g
                node = g if self.jump_n >= 3 else self.node
        else:
            self.jump_n = 0
        self.node = node

 ##################################################################      
        path = self.paths[node][::-1]  # node ... goal
        self.hops = len(path) - 1

        #Change 9: going nowhere? If the last STUCK_WINDOW steps visited fewer than STUCK_UNIQUE
        # different nodes (e.g. shuttling between the same two walls), the graph route is not
        # working here -> explore the maze directly for a while.
        self.recent.append(node)
        if len(self.recent) > STUCK_WINDOW:
            self.recent.pop(0)
        if (self.explore_on and len(self.recent) == STUCK_WINDOW  # Change 18: only if --explore
                and len(set(self.recent)) < STUCK_UNIQUE and self.lock == 0):
            return self._start_explore(obs, "region")

        # if self.hops <= self.near_hops:
        #     r = self._approach(obs)
        #     if r is not None:
        #         return r
        # self.appr_steps = 0

        #######################################################################################################
        #Change 5b:
        #Change 8c: the goal look-around (goal-scan) runs as one uninterrupted ~360 deg turn.
        # While the robot turns on the spot its view changes, so R relocalises it and `hops`
        # jumps (e.g. 1 -> 4). Before 8c that jump stopped the scan after 2-3 steps, so a full
        # circle never happened. Now: start the scan at hops <= 1, then finish it whatever hops
        # says, and only reset the counter once a finished scan is followed by leaving the goal area.
        if self.hops > 3 and self.scan >= 105:  # Change 8c: was `if self.hops > 3:`
            self.scan = 0  # left the goal area after a full look-around: allow a fresh one next time
        if self.hops <= self.near_hops or self.lock > 0 or 0 < self.scan < 105:  # Change 8c: + scan in progress
            r = self._approach(obs)
            if r is not None:
                self.lock = 12  # Last-Mile: once the goal is seen, stay with the geometric controller
                #Change 11: a goal photo was recognised, so the look-around has done its job. End it
                # here; otherwise the scan's LEFT turns undo the approach's turn toward the goal
                # (seen in sptm7_geo: approach RIGHT 8, then goal-scan LEFT x5, repeated for hundreds of steps).
                if 0 < self.scan < 105:
                    self.scan = 105
                return r
            if self.lock > 0 and self.search_left > 0:  # finish the estimated turn, but no further
                self.lock -= 1
                self.search_left -= 1
                return self._out((self.last_appr_turn, 1), "approach-search")
            #Change 11: turned toward the goal heading from a SIDE photo, but the front photo is not in
            # view yet -> the robot is probably a little off the goal spot: edge forward a few steps
            # (Last-Mile: turn, then creep) instead of letting a scan or L turn it away again.
            if self.lock > 0 and self.creep > 0:
                self.lock -= 1
                self.creep -= 1
                return self._out((Action.FORWARD, 2), "approach-creep")
            #Change 17: the front photo was being tracked a moment ago and one weak frame lost it.
            # sptm7_goto step 2000: aligned to 3.4 deg, 14% too close, backing up -- then the next
            # frame had < 20 inliers, L took over and drove away. Instead, sweep the heading a little
            # (1 tick left, 2 right, 3 left, ... ~+-10 deg) to find the front photo again.
            if self.lock > 0 and self.last_k == 0 and self.reacq < 6:
                self.lock -= 1
                self.reacq += 1
                turn = Action.LEFT if self.reacq % 2 else Action.RIGHT
                return self._out((turn, self.reacq), "approach-reacquire")
            self.lock = 0
            # Change 8b/8c: at the goal but no goal photo matched -> turn LEFT 3 ticks (~10 deg) per step,
            # up to 105 ticks (~360 deg). Each step _approach() above re-checks the 4 goal photos, so the
            # scan ends early as soon as one is recognised and the approach takes over.
            if self.scan < 105 and (self.hops <= 1 or self.scan > 0):  # start at the goal, then finish it
                self.scan += 3
                #Change 14 (diagnostic): log how well each goal photo matched at this heading
                self.state = f"scan {self.scan}/105 goal inliers F/L/B/R=" + "/".join(map(str, self.goal_ns))
                return self._out((Action.LEFT, 3), "goal-scan")
            #Change 9: a full 360 deg look-around at the "goal" saw no goal photo -> the goal is not
            # here (wrong goal node or wrong localisation). Go and explore instead of circling here.
            if self.explore_on and self.scan >= 105 and self.hops <= 1:  # Change 18: only if --explore
                return self._start_explore(obs, "no-goal-here")

        self.appr_steps = 0
        #######################################################################################################
        top = min(self.h_max, self.hops)
        l = self.h_min if self.hops >= self.h_min else 0
        for k in range(top, self.h_min - 1, -1):  # furthest waypoint still "reachable"
            if p[path[k]] >= self.s_reach:
                l = k
                break
        wp = path[l]
        # name = self.scorer.locomote(obs.image, self._frame_img(wp), wp)
        # self.state = f"wp +{l} p={p[wp]:.2f} best={p.max():.2f}"
        # n = self.burst if name in ("FORWARD", "BACKWARD") else 1
        # return self._out((Action[name], n), "follow")

        #Change 24: steer by the MEASURED angle to the waypoint photo (SuperGlue + yaw from the
        # essential matrix, VO tutorial) and turn by that whole angle at once, then drive. In
        # sptm7_side the robot spent ~40 steps at one left corner flipping LEFT/RIGHT (L sits on its
        # turn-or-go-straight boundary there) and then turned ~95 deg in 28 one-tick decisions.
        # L remains the fallback whenever the waypoint photo does not match well enough.
        if self.sg_on:
            g = self._geo_steer(obs, wp, l)
            if g is not None:
                return g

        #Change 2: Anti - wobble rule change
        name = self.scorer.locomote(obs.image, self._frame_img(wp), wp)
        # anti-wobble: LEFT, RIGHT, LEFT (each turn undoing the last) -> go forward instead
        if name in ("LEFT", "RIGHT"):
            self.flips = self.flips + 1 if self.last_turn not in (None, name) else 0
            self.last_turn = name
            if self.flips >= 2:
                name, self.flips, self.last_turn = "FORWARD", 0, None
        else:
            self.flips, self.last_turn = 0, None
        self.state = f"wp +{l} p={p[wp]:.2f} best={p.max():.2f}"
        n = self.burst if name in ("FORWARD", "BACKWARD") else 1
        return self._out((Action[name], n), "follow")

    def _geo_steer(self, obs: Observation, wp: int, l: int):
        """Change 24: visual teach-and-repeat step toward waypoint frame ``wp`` (``l`` hops ahead).
        Match the live view with the waypoint photo; if they share enough structure, read the
        rotation between them (yaw_from_e, sign verified by tools/compare_yaw.py: + = the robot is
        turned left of the waypoint view, so turn right) and turn by that angle (<= 6 ticks ~ 20 deg
        per step), or drive forward 2 ticks when already lined up. None -> let L decide."""
        if wp not in self._wpf:
            if len(self._wpf) > 200:  # keep GPU memory bounded
                self._wpf.clear()
            self._wpf[wp] = self.gm.features(self._frame_img(wp))
        m = self.gm.match(self._wpf[wp], self.gm.features(obs.image), self.focal)
        if m.n < STEER_MIN:
            return None
        y = yaw_from_e(m, self.focal)
        if y is None:
            return None
        y = SIGN * y  # same convention as _approach: + = turn right
        self.flips, self.last_turn = 0, None  # a measured turn is not a wobble
        self.state = f"wp +{l} steer yaw={y:+.0f} n={m.n}"
        if abs(y) > STEER_DEG:
            turn = Action.RIGHT if y > 0 else Action.LEFT
            return self._out((turn, int(np.clip(round(abs(y) / TURN_DEG), 1, 6))), "steer-turn")
        return self._out((Action.FORWARD, 2), "steer-fwd")

    # ---- Change 9: autonomous exploration

    def _start_explore(self, obs: Observation, why: str):
        """Begin an exploration spell of EXPLORE_STEPS steps (``why`` is only for the log)."""
        self.explore, self.ex_turning, self.ex_force = EXPLORE_STEPS, 0, 0
        self.recent, self.lock = [], 0
        #Change 13: each spell follows walls on one side; alternate the side between spells so a
        # second spell explores the parts the first one skipped.
        self.ex_pref = Action.LEFT if self.ex_pref == Action.RIGHT else Action.RIGHT
        self.ex_fwd, self.ex_moving = 0, False
        print(f"[{self.t}] explore start ({why}) at node {self.node} hops {self.hops}, prefers {self.ex_pref.name}")
        return self._explore(obs)

    @staticmethod
    def _free_space(image: np.ndarray) -> tuple[float, float, float]:
        """Free floor depth (0..1) below the horizon: (ahead, left, right) -- see Change 12 below.
        The maze floor is light grey or light blue and untextured, walls are textured, and the
        ceiling (also light) is above the horizon, so the lower image is enough. A wall close
        ahead covers the lower centre, so ``ahead`` drops as the robot nears it."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        grey = (sat < 25) & (val > 190)  # light grey floor
        blue = (hue > 95) & (hue < 125) & (sat > 25) & (sat < 110) & (val > 170)  # light blue floor tiles
        floor = (grey | blue)[int(0.5 * h):].astype(np.uint8)  # below the horizon only
        #Change 12: busy wall textures contain light grey/blue patches that used to count as floor,
        # so `ahead` jumped around. Now: (1) close small holes in the floor (tile seams), then
        # (2) per image column, measure only the floor that is CONNECTED to the bottom edge -- the
        # run of floor pixels from the robot's feet up to the first wall pixel. A patch on a wall
        # above the wall's base is not connected, so it no longer counts. The run length is a
        # "free depth": 0 = wall at the robot's feet, 1 = floor all the way to the horizon.
        floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
        up = floor[::-1]  # row 0 = bottom of the image (the robot's feet)
        run = np.where(up.all(axis=0), up.shape[0], np.argmin(up, axis=0))  # first non-floor row
        depth = run / up.shape[0]
        third = w // 3
        ahead = float(np.median(depth[int(0.4 * w): int(0.6 * w)]))
        return ahead, float(np.median(depth[:third])), float(np.median(depth[-third:]))

    def _explore(self, obs: Observation):
        """One exploration step: drive along open floor, turn away from walls, no graph.
        - every 3rd step look for the goal photos, with a stricter threshold (+10) because
          false matches reach ~23 inliers; a hit hands over to the approach (lock)
        - floor open ahead -> FORWARD 3 ticks (Change 12: with hysteresis, see OPEN/OPEN_STOP);
          where a side really opens up (side floor depth >= SIDE_OPEN) it sometimes takes that
          side corridor (~90 deg), so the robot covers branches instead of always going straight
        - wall ahead -> pick the more open side ONCE and keep turning that way until the
          floor ahead opens up again (a committed turn, so no left-right wobble)
        - driving into a wall anyway is caught by the existing stuck detector (back off + turn)"""
        self.explore -= 1
        if self.explore == 0:  # spell over: back to SPTM, relocalising from scratch
            self.node, self.scan, self.recent = -1, 0, []
        if self.t % 3 == 0:
            r = self._approach(obs, extra=10)
            if r is not None:
                self.explore, self.lock, self.node = 0, 12, -1  # goal seen: approach takes over
                return r
        ahead, lft, rgt = self._free_space(obs.image)
        self.state = f"explore {self.explore} ahead={ahead:.2f} L={lft:.2f} R={rgt:.2f}"
        self.ex_since += 1
        if self.ex_force > 0:  # finishing a side-corridor turn
            #Change 13: stop the branch turn as soon as the side corridor is straight ahead (at least
            # ~50 deg turned and the floor ahead is open). sptm7_ex2 kept turning the full ~90 deg
            # with ahead=0.77 and overshot the corridor it was turning into.
            if self.ex_force <= 3 and ahead >= OPEN:
                self.ex_force, self.ex_moving, self.ex_fwd = 0, True, 0
                return self._out((Action.FORWARD, 3), "explore-fwd")
            self.ex_force -= 1
            return self._out((self.ex_turn, 3), "explore-branch")
        #Change 12: hysteresis -- start driving at OPEN, keep driving until below OPEN_STOP, so the
        # robot no longer flips forward/turn every step when `ahead` hovers around one value.
        go = ahead >= (OPEN_STOP if self.ex_moving else OPEN)
        if go or self.ex_turning > 36:  # open (or turned a full circle: just try forward)
            self.ex_turning, self.ex_moving = 0, True
            #Change 12: only branch where a side really opens up (floor reaching far to that side,
            # i.e. a side corridor or junction), not at random in the middle of a corridor, and not
            # more often than every BRANCH_COOLDOWN steps. ~90 deg = 9 x 3 ticks.
            sides = [(s, a) for s, a in ((lft, Action.LEFT), (rgt, Action.RIGHT)) if s >= SIDE_OPEN]
            #Change 13: only branch after driving straight for >= 3 steps. Right after a turn the
            # side columns still see the corridor the robot just turned into, which looked like a
            # side opening and made it turn ~90 deg more (sptm7_ex2 step 224).
            if (sides and self.ex_fwd >= 3 and self.ex_since >= BRANCH_COOLDOWN
                    and self.rng.random() < BRANCH_P):
                self.ex_turn = max(sides, key=lambda x: x[0])[1]
                self.ex_force, self.ex_since, self.ex_moving, self.ex_fwd = 8, 0, False, 0
                return self._out((self.ex_turn, 3), "explore-branch")
            self.ex_fwd += 1  # Change 13: consecutive forward steps
            return self._out((Action.FORWARD, 3), "explore-fwd")
        self.ex_moving, self.ex_fwd = False, 0
        if self.ex_turning == 0:  # a new wall: choose a side once and stick to it
            if abs(lft - rgt) < 0.1:
                #Change 13: when neither side is clearly more open (often L=R=0 at a wall), turn to
                # this spell's preferred side instead of a coin flip. Always preferring the same side
                # works like a wall follower: it keeps working along the maze instead of reversing.
                self.ex_turn = self.ex_pref
            else:
                self.ex_turn = Action.LEFT if lft > rgt else Action.RIGHT
        self.ex_turning += 1
        return self._out((self.ex_turn, 3), "explore-turn")

    def _goto(self, bearing: float, k: int, m):
        """Change 15b: drive toward the goal spot (VO tutorial: t points to the other camera).
        ``bearing`` is the direction to the spot goal photo ``k`` was taken from, in the live
        view (+ = right). Turn toward it in steps of <= 6 ticks (~20 deg), then drive forward.
        Each step re-matches the photos, so the direction is corrected as the robot moves; once
        the front photo matches well enough, the normal alignment and check-in take over."""
        if self.bear_flip:  # start-up check found the sign reversed: point the other way
            bearing = ((bearing + 360.0) % 360.0) - 180.0
        self.search_left = self.creep = 0
        self.last_k, self.reacq = k, 0  # Change 17
        self.state = f"goto view{k} n={m.n} bearing={bearing:+.0f}"
        if abs(bearing) > 20:  # the bearing is good to ~10-20 deg (compare_bearing.py), so don't fuss below that
            turn = Action.RIGHT if bearing > 0 else Action.LEFT
            return self._out((turn, int(np.clip(round(abs(bearing) / TURN_DEG), 1, 6))), "approach-goto-turn")
        return self._out((Action.FORWARD, 3), "approach-goto-fwd")

    def _side_align(self, k: int, err: float, r: float):
        """Change 23: two-axis approach with the 4 goal photos (all taken from ONE spot, 90 deg apart).
        ``err`` = heading error relative to side/back photo ``k`` itself (+ = turn right), ``r`` = how
        much bigger the scene looks than in that photo (> 1: too close, < 1: too far).
          1. face the same way as photo k (small turns, <= 3 ticks)
          2. drive forward/back until the scene has the same size -> on the spot along that axis
          3. turn by photo k's offset to face front; the front photo then fixes the other axis
             and checks in (it is tracked / re-acquired right after this turn)."""
        self.search_left = self.creep = 0
        if abs(err) > 8:
            turn = Action.RIGHT if err > 0 else Action.LEFT
            self.last_appr_turn = turn
            return self._out((turn, int(np.clip(round(abs(err) / TURN_DEG), 1, 3))), f"approach-side{k}-turn")
        if r < 0.9:
            return self._out((Action.FORWARD, 2), f"approach-side{k}-fwd")
        if r > 1.1:
            return self._out((Action.BACKWARD, 2), f"approach-side{k}-back")
        off = VIEW_OFFSET[k]  # facing photo k now, at the right distance: turn to face front
        turn = Action.RIGHT if off > 0 else Action.LEFT
        self.last_appr_turn = turn
        self.last_k, self.reacq = 0, 0  # next: look for the front photo (re-acquire sweep if needed)
        return self._out((turn, int(round(abs(off) / TURN_DEG))), f"approach-side{k}-rotate")

    def _approach(self, obs: Observation, extra: int = 0):
        # feat = self.matcher.features(obs.image)
        # best = None
        # for k, tf in enumerate(self.tfeat):
        #     m = self.matcher.match(tf, feat)
        #     if best is None or m.n > best[1].n:
        #         best = (k, m)
        # k, m = best
        # if m.n < 12: #Originally 12
        #     return None
        ####################################################
        #Change 3d: 
        feat = self.gm.features(obs.image)
        gf = self.focal if self.sg_on else None
        ms = [self.gm.match(tf, feat, gf) for tf in self.tfeat]
        self.goal_ns = [x.n for x in ms]  # Change 14: remember for the log
        min_n = self.goal_min + extra  # 16 is noise level; Change 9: +extra while exploring (false hits reach ~23)
        #Change 17: while already tracking the front photo (last step used it, still locked on),
        # accept it with TRACK_RELAX fewer inliers, so one weaker frame does not drop the goal.
        tracking = self.sg_on and self.lock > 0 and self.last_k == 0
        front_min = min_n - TRACK_RELAX if tracking else min_n
        if ms[0].n >= front_min:  # front photo visible: it alone fixes the goal heading and distance
            k, m = 0, ms[0]
        else:  # side/back photos only steer coarsely (90-180 deg turns), so demand a clearly stronger match
            k = int(np.argmax([x.n for x in ms]))
            m = ms[k]
            #Change 12: a side/back photo triggers a 30-180 deg turn, and in every run so far a
            # side match of 20-23 inliers near the goal turned the robot AWAY from it (sptm7_creep:
            # view1 n=20 -> turn right + creep -> hops 1 became hops 5-6). Ask a bit more of them.
            side_min = min_n + (SIDE_EXTRA if self.sg_on and k != 0 else 0)
            if m.n < side_min:
                #Change 15b: not strong enough to trust a turn-on-the-spot, but a moderate match with
                # ANY goal photo still tells us where the goal spot is (all 4 photos were taken there).
                # sptm7_diag showed exactly this case: the robot stood near but OFF the goal spot,
                # with left/back photos at 20-23 and the front at 17 -- turning alone could never work.
                if self.bear_ok and m.n >= GOTO_MIN + extra:
                    b = bearing_from_e(m, self.focal)
                    if b is not None:
                        return self._goto(b, k, m)
                return None

        #########################################################################
        ####################################################

        self.appr_steps += 1
        self.search_left = 0
        self.creep = 0  # Change 11: a fresh goal-photo match replaces any pending creep
        self.last_k, self.reacq = k, 0  # Change 17: remember which photo we are tracking
        y = yaw_from_e(m, self.focal) if self.sg_on else None  # VO tutorial: yaw from R, not biased by translation
        s = y if y is not None else shift_deg(m, self.focal)

        left = SIGN * s + VIEW_OFFSET[k]
        r = scale_ratio(m)
        self.state = f"approach view{k} n={m.n} left={left:+.1f} scale={r:.2f}"
        #Change 21: commit as soon as the FRONT photo says we are inside a (slightly safe) version
        # of the scoring gate, instead of insisting on perfect alignment. Not checking in scores 0
        # anyway, so a near-miss is worth taking -- more so as the run goes on (see CHECKIN_STAGES).
        if k == 0 and self.sg_on:
            for frac_from, ci_n, ci_head, ci_scale in CHECKIN_STAGES:
                if getattr(self, "frac", 0.0) >= frac_from and m.n >= ci_n and abs(left) <= ci_head \
                        and abs(r - 1.0) <= ci_scale:
                    self.state += f" checkin-stage>={frac_from:.0%}"
                    return self._out(Action.CHECKIN, "arrived")
        #Change 23: a SIDE/BACK photo is used to fix the robot's POSITION along that photo's direction
        # first (align with it, then drive until the scene has the same size), and only then turn
        # by that photo's offset to face front. Before, the robot turned 90-180 deg at once from
        # wherever it stood, so it never got onto the spot (sptm7_r2: on the goal node, left/right
        # photos at 17-22 inliers, front never above 18).
        if k != 0 and self.sg_on:
            return self._side_align(k, left - VIEW_OFFSET[k], r)
        turn = Action.RIGHT if left > 0 else Action.LEFT
        ############################
        #Change 5d:
        self.last_appr_turn = turn
        ############################
        # if k != 0 or abs(left) > 4:
        #     if k != 0 and abs(left) > 20:
        #         return self._out((turn, int(np.clip(round(abs(left) / TURN_DEG), 1, 8))), "approach-turn")
        #     if abs(left) > 4:
        #         return self._out((turn, int(np.clip(round(abs(left) / TURN_DEG), 1, 3))), "approach-turn")
        
########################################################################################################################
# Change: 6-5:        
        need = int(round(abs(left) / TURN_DEG))  # ticks to face the goal heading
        if k != 0 or abs(left) > 4:
            if k != 0 and abs(left) > 20:
                n_t = int(np.clip(need, 1, 8))
                self.search_left = need - n_t  # the rest of the turn, if the match is lost
                self.creep = 4  # Change 11: after this side-photo turn, edge forward up to 4 x 2 ticks
                return self._out((turn, n_t), "approach-turn")
            if abs(left) > 4:
                n_t = int(np.clip(need, 1, 3))
                self.search_left = need - n_t
                return self._out((turn, n_t), "approach-turn")
######################################################################################################################
        #Change 17: widen the distance tolerance after 10 approach steps (was 25), so the robot does
        # not shuffle back and forth for ages around the exact spot
        tol = self.scale_tol if self.appr_steps <= 10 else 2 * self.scale_tol
        if r < 1 - tol:
            return self._out((Action.FORWARD, int(np.clip(round((1 / r - 1) * 12), 1, 3))), "approach-fwd")
        if r > 1 + tol:
            return self._out((Action.BACKWARD, int(np.clip(round((r - 1) * 12), 1, 2))), "approach-back")
        #Change 17: check in once the FRONT photo is lined up (heading within 4 deg, distance within
        # tol) with at least CHECKIN_MIN inliers. Before: >= 25 inliers or > 25 approach steps, which
        # the robot never reached at the goal (20-22 inliers there). The score gate is 20 deg / 0.2 m.
        if k == 0 and abs(left) <= 4 and m.n >= CHECKIN_MIN:
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
    # Change 16: default step budget 4000 (was None = only the server's limit). The agent checks in
    # at budget - 12; on a real challenge the server's own limit (obs.steps_left) still applies.
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--s-local", type=float, default=0.7)
    parser.add_argument("--s-reach", type=float, default=0.8)
    parser.add_argument("--s-shortcut", type=float, default=0.9)
    #######################################################################################################
    #Change 3e: 
    parser.add_argument("--superglue", action="store_true", help="SuperPoint+SuperGlue for the goal photos")
    #Change 10: shortcut check threshold (only used with --superglue)
    parser.add_argument("--geo-min", type=int, default=20, help="min SuperGlue inliers to keep a shortcut")
    #Change 13: the local simulator draws a fresh random motion seed every run, so two runs differ
    # even with identical code. A fixed seed replays the same motion noise -> fair before/after tests.
    parser.add_argument("--motion-seed", type=int, default=None, help="local runs only: replay this motion seed")
    #Change 18: exploration mode is off by default; pass --explore to switch it back on
    parser.add_argument("--explore", action="store_true", help="enable the autonomous exploration mode (Change 9)")
    #Change 19: which place recognition to use: the trained R network, or AnyLoc (DINOv2 + VLAD)
    parser.add_argument("--place", choices=["r", "anyloc"], default="r", help="place recognition: r | anyloc")
    #######################################################################################################
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()
    data = cli.exploration_data(args, args.data)
    scorer = TorchScorer(args.r, args.l, args.device)
    #Change 19 (fix 1C): --place anyloc swaps R (place recognition) for the AnyLoc-style descriptor;
    # L (steering) stays the trained network. The vocabulary is built once from the exploration data.
    if args.place == "anyloc":
        from anyloc import AnyLocScorer
        scorer = AnyLocScorer(scorer, device=args.device)
        scorer.fit(data)
    # agent = SptmAgent(
    #     data, scorer, burst=args.burst, budget=args.budget, s_local=args.s_local,
    #     s_reach=args.s_reach, s_shortcut=args.s_shortcut, log_dir=args.log_dir, verbose=args.verbose,
    # )
#############################################################################################################
# Change 3f:
    goal_matcher = None
    if args.superglue:
        from sg_matcher import SGMatcher
        goal_matcher = SGMatcher(device=args.device)
    agent = SptmAgent(
        data, scorer, burst=args.burst, budget=args.budget, s_local=args.s_local,
        s_reach=args.s_reach, s_shortcut=args.s_shortcut, log_dir=args.log_dir, verbose=args.verbose,
        goal_matcher=goal_matcher, goal_min=20 if args.superglue else 12,
        geo_min=args.geo_min,  # Change 10
        explore=args.explore,  # Change 18
    )
##################################################################################################################
    #Change 13: pass the motion seed to the local simulator (ignored for real challenge runs)
    local_opts = {"motion_seed": args.motion_seed} if args.local is not None and args.motion_seed is not None else {}
    run(agent, cli.challenge(args), **cli.run_options(args), **local_opts)


if __name__ == "__main__":
    main()

"""
SPTM agent (Savinov, Dosovitskiy, Koltun, ICLR 2018) rebuilt for this challenge.

    uv run source/sptm_agent.py --local 7 --budget 2000 --verbose
    uv run source/sptm_agent.py --challenge <id>

Memory graph : one node per exploration frame; edges between consecutive frames, plus
               "visual shortcuts" between far-apart frames that a learned network R says are
               the same place for a whole WINDOW of neighbouring frames (not one lucky pair).
               (Change 25) The drives are joined at the spawn, where they all start, and a
               shortcut is kept only if SuperGlue confirms a long RUN of consecutive pairs.
Localisation : R scores the camera frame against every node. Search near the previous node
               first; fall back to a global search only when nothing nearby is similar enough.
Planning     : shortest path from the estimated node to the goal node (Change 20: weighted
               Dijkstra -- driven edges cost 1, weakly confirmed shortcuts cost more).
Place recog. : the learned R network, or (Change 19, --place anyloc) an AnyLoc-style DINOv2+VLAD
               descriptor with a margin check against look-alike places (source/anyloc.py).
Waypoint     : the furthest node along that path that R still says is "the same place" as
               what we see now (so it is close and reachable).
Locomotion   : a learned network L maps (current frame, waypoint frame) -> FORWARD/LEFT/RIGHT.
Last metre   : same RANSAC servo on the four goal photos as bayes_agent.py, then CHECKIN.
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
from bayes_loc import BayesLocaliser
from geometry import Matcher, bearing_from_e, scale_ratio, shift_deg, yaw_from_e
from bayes_agent import TURN_DEG, VIEW_OFFSET, _intrinsic_focal
from vis_nav_sdk import Action, Agent, Observation, Result, SessionInfo, run
from vlad import REVERSE, load_frames

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
STEER_MIN = 40  # Change 26: was 20. SuperGlue inliers with the waypoint photo needed to trust the
# measured angle. In the first run with Change 24 (maze 7, 2026-10-26) matches of 20-29 inliers gave
# +71, +52, -64, -40 deg on consecutive steps: 3868 of 3993 ticks were turns, 35 moves forward.
STEER_DEG = 10.0  # turn when the measured angle exceeds this; otherwise drive forward
STEER_SPIN = 5  # Change 26: at most this many measured turns in a row (~100 deg) ...
STEER_REST = 10  # ... or one that undoes the last; then L steers for this many steps
STALL_FWD = 40  # Change 26: the watchdog only drops a shortcut after >= 40 forward ticks without progress
PACE_TICKS = 4000  # Change 26: CHECKIN_STAGES pace over the first 4000 ticks, whatever the step limit

# Change 27: a far relocalisation jump must be confirmed by SuperGlue: >= JUMP_MIN inliers between the
# live view and the new node, and JUMP_MARGIN more than with the current node. Ground truth on the
# 2026-10-26 maze 7 run (125 far jumps): this blocks all 14 jumps that moved the belief > 0.5 m further
# from the truth (median 8 inliers) and keeps 14 of the 20 that corrected it (median 58).
JUMP_MIN, JUMP_MARGIN = 40, 10

# Change 44 (--think-confused): time is not scored, steps are. When the believed photo has not changed
# for CONFUSED_STEPS decisions (stuck, searching for the waypoint, undoing without progress), think before
# moving: check the live view with SuperGlue against the next THINK_AHEAD photos of the route, and believe a
# route photo that clearly matches better (JUMP_MIN, and JUMP_MARGIN more than the believed photo). Only
# forward along the route, and not within near_hops + THINK_AHEAD of the goal, where those photos ARE the
# goal area and an early "arrived" started a 360-deg goal scan again and again.
CONFUSED_STEPS, THINK_AHEAD = 3, 5

# Change 45 (--goal-place): the goal is ONE exploration photo, the best SuperGlue match with any of the 4
# goal photos among AnyLoc's top candidates -- and setup checked only the top 6 per goal photo. On the
# local mazes the photos really at the goal spot can rank much lower (maze 29: 45th-53rd), so a look-alike
# won and the robot drove to the wrong place from the first step (maze 29: 3.4 m away, 58 inliers, while
# the goal-spot photo 1037 has 95). Checked on 13 mazes, 7 of them never used before: the top-6 rule picked
# a goal photo metres away in 4 (29, 33, 34, 67); checking the top GOAL_TOPK picks the strongest match and
# fixed 3 of them (67 has no distinctive photo of the goal spot at all), never choosing worse. ~10-15 s
# once in setup, no steps. (Linking other passes of the goal spot was tried too and dropped: on maze 70 a
# link put the robot into the close approach at a look-alike 0.5-0.8 m from the goal.)
GOAL_TOPK = 64

# Change 46 (--confirm-checkin): the approach checked in on the FIRST reading inside the gate. Near the
# goal the front-photo readings are noisy (15-29 inliers), and twice one fluke ended a run that had found
# the goal: maze 25 checked in 0.28 m off (one reading: 21 inliers, +7.6 deg, scale 1.08, between -46 and
# -32 deg), maze 70 0.81 m off facing backwards (25 inliers, +3.1 deg, scale 1.02, after 0.72-0.90).
# Now a passing reading is confirmed CHECKIN_CONFIRM more times in a row, each from a fresh view: a 1-tick
# turn toward the goal heading (waiting would re-read the same picture). About 2 ticks per real arrival.
# Maze 70 then showed the look-alike is not always a fluke: 3 fresh readings agreed (25 inliers, -0.6..+4.6
# deg, scale 0.95-1.02) 0.81 m away, facing backwards. What separates them is the evidence: every correct
# check-in so far had 43-122 inliers (6 dev runs: 84-104), every wrong one 21-25 (false matches reach ~23).
# So a check-in also needs CHECKIN_STRONG inliers; a weaker "arrival" hands back to navigation instead:
# the close approach (and its 360-deg look-around) pauses for APPROACH_PAUSE decisions while the robot is
# not on the goal photo itself, so the route drives it to the goal photo (maze 70 hovered at a look-alike
# 0.5-0.8 m away, refusing it 94 times, until the budget ran out).
CHECKIN_CONFIRM, CHECKIN_STRONG, APPROACH_PAUSE = 2, 35, 15

REV_MAX = 12  # Change 28: undo-steps in a row without reaching an earlier photo, then re-plan (Change 32)

# Change 32: fixes from the code audit of 2026-10-26 (things that confused the robot)
SEEN_P = 0.3  # waypoint photo with p below this is not in view ...
SEARCH_TICKS = 105  # ... so turn on the spot, up to a full circle (~3.44 deg per tick), to find it
SPAWN_FRAMES = 5  # the first frames of every drive are at the spawn, where every run starts

# Change 33 (from c32_m7): the planner zig-zagged between two drives that run along the same corridor
# (route from node 65: 7 switches within ~25 photos), because a shortcut cost the same as a driven step.
# Every switch makes the next waypoint a photo from another drive, which L was never trained on.
SWITCH_COST = 4.0  # extra cost of every shortcut: switch drives only when it saves real distance
SEARCH_AFTER = 4  # waypoint out of view this many decisions in a row before turning to look for it

# Change 34: the front and back goal photos both look along the goal heading, so lining up with them
# fixes only the distance along that line. c33_m7 checked in -0.01 m along it but 0.34 m to the side.
# Before checking in, also line up with the left (or else the right) goal photo: that fixes sideways.
SIDE_TICKS = int(round(90 / TURN_DEG))  # a ~90 deg turn
# Change 35: when SuperGlue confirms a far jump while the photo being left still matches the live view
# this well, both photos show the robot's current place: link them (run-time loop closing, lecture 8).
# c33_m8: the belief moved from drive 0's photo (225 hops) to drive 1's photo of the same spot (444
# hops); with no link between them the plan flipped and the robot drove away from the goal.
LINK_OLD = 15

# Change 38 (--smooth-steer): one steady steering decision instead of three controllers voting per step.
# t_m52 stalled 468 decisions with L flipping LEFT/RIGHT every step (its argmax sits on the boundary
# when the waypoint is straight ahead); t_m7 stalled 500 with the waypoint search turning LEFT while
# L turned RIGHT. Smooth L's probabilities over time, read "left ~ right" as "ahead", commit to a turn
# direction, and make the search turn the way L wants.
EMA_KEEP = 0.6  # weight of the past in L's smoothed probabilities
TURN_MARGIN = 0.15  # a turn must beat FORWARD (and the other turn) by this much

# Change 39 (--goal-watch): the approach used to start only within near_hops (3) hops on the map. In
# t_m8 the robot stood 0.37 m from the goal while the map said 28 hops (the drive went the long way
# round there), so it never approached. Every GOAL_WATCH_EVERY decisions, match the 4 goal photos;
# a strong match (false matches reach ~23 inliers) starts the approach whatever the hops say.
GOAL_WATCH_EVERY = 5
GOAL_SEEN = 40

# Change 40 (--steer-hold): hysteresis for the measured-angle steering (Change 24). Taking control needs
# STEER_MIN inliers with the waypoint photo; keeping it, when the previous decision was a measured one,
# needs only STEER_KEEP -- tracking accepts weaker evidence than (re)detection (ORB-SLAM keeps tracking
# on ~30 inliers but relocalises only on ~50; lecture 10: short-term tracker + stricter re-detector;
# Change 17 does the same for the front goal photo). The dev-challenge dry run lost ~715 ticks at nodes
# 3512-3520: a measured turn at 40-48 inliers, then L undid it whenever the match dipped below 40.
STEER_KEEP = 30

# Change 43 (with --map-links): a map link joins two photos that agree on the map AND in the images, but
# on the local mazes 1-3% of them had a short wall stub between the photos (two views of one open area:
# no image test separated them -- sideways direction, homography fit and runs of matching pairs all
# failed). So a link is watched once it is among the next LINK_LOOK edges of the route: a bump (the stuck
# check) or no fewer hops within LINK_PATIENCE decisions drops it at once, and the planner goes another
# way. Like Change 25c's dropped shortcuts, it is back for the next session.
LINK_LOOK, LINK_PATIENCE = 3, 30

# Change 29: no progress -> the next DETOUR_EDGES edges of the route cost DETOUR_COST times more, so
# the planner tries another way if there is one (repeated stalls compound; nothing is deleted)
# Change 41 (--detour-once): ... and each edge is made costlier at most once. Repeated stalls on one
# stretch compounded x4, x16, x64: in a dev-challenge run photos 843 and 846, three apart on one drive,
# ended up 177 and 626 hops from the goal, and the belief flipping between them flipped the plan every
# step (decisions 1120-1250); maze 7 (s1 suite) blew its route up from 103 to 415 hops the same way.
DETOUR_EDGES, DETOUR_COST = 10, 4.0

# Change 19 (fix 1A): margin check for relocalisation (VPR tutorial). The best match must beat the
# best match elsewhere in the maze by at least 10% of the typical same-vs-different similarity gap.
MARGIN = 0.10

# Change 25: a map without false shortcuts. Ground truth on local mazes 7 and 8 (re-recorded with
# poses): ~2/3 of the SuperGlue-verified shortcuts joined places > 0.5 m apart (repeated wall
# textures), and the planned route used 3 of them on both mazes, so the robot circled ~5 m from the
# goal believing it was 1-2 hops away. No inlier threshold fixed that (>= 60 still left one on the route).
MIN_RUN = 8  # Change 25b: keep a shortcut only on a run of >= 8 verified pairs (0 false on both mazes)
RUN_GAP = 2  # pairs allowed to be missing inside such a run
STALL_STEPS = 150  # Change 25c: no fewer hops to the goal for this many steps -> drop the next shortcut


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
        #Change 25: join the drives at the spawn; sequence check of shortcuts (0 or 1 = off)
        join_starts: bool = True,
        min_run: int = MIN_RUN,
        #Change 26: steer by the measured angle to the waypoint photo (Change 24), else L only
        steer: bool = True,
        #Change 37: "argmax" = best match + jump rules, "bayes" = Bayes filter over all photos
        localiser: str = "argmax",
        #Change 38: smoothed, committed steering (see EMA_KEEP)
        smooth_steer: bool = False,
        #Change 39: check the goal photos every few steps, not only near the goal node
        goal_watch: bool = False,
        #Change 40/41: keep measured steering down to STEER_KEEP inliers / make each edge costlier once
        steer_hold: bool = False,
        detour_once: bool = False,
        #Change 42: a map in metres adds verified same-place links (map_ticks: and every edge costs ticks)
        map_links: bool = False,
        map_ticks: bool = False,
        #Change 44: when the believed photo stays the same, check the next route photos with SuperGlue
        think_confused: bool = False,
        #Change 45: choose the goal photo among far more candidates
        goal_place: bool = False,
        #Change 46: check in only when the gate holds on fresh readings in a row
        confirm_checkin: bool = False,
    ) -> None:
        self.explore_on = explore
        self.steer_on = steer
        self.smooth_steer = smooth_steer
        self.goal_watch = goal_watch
        self.steer_hold, self.detour_once = steer_hold, detour_once
        self.think_confused, self.same_n = think_confused, 0  # Change 44
        self.goal_place = goal_place  # Change 45
        self.confirm_checkin, self.ci_ok, self.appr_pause = confirm_checkin, 0, 0  # Change 46
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
        #Change 25a: every exploration drive starts at the spawn (true in all 42 random local mazes
        # checked, within 0.16 m), so join the drives there. Then every node is reachable over
        # driven edges alone, whatever happens to the shortcuts. These edges are never removed.
        self.trusted = {(i, i + 1) for s, e in self.bounds for i in range(s, e - 1)}
        self.join_starts = join_starts  # Change 32: also anchors the first position belief at the spawn
        if join_starts:
            s0 = self.bounds[0][0]
            for s, _ in self.bounds[1:]:
                n_in = self.gm.match(self.gm.features(self._frame_img(s0)), self.gm.features(self._frame_img(s)),
                                     FOCAL_GUESS if self.sg_on else None).n
                self.G.add_edge(s0, s, weight=1.0)
                self.trusted.add((s0, s))
                print(f"spawn join: drive starts {s0} and {s} ({n_in} inliers)"
                      + ("" if n_in >= 20 else " -- WARNING: weak match, do the drives start at one spot?"))
        self.shortcuts = self._shortcuts(shortcut_gap, shortcut_k, window, s_shortcut)
        #Change 10: R alone proposes shortcuts, and it confuses look-alike walls (e.g. node 1437,
        # a purple-wall corridor, was joined to the goal area, so the robot thought it was 1 hop
        # from the goal while far away). Like ORB-SLAM2's loop closing: appearance proposes,
        # geometry confirms -- keep a shortcut only if SuperGlue + RANSAC agree. Only with --superglue.
        if self.sg_on:
            #Change 30: run the sequence check on R's candidates BEFORE SuperGlue. Verified pairs are a
            # subset of the candidates, so a candidate that is not on a run of min_run candidates can
            # never end up on a run of min_run verified pairs -- dropping it first changes nothing.
            # Maze 7: SuperGlue on 352 pairs instead of 11,087 (22 s instead of 10+ min), and the
            # final 350 shortcuts are exactly the same set.
            if min_run > 1:
                self.shortcuts = self._sequence_filter(self.shortcuts, min_run)
            self.shortcuts = self._verify_shortcuts(self.shortcuts, geo_min)
        #Change 25b: a pair check alone is not enough -- a repeated wall texture gives 20-60 inliers
        # between two different places. Two drives through the same corridor, though, keep matching
        # frame after frame, while a look-alike wall matches for 1-3 frames (see _sequence_filter).
        if min_run > 1:
            self.shortcuts = self._sequence_filter(self.shortcuts, min_run)
        #Change 15b: check the sign of bearing_from_e on our own data before trusting it
        self.bear_flip = self._calibrate_bearing() if self.sg_on else None
        self.bear_ok = self.bear_flip is not None
        #Change 20 (fix 2B, practical substitute for metric planning): MASt3R-SLAM is not usable
        # here (no poses in the exploration data, and its CUDA extensions need Linux), so instead the
        # planner is told HOW SURE each edge is. Edges the robot actually drove cost 1; a shortcut
        # costs 1 + (40 - inliers)/20 if SuperGlue confirmed it with fewer than 40 inliers (so 20
        # inliers -> 2.0, 40+ -> 1.0). Dijkstra then prefers driven and strongly confirmed routes.
        self.G.add_edges_from(self.shortcuts, weight=1.0)
        sc_inliers = getattr(self, "sc_inliers", {})
        for e in self.shortcuts:  # Change 25b: only the shortcuts that passed the sequence check
            w = 1.0 + max(0, 40 - sc_inliers[e]) / 20.0 if e in sc_inliers else 1.0
            self.G.edges[e]["weight"] = w + SWITCH_COST  # Change 33
        print(f"graph: {self.G.number_of_edges()} edges, {len(self.shortcuts)} verified shortcuts")
        #Change 42 (--map-links): the routes retraced the drives' wandering -- on the dev challenge the
        # planned route alone was ~2,150 ticks against a true shortest of ~760. A map in metres
        # (metric_map.py) links photos that are one place and costs every edge in ticks; on the five
        # local mazes the planned routes got 9-37% shorter (tools/route_check.py).
        self.map_pos = self.map_head = None
        self.map_ticks = False
        self.map_links: list[tuple[int, int]] = []
        self.map_link_set: set[tuple[int, int]] = set()  # Change 43: the links the watchdog looks after
        if (map_links or map_ticks) and self.sg_on:
            self.add_map_links(Path(data_dir), ticks=map_ticks)
        self.stall_steps = STALL_STEPS
        self.banned: list[tuple[int, int, dict]] = []  # Change 25c: shortcuts dropped in this session
        self.penalised: dict[tuple[int, int], float] = {}  # Change 29: original weights of detoured edges
        self.links: list[tuple[int, int]] = []  # Change 35: same-place links added during this session
        #Change 37: the Bayes filter moves along the drives, the verified shortcuts and the spawn join
        self.bayes = None
        if localiser == "bayes":
            jumps = list(self.shortcuts) + [e for e in self.trusted if abs(e[0] - e[1]) != 1]
            self.bayes = BayesLocaliser(self.n, self.bounds, jumps)
        self.bayes_conf = 0.0
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

    @staticmethod
    def _sequence_filter(pairs: list[tuple[int, int]], min_run: int, gap: int = RUN_GAP) -> list[tuple[int, int]]:
        """Change 25b: SeqSLAM-style check. Keep shortcut (i, j) only if it lies on a run of
        >= ``min_run`` shortcuts (i+k, j+k) -- both drives going the same way -- or (i+k, j-k) --
        opposite ways, with at most ``gap`` missing in a row. Against ground truth on mazes 7 and 8,
        a run of >= 8 kept 350 and 485 shortcuts with none false; runs of 1-3 were 88% false."""
        kept = set(pairs)

        def run(i: int, j: int, sgn: int) -> int:
            n = 1
            for step in (1, -1):
                k, miss = step, 0
                while miss <= gap:
                    a, b = i + k, j + sgn * k
                    n, miss = (n + 1, 0) if (min(a, b), max(a, b)) in kept else (n, miss + 1)
                    k += step
            return n

        out = [(i, j) for i, j in pairs if max(run(i, j, +1), run(i, j, -1)) >= min_run]
        print(f"sequence check: kept {len(out)} of {len(pairs)} shortcuts (runs of >= {min_run} pairs)")
        return out

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
        #Change 25c: fewest hops to the goal so far, and the step when that was reached
        self.best_hops, self.best_t = 10**9, 0
        #Change 26: forward ticks driven (in total / when best_hops last improved), measured-turn
        # streak and its direction, and steps left with measured steering switched off
        self.fwd_ticks, self.fwd_at_best = 0, 0
        self.steer_turns, self.steer_last, self.steer_off = 0, None, 0
        #Change 28/31: earliest photo reached while reversing, and undo-steps since then
        self.rev_node, self.rev_n = -1, 0
        #Change 32/33: ticks spent turning on the spot to find a waypoint photo that is not in view, and
        # for how many decisions in a row the waypoint has been out of view
        self.wp_search, self.unseen = 0, 0
        #Change 34: sideways line-up done / goal photo used for it (1 = left, 3 = right, None = not now)
        self.side_done, self.side_target = False, None
        #Change 35: inliers (photo being left, new photo) of the last far-jump check
        self.last_jump = (0, 0)
        #Change 38: L's smoothed probabilities, the committed turn, and the side L last leaned to
        self.l_ema, self.commit, self.l_side = None, None, "LEFT"
        #Change 40: the last decision was a measured-angle one (steer-turn / steer-fwd)
        self.geo_last = False
        #Change 43: the map link being watched, and the decision, hops and stuck count when it began
        self.link_watch, self.link_t, self.link_hops, self.link_stuck = None, 0, 0, 0
        #################################################


    def setup(self, info: SessionInfo) -> None:
        self._reset()
        self.same_n = 0  # Change 44
        self.ci_ok, self.appr_pause = 0, 0  # Change 46
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
            for i in np.argsort(-pk)[:GOAL_TOPK if self.goal_place else 6]:  # Change 45
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
    
        #Change 25c: a new session starts from the full map again
        for a, b, w in self.banned:
            self.G.add_edge(a, b, **w)
        for e, w in self.penalised.items():  # Change 29
            self.G.edges[e]["weight"] = w
        for e in self.links:  # Change 35
            if self.G.has_edge(*e):
                self.G.remove_edge(*e)
        self.banned, self.penalised, self.links, self._local = [], {}, [], {}
        if self.bayes is not None:  # Change 37: every run starts at the spawn = the drives' first photos
            self.bayes.reset([i for s, e in self.bounds for i in range(s, min(s + SPAWN_FRAMES, e))])
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
        #Change 26: the challenge allows 60,000 ticks by default, so pacing by the whole limit would
        # keep the strict check-in stage for ~18,000 ticks. Pace over the first PACE_TICKS instead.
        self.frac = obs.step / max(min(total, PACE_TICKS), 1)
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
            #Change 32: back off AWAY from what blocked us (all stuck events in c31/c31_anyloc came while
            # reversing into a wall, and this used to back up even further), and keep the position
            # belief: it used to be erased here, and the unchecked relocalisation that followed put the
            # robot 4.4 m wrong (c31_anyloc step 1137). Stop reversing here too: it just hit a wall.
            escape = Action.FORWARD if self.prev_act == "BACKWARD" else Action.BACKWARD
            self.rev_n = REV_MAX
            return self._out((escape, 4), "stuck")

        #Change 9: while exploring, skip SPTM localisation/planning and drive from the image
        if self.explore > 0:
            return self._explore(obs)

        q = np.asarray(self.scorer.embed([obs.image]), np.float32)[0]
        p = self._prob_to_all(q)
        #Change 37: --localiser bayes keeps a probability for every map photo (bayes_loc.py) instead of
        # one best match patched with jump rules; --localiser argmax (default) is the method below.
        node = self._bayes_node(obs, p) if self.bayes is not None else self._best_match_node(obs, q, p)
        self.same_n = self.same_n + 1 if node == self.node else 0  # Change 44: decisions at this photo
        self.node = node
        #Change 44: confused (the same photo CONFUSED_STEPS decisions in a row)? think before moving
        if (self.think_confused and self.sg_on and self.same_n >= CONFUSED_STEPS
                and len(self.paths[node]) - 1 > self.near_hops + THINK_AHEAD):
            better = self._think(obs, node)
            if better != node:
                node = self.node = better
                self.same_n = 0

 ##################################################################      
        path = self.paths[node][::-1]  # node ... goal
        self.hops = len(path) - 1
        if self.map_link_set and self._watch_links(path):  # Change 43: a link that led nowhere was dropped
            path = self.paths[node][::-1]
            self.hops = len(path) - 1

        #Change 25c: safety net for a shortcut that slipped through the checks. If the robot has not
        # got any closer to the goal (in hops) for STALL_STEPS steps while still far from it, drop
        # the first shortcut on its route and plan again. Driven edges and the spawn join are never
        # dropped, so the goal always stays reachable -- in the worst case by retracing a drive.
        #Change 26: ... and only while the robot is actually driving: in the 2026-10-26 maze 7 run it
        # spun on the spot, and the watchdog dropped 4 real shortcuts near the start for nothing.
        if self.hops < self.best_hops:
            self.best_hops, self.best_t, self.fwd_at_best = self.hops, self.t, self.fwd_ticks
        #Change 29: and if there is no shortcut to blame (or the robot hardly drove), do not keep
        # repeating the same stretch: make the next part of the route costlier and re-plan, so the
        # robot tries another way where one exists (sptm7_c27 looped at nodes 654-660 for ~1000 steps).
        elif self.t - self.best_t > self.stall_steps and self.hops > self.near_hops and self.lock == 0:
            moved = self.fwd_ticks - self.fwd_at_best >= STALL_FWD
            if not (moved and self._ban_shortcut(path)):
                self._detour(path)
            path = self.paths[node][::-1]
            self.hops = len(path) - 1
            self.best_hops, self.best_t, self.fwd_at_best = self.hops, self.t, self.fwd_ticks

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
        if self.lock == 0 and self.hops > self.near_hops:  # Change 34: left the goal area: start over there
            self.side_done, self.side_target = False, None
        if (self.goal_watch and self.sg_on and self.lock == 0 and self.hops > self.near_hops
                and self.t % GOAL_WATCH_EVERY == 0):  # Change 39
            live = self.gm.features(obs.image)
            seen = [self.gm.match(tf, live, self.focal).n for tf in self.tfeat]
            if max(seen) >= GOAL_SEEN:
                self.lock = 12  # the approach below takes over
                print(f"[{self.t}] goal photo {int(np.argmax(seen))} seen ({max(seen)} inliers) "
                      f"at {self.hops} hops on the map: approaching")
        if self.hops > 3 and self.scan >= 105:  # Change 8c: was `if self.hops > 3:`
            self.scan = 0  # left the goal area after a full look-around: allow a fresh one next time
        self.appr_pause = max(0, self.appr_pause - 1)  # Change 46: > 0 only after a refused weak check-in
        if ((self.hops <= self.near_hops or self.lock > 0 or 0 < self.scan < 105)  # Change 8c: + scan in progress
                and not (self.appr_pause and self.hops >= 1)):
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
        #Change 28: the route runs BACKWARDS along an exploration drive -- the next node is the photo the
        # drive took just before this one. L was only trained on waypoints AHEAD (sptm_train.sample_l:
        # b = a + 1..LOOKAHEAD) and has no BACKWARD output, so there it turned on the spot while the
        # measured-angle steering turned it back (sptm7_c27: 3171 of 4929 steps at nodes 640-660, the
        # true position never changing). Teach-and-repeat in reverse instead: undo the move the drive
        # made there (FORWARD -> BACKWARD, LEFT -> RIGHT), so the camera keeps facing the way the drive's
        # photos were taken and localisation keeps matching them. Re-decided every step.
        #Change 31: count progress by the EARLIEST photo reached, not by "same node as last step". In
        # sptm7_c29 the belief flipped 1083 <-> 1085 every step, so the per-node counter reset every
        # step and the robot alternated LEFT 2 / RIGHT 2 for ~2450 steps without moving at all.
        #Change 32: when reversing stops making progress, re-plan around that stretch (Change 29's detour)
        # instead of handing over to L for 20 steps: L cannot drive backwards, so it drove FORWARD and
        # undid the reversing (c31: BACKWARD x12, then FORWARD x12, again and again at nodes 603/620).
        nxt = path[1] if len(path) > 1 else -1
        if nxt == node - 1 and self.traj[nxt] == self.traj[node]:
            if self.rev_node < 0 or self.traj[node] != self.traj[self.rev_node] or node < self.rev_node:
                self.rev_node, self.rev_n = node, 0  # reached an earlier photo: real progress
            else:
                self.rev_n += 1
            if self.rev_n >= REV_MAX:
                self._detour(path)
                path = self.paths[node][::-1]
                self.hops = len(path) - 1
                self.rev_node, self.rev_n = -1, 0
                nxt = path[1] if len(path) > 1 else -1
            if nxt == node - 1 and self.traj[nxt] == self.traj[node]:  # still the way to go: keep reversing
                undo = REVERSE[self.frames[nxt].action]
                self.state = f"reverse {node}->{nxt}: undo {self.frames[nxt].action}"
                return self._out((Action[undo], 2), "reverse")
        else:
            self.rev_node, self.rev_n = -1, 0
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
        if self.steer_off > 0:  # Change 26: measured steering is resting after a spin; L steers
            self.steer_off -= 1
        elif self.sg_on and self.steer_on:
            g = self._geo_steer(obs, wp, l)
            if g is not None:
                return g

        #Change 32: the waypoint photo is not in view (and SuperGlue did not find it above): L would steer
        # blind -- c31_anyloc: 2410 of 3589 L decisions had the waypoint at p < 0.1, and L was only ever
        # trained on waypoints AHEAD ON THE SAME DRIVE (sptm_train.sample_l), not across shortcuts. Turn on
        # the spot to find it first (each step re-checks p and SuperGlue); only after a full circle
        # without finding it does L get to try.
        #Change 33: ... but only after the waypoint has been out of view SEARCH_AFTER decisions in a row,
        # and then committed until it is seen again or a full circle is done. In c32_m7 a one-frame
        # search alternated with L every few steps (search 1, L 8, search 4, L 3, ...): search turned
        # LEFT, L turned back, no progress (plateau at node 65 for ~2000 decisions).
        if p[wp] >= SEEN_P:
            self.unseen, self.wp_search = 0, 0  # in view: a later search may use a full circle again
        else:
            self.unseen += 1
        if self.unseen >= SEARCH_AFTER and self.wp_search < SEARCH_TICKS:
            # (not reset when only the waypoint changes: the belief flickers while turning, and that
            # would restart the search every step and spin for ever)
            self.wp_search += 3
            self.state = f"look for wp +{l} ({self.wp_search}/{SEARCH_TICKS} ticks) p={p[wp]:.2f}"
            # Change 38: with --smooth-steer, look the way L leans instead of always LEFT
            look = Action[self.l_side] if self.smooth_steer else Action.LEFT
            return self._out((look, 3), "wp-search")

        #Change 32: the anti-wobble rule (Change 2: after LEFT, RIGHT, LEFT force FORWARD) is removed. It
        # was written for the old L-only steering, drove straight past turns, and Change 26's anti-spin
        # now handles turning back and forth.
        if self.smooth_steer:  # Change 38
            name = self._smooth_l(obs, wp)
        else:
            name = self.scorer.locomote(obs.image, self._frame_img(wp), wp)
        self.state = f"wp +{l} p={p[wp]:.2f} best={p.max():.2f}"
        n = self.burst if name in ("FORWARD", "BACKWARD") else 1
        return self._out((Action[name], n), "follow")

    def _smooth_l(self, obs: Observation, wp: int) -> str:
        """Change 38: L's decision, smoothed over time and committed. Left and right about equally
        likely means the waypoint is straight ahead: go forward. A turn must beat forward by
        TURN_MARGIN, and once committed the robot only switches sides on clearly stronger evidence."""
        cur, target = obs.image, self._frame_img(wp)
        if hasattr(self.scorer, "locomote_probs"):
            pr = np.asarray(self.scorer.locomote_probs(cur, target, wp), float)
        else:  # scorers without probabilities (the tests' stand-in): L's choice as one-hot
            pick = self.scorer.locomote(cur, target, wp)
            pr = np.array([pick == a for a in ("FORWARD", "LEFT", "RIGHT")], float)
        self.l_ema = pr if self.l_ema is None else EMA_KEEP * self.l_ema + (1 - EMA_KEEP) * pr
        fwd, left, right = self.l_ema
        self.l_side = "LEFT" if left >= right else "RIGHT"
        if abs(left - right) < TURN_MARGIN:  # no side clearly preferred: the waypoint is ahead
            self.commit = None
            return "FORWARD"
        side = self.l_side
        if self.commit not in (None, side) and abs(left - right) < 2 * TURN_MARGIN:
            side = self.commit  # committed: switch sides only on clearly stronger evidence
        if max(left, right) > fwd + TURN_MARGIN:
            self.commit = side
            return side
        self.commit = None
        return "FORWARD"

    def _detour(self, path: list[int]) -> None:
        """Change 29: make the next DETOUR_EDGES edges of ``path`` (node ... goal) DETOUR_COST times
        costlier and re-plan. Edges are only made costlier, never removed, so the goal stays reachable."""
        for a, b in list(zip(path, path[1:]))[:DETOUR_EDGES]:
            e = (min(a, b), max(a, b))
            if self.detour_once and e in self.penalised:  # Change 41: at most once per edge
                continue
            self.penalised.setdefault(e, self.G.edges[e]["weight"])
            self.G.edges[e]["weight"] *= DETOUR_COST
        self.paths = nx.single_source_dijkstra_path(self.G, self.goal, weight="weight")
        self._local = {}
        new = self.paths[path[0]][::-1]
        print(f"[{self.t}] no progress for {self.stall_steps} steps at hops {len(path) - 1}: detour around "
              f"nodes {path[0]}..{path[min(DETOUR_EDGES, len(path) - 1)]}, next node now {new[1] if len(new) > 1 else new[0]} "
              f"({len(new) - 1} hops)")

    def _think(self, obs: Observation, node: int) -> int:
        """Change 44: SuperGlue inliers of the live view with ``node`` and with the next THINK_AHEAD photos of
        its route; the route photo that clearly matches best (JUMP_MIN, and JUMP_MARGIN more) becomes the node."""
        ahead = self.paths[node][::-1][1:THINK_AHEAD + 1]
        live = self.gm.features(obs.image)
        here = self.gm.match(self._node_feat(node), live, self.focal).n
        n, best = max((self.gm.match(self._node_feat(j), live, self.focal).n, j) for j in ahead)
        if n >= JUMP_MIN and n >= here + JUMP_MARGIN:
            print(f"[{self.t}] thought it over: at route photo {best} ({n} inliers), not {node} ({here})")
            return best
        return node

    def _node_feat(self, i: int):
        """SuperGlue features of map frame ``i``, cached (Change 24's waypoint cache)."""
        if i not in self._wpf:
            if len(self._wpf) > 200:  # keep GPU memory bounded
                self._wpf.clear()
            self._wpf[i] = self.gm.features(self._frame_img(i))
        return self._wpf[i]

    def _bayes_node(self, obs: Observation, p: np.ndarray) -> int:
        """Change 37: one Bayes-filter step (bayes_loc.py). SuperGlue re-ranks the most likely photos;
        among the plausible photos, plan from the one with the shortest route to the goal."""
        sg = None
        if self.sg_on:
            live = self.gm.features(obs.image)
            sg = lambda i: self.gm.match(self._node_feat(i), live, self.focal).n  # noqa: E731
        self.bayes.step(np.where(self.reach, p, 0.0), sg)
        node, self.bayes_conf = self.bayes.choose(lambda i: len(self.paths[i]) if i in self.paths else 10**9)
        if node not in self.paths:  # only unreachable photos plausible (e.g. --no-join): keep the last one
            node = self.node if self.node in self.paths else max(self.paths, key=lambda i: self.bayes.b[i])
        return node

    def _best_match_node(self, obs: Observation, q: np.ndarray, p: np.ndarray) -> int:
        """The original localisation: the best match near the last node, else a far jump that must
        be confirmed (3 frames, margin check, SuperGlue -- Changes 4, 19, 27, 32, 35)."""
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
                #Change 32: every run starts at the spawn, where every exploration drive starts, so the
                # first belief is the best match among the drives' first frames, not the best match
                # anywhere in the maze (c31_anyloc began at node 25, 1.25 m from the robot).
                if self.t == 1 and self.join_starts:
                    spawn = np.array([i for s, e in self.bounds for i in range(s, min(s + SPAWN_FRAMES, e))])
                    node = int(spawn[np.argmax(p[spawn])])
                else:
                    node = g  # no previous estimate (exploration mode forgets it on purpose): accept
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
                #Change 27: R reads ~1.00 for many look-alike places, so three frames in a row is not
                # enough. In the maze 7 run the robot was localised to 0.1 m, 120 hops from the goal,
                # when a jump to node 580 (2.3 m off, 34 inliers) flipped its route for good.
                if node == g and self.sg_on and not self._jump_ok(obs, g):
                    node = self.node
                elif node == g and self.sg_on and self.last_jump[0] >= LINK_OLD:
                    self._link(self.node, g)  # Change 35
        else:
            self.jump_n = 0
        return node

    def add_map_links(self, data_dir: Path, motion: dict | None = None, ticks: bool = False) -> None:
        """Change 42: build the metric map (metric_map.py) and add the same-place links it proposes and
        SuperGlue confirms, each costing what a verified shortcut costs (1 + SWITCH_COST photo hops).
        ``ticks`` (--map-ticks): instead cost EVERY edge in ticks, plus SWITCH_TICKS for a jump to other
        photos. That planned 34% shorter routes on maze 8 but drove 2x longer (5,406 vs 2,618 ticks):
        the cheapest routes ran drives backwards and hopped between drives, and the robot, which steers
        to look like the next photo, lost its waypoint 4x as often. ``motion``: per-label motion for
        the map (tools pass values learned without the maze they check). Runs before connecting."""
        import metric_map as mm

        t0 = time.time()
        drives = mm.load_drives(data_dir)
        meas = [mm.measurements(self.gm, dr["images"], Path("cache") / "odometry") for dr in drives]
        where = mm.frame_index(self.frames, drives)
        closures = [(where[a], where[b]) for a, b in self.shortcuts]
        c0, q0 = meas[0]["comp"][0], meas[0]["conf"][0]
        start = float(c0) if q0 >= mm.COMPASS_MIN and not np.isnan(c0) else 0.0  # fixes the frame only
        xy, heads, _ = mm.build_map(drives, meas, closures, start, self.gm, motion)
        self.map_pos = np.array([xy[d][k] for d, k in where])
        self.map_head = np.array([heads[d][k] for d, k in where])
        links, n_cand, n_checked = mm.propose_links(self.frames, self.map_pos, self.map_head, self.gm,
                                                     self.G.has_edge)
        self.G.add_edges_from(((a, b) for a, b, _ in links), weight=1.0 + SWITCH_COST)
        self.map_ticks = ticks
        if ticks:
            for a, b in self.G.edges:
                self.G.edges[a, b]["weight"] = mm.edge_cost(self.frames, self.map_pos, self.map_head, a, b)
        self.map_links = [(a, b) for a, b, _ in links]
        self.map_link_set = {(min(a, b), max(a, b)) for a, b in self.map_links}
        print(f"map links: {len(links)} added ({n_checked} of {n_cand} candidates checked with SuperGlue)"
              + ("; every edge now costs its ticks" if ticks else "") + f" ({time.time() - t0:.0f} s)")

    def _link(self, a: int, b: int) -> None:
        """Change 35: link map photos ``a`` and ``b``, which the robot has just seen as the same place,
        and re-plan. Costs like any other drive switch; the watchdog can still drop it if it misleads."""
        e = (min(a, b), max(a, b))
        if a == b or self.G.has_edge(*e):
            return
        if self.map_ticks:  # Change 42 --map-ticks: edges cost ticks, so this one does too
            import metric_map as mm

            self.G.add_edge(*e, weight=mm.edge_cost(self.frames, self.map_pos, self.map_head, *e))
        else:
            self.G.add_edge(*e, weight=1.0 + SWITCH_COST)
        self.links.append(e)
        self.paths = nx.single_source_dijkstra_path(self.G, self.goal, weight="weight")
        self.reach[:] = False
        self.reach[list(self.paths)] = True
        self._local = {}
        print(f"[{self.t}] linked {a} <-> {b}: both seen as this place "
              f"({self.last_jump[0]} and {self.last_jump[1]} inliers)")

    def _jump_ok(self, obs: Observation, g: int) -> bool:
        """Change 27: accept a far jump to node ``g`` only if the live view matches g's photo
        clearly (>= JUMP_MIN inliers) and clearly better than the current node's photo."""
        live = self.gm.features(obs.image)
        n_new = self.gm.match(self._node_feat(g), live, self.focal).n
        n_old = self.gm.match(self._node_feat(self.node), live, self.focal).n
        ok = n_new >= JUMP_MIN and n_new >= n_old + JUMP_MARGIN
        self.last_jump = (n_old, n_new)  # Change 35
        if self.verbose:
            print(f"[{self.t}] jump {self.node} -> {g}: inliers old {n_old} new {n_new} -> "
                  + ("accepted" if ok else "rejected"))
        return ok

    def _geo_steer(self, obs: Observation, wp: int, l: int):
        """Change 24: visual teach-and-repeat step toward waypoint frame ``wp`` (``l`` hops ahead).
        Match the live view with the waypoint photo; if they share enough structure, read the
        rotation between them (yaw_from_e, sign verified by tools/compare_yaw.py: + = the robot is
        turned left of the waypoint view, so turn right) and turn by that angle (<= 6 ticks ~ 20 deg
        per step), or drive forward 2 ticks when already lined up. None -> let L decide."""
        m = self.gm.match(self._node_feat(wp), self.gm.features(obs.image), self.focal)
        need = STEER_KEEP if self.steer_hold and self.geo_last else STEER_MIN  # Change 40
        if m.n < need:
            return None
        y = yaw_from_e(m, self.focal)
        if y is None:
            return None
        y = SIGN * y  # same convention as _approach: + = turn right
        self.state = f"wp +{l} steer yaw={y:+.0f} n={m.n}" + (" (hold)" if m.n < STEER_MIN else "")
        if abs(y) > STEER_DEG:
            turn = Action.RIGHT if y > 0 else Action.LEFT
            #Change 26: spinning on the spot -- a turn that undoes the last measured turn, or more
            # than STEER_SPIN turns in a row -- means the angle is noise here: let L steer for a while
            self.steer_turns = self.steer_turns + 1 if self.steer_last in (None, turn) else STEER_SPIN + 1
            self.steer_last = turn
            if self.steer_turns > STEER_SPIN:
                self.steer_turns, self.steer_last, self.steer_off = 0, None, STEER_REST
                return None
            return self._out((turn, int(np.clip(round(abs(y) / TURN_DEG), 1, 6))), "steer-turn")
        self.steer_turns, self.steer_last = 0, None
        return self._out((Action.FORWARD, 2), "steer-fwd")

    def _watch_links(self, path: list[int]) -> bool:
        """Change 43: watch the first map link among the next LINK_LOOK edges of ``path`` (node ... goal);
        drop it after a bump or LINK_PATIENCE decisions without fewer hops, and re-plan. True if dropped."""
        ahead = [(min(a, b), max(a, b)) for a, b in zip(path, path[1:LINK_LOOK + 1])]
        link = next((e for e in ahead if e in self.map_link_set and self.G.has_edge(*e)), None)
        if link != self.link_watch:  # another link (or none) ahead now: start watching it
            self.link_watch, self.link_t, self.link_hops, self.link_stuck = link, self.t, self.hops, self.n_stuck
            return False
        if link is None:
            return False
        bumped = self.n_stuck > self.link_stuck
        if not bumped and (self.t - self.link_t <= LINK_PATIENCE or self.hops < self.link_hops):
            return False
        w = dict(self.G.edges[link])
        self.G.remove_edge(*link)
        paths = nx.single_source_dijkstra_path(self.G, self.goal, weight="weight")
        if path[0] not in paths:  # it was the only way on: keep it, and give it another LINK_PATIENCE
            self.G.add_edge(*link, **w)
            self.link_t, self.link_stuck = self.t, self.n_stuck
            return False
        self.banned.append((*link, w))
        self.paths, self._local = paths, {}
        self.reach[:] = False
        self.reach[list(paths)] = True
        self.link_watch = None
        why = "bumped into something" if bumped else f"no fewer hops in {LINK_PATIENCE} steps"
        print(f"[{self.t}] map link {link[0]}-{link[1]} dropped ({why}), now {len(paths[path[0]]) - 1} hops")
        return True

    def _ban_shortcut(self, path: list[int]) -> bool:
        """Change 25c: remove the first shortcut on ``path`` (node ... goal) whose removal still
        leaves the robot's node connected to the goal, and re-plan. False if there is none."""
        for a, b in zip(path, path[1:]):
            e = (min(a, b), max(a, b))
            if e in self.trusted:
                continue
            w = dict(self.G.edges[e])
            self.G.remove_edge(*e)
            paths = nx.single_source_dijkstra_path(self.G, self.goal, weight="weight")
            if path[0] not in paths:  # it was the only way on: keep it
                self.G.add_edge(*e, **w)
                continue
            self.banned.append((*e, w))
            self.paths, self._local = paths, {}
            self.reach[:] = False
            self.reach[list(paths)] = True
            print(f"[{self.t}] no progress for {self.stall_steps} steps at hops {len(path) - 1}: dropped "
                  f"shortcut {e[0]}-{e[1]}, now {len(paths[path[0]]) - 1} hops")
            return True
        return False

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
        if self.side_target is not None:  # Change 34: the sideways axis is lined up now
            self.side_target, self.side_done = None, True
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
        #Change 34: sideways line-up in progress -- use the chosen side photo, nothing else. If it is not
        # in view, try the other side (180 deg turn); if neither is, face front again and check in as before.
        if self.side_target is not None:
            if ms[self.side_target].n < max(12, min_n - TRACK_RELAX):
                if self.side_target == 1:
                    self.side_target = 3
                    self.state = "sideways: left photo not in view, trying the right one"
                    return self._out((Action.RIGHT, 2 * SIDE_TICKS), "approach-sideways")
                self.side_target, self.side_done = None, True
                self.last_k, self.reacq = 0, 0
                self.state = "sideways: no side photo in view, back to the front photo"
                return self._out((Action.LEFT, SIDE_TICKS), "approach-sideways")
            k, m = self.side_target, ms[self.side_target]
        elif ms[0].n >= front_min:  # front photo visible: it alone fixes the goal heading and distance
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
                    if not self.side_done:  # Change 34: line up sideways first (left photo, then right)
                        self.side_target = 1
                        self.state += " -> line up sideways first (left photo)"
                        return self._out((Action.LEFT, SIDE_TICKS), "approach-sideways")
                    self.state += f" checkin-stage>={frac_from:.0%}"
                    return self._checkin(left, m.n)  # Change 46: confirmed first with --confirm-checkin
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
            return self._checkin(left, m.n)  # Change 46: confirmed first with --confirm-checkin
        return self._out((Action.FORWARD, 1), "approach-nudge")

    def _checkin(self, left: float, n: int):
        """Change 46: CHECKIN -- with --confirm-checkin only with CHECKIN_STRONG inliers (else None: not
        here after all, navigation takes over) and after CHECKIN_CONFIRM more passing readings in a row,
        each after a 1-tick turn toward the goal heading (+ left = turn right, as in _approach)."""
        if self.confirm_checkin and n < CHECKIN_STRONG:
            self.state += f" -> too few inliers to check in ({n} < {CHECKIN_STRONG}): back to the route"
            self.appr_pause, self.lock, self.creep, self.search_left = APPROACH_PAUSE, 0, 0, 0
            self.scan = max(self.scan, 105)  # no 360-deg look-around here either
            return None
        if self.confirm_checkin and self.ci_ok < CHECKIN_CONFIRM:
            self.ci_ok += 1
            self.state += f" -> confirm check-in {self.ci_ok}/{CHECKIN_CONFIRM}"
            return self._out((Action.RIGHT if left > 0 else Action.LEFT, 1), "approach-confirm")
        return self._out(Action.CHECKIN, "arrived")

    def _out(self, a, why: str):
        if why != "approach-confirm":  # Change 46: only passing readings IN A ROW count
            self.ci_ok = 0
        act = a[0] if isinstance(a, tuple) else a
        self.prev_act = act.name if act in (Action.FORWARD, Action.BACKWARD) else None
        self.geo_last = why in ("steer-turn", "steer-fwd")  # Change 40
        if act == Action.FORWARD:  # Change 26: for the watchdog's "is it driving at all" check
            self.fwd_ticks += a[1] if isinstance(a, tuple) else 1
        if self.verbose or why in ("arrived", "budget"):
            print(f"[{self.t}] node {self.node} hops {self.hops} {why} {a} {self.state}")
        self.log.append(dict(t=self.t, node=self.node, hops=self.hops, why=why, act=str(a), state=self.state))
        return a

    def hud(self) -> list[str]:
        return [f"SPTM node {self.node} goal {getattr(self, 'goal', -1)} hops {self.hops}", self.state]


def main(agent_cls: type | None = None) -> None:
    """``agent_cls``: the class to build -- my_agent.py passes MyAgent; default SptmAgent."""
    import argparse

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
    # Change 26: back to None. The challenge allows 60,000 ticks unless it says otherwise (SDK
    # DEFAULT_MAX_STEPS), and a slow arrival still earns distance + heading points (up to 6 of 10),
    # while a forced check-in at 4000 away from the goal earns 0. Pass --budget for short local tests.
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--s-local", type=float, default=0.7)
    parser.add_argument("--s-reach", type=float, default=0.8)
    parser.add_argument("--s-shortcut", type=float, default=0.9)
    #######################################################################################################
    #Change 3e: 
    #Change 36: on by default -- nearly every safeguard (shortcut and jump checks, steering, the last
    # metre) needs it, and with only 3 attempts a forgotten flag must not cost one. --no-superglue turns it off.
    parser.add_argument("--superglue", action=argparse.BooleanOptionalAction, default=True,
                        help="SuperPoint+SuperGlue for the goal photos and all geometric checks")
    #Change 10: shortcut check threshold (only used with --superglue)
    parser.add_argument("--geo-min", type=int, default=20, help="min SuperGlue inliers to keep a shortcut")
    #Change 13: the local simulator draws a fresh random motion seed every run, so two runs differ
    # even with identical code. A fixed seed replays the same motion noise -> fair before/after tests.
    parser.add_argument("--motion-seed", type=int, default=None, help="local runs only: replay this motion seed")
    #Change 18: exploration mode is off by default; pass --explore to switch it back on
    parser.add_argument("--explore", action="store_true", help="enable the autonomous exploration mode (Change 9)")
    #Change 19: which place recognition to use: the trained R network, or AnyLoc (DINOv2 + VLAD)
    #Change 36: AnyLoc by default -- belief > 1 m off in 15-19% of steps vs 36% with R on maze 7, 1-5% on maze 8
    parser.add_argument("--place", choices=["r", "anyloc"], default="anyloc", help="place recognition: anyloc | r")
    #Change 25: clean map -- drives joined at the spawn, shortcuts kept only on long verified runs
    parser.add_argument("--min-run", type=int, default=MIN_RUN, help="shortcut needs a run of this many verified pairs (0 = off)")
    parser.add_argument("--no-join", action="store_true", help="do not join the exploration drives at the spawn")
    #Change 26: switch the measured-angle steering (Change 24) off and let L steer alone
    parser.add_argument("--no-steer", action="store_true", help="steer with L only, not by the measured angle")
    #Change 37: localisation method
    parser.add_argument("--localiser", choices=["argmax", "bayes"], default="argmax",
                        help="argmax: best match + jump rules; bayes: Bayes filter over all map photos")
    #Change 38: smoothed, committed steering
    parser.add_argument("--smooth-steer", action="store_true",
                        help="smooth L over time, 'left ~ right' = ahead, commit to a turn direction")
    #Change 39: watch for the goal photos everywhere, not only near the goal node
    parser.add_argument("--goal-watch", action="store_true",
                        help="every 5 decisions match the goal photos; a strong match starts the approach")
    #Change 40/41: commit to decisions instead of re-deciding every step
    parser.add_argument("--steer-hold", action="store_true",
                        help="measured steering, once in control, keeps it down to STEER_KEEP inliers")
    parser.add_argument("--detour-once", action="store_true",
                        help="a stall makes each route edge costlier at most once (no stacking)")
    #Change 42: plan on a map in metres
    parser.add_argument("--map-links", action="store_true",
                        help="build a metric map before connecting and add its verified same-place links")
    parser.add_argument("--confirm-checkin", action="store_true",
                        help="check in only with 35+ inliers and on 3 fresh readings in a row (1-tick turns between)")
    parser.add_argument("--goal-place", action="store_true",
                        help="choose the goal photo among the top 64 candidates per goal photo (not 6)")
    parser.add_argument("--think-confused", action="store_true",
                        help="when the believed photo stays the same, check the next route photos with SuperGlue")
    parser.add_argument("--map-ticks", action="store_true",
                        help="experimental: like --map-links, and cost every edge in ticks (drove 2x longer on maze 8)")
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
    agent = (agent_cls or SptmAgent)(
        data, scorer, burst=args.burst, budget=args.budget, s_local=args.s_local,
        s_reach=args.s_reach, s_shortcut=args.s_shortcut, log_dir=args.log_dir, verbose=args.verbose,
        goal_matcher=goal_matcher, goal_min=20 if args.superglue else 12,
        geo_min=args.geo_min,  # Change 10
        explore=args.explore,  # Change 18
        join_starts=not args.no_join, min_run=args.min_run,  # Change 25
        steer=not args.no_steer,  # Change 26
        localiser=args.localiser,  # Change 37
        smooth_steer=args.smooth_steer,  # Change 38
        goal_watch=args.goal_watch,  # Change 39
        steer_hold=args.steer_hold, detour_once=args.detour_once,  # Change 40/41
        map_links=args.map_links, map_ticks=args.map_ticks,  # Change 42
        think_confused=args.think_confused,  # Change 44
        goal_place=args.goal_place,  # Change 45
        confirm_checkin=args.confirm_checkin,  # Change 46
    )
##################################################################################################################
    #Change 13: pass the motion seed to the local simulator (ignored for real challenge runs)
    local_opts = {"motion_seed": args.motion_seed} if args.local is not None and args.motion_seed is not None else {}
    run(agent, cli.challenge(args), **cli.run_options(args), **local_opts)


if __name__ == "__main__":
    main()

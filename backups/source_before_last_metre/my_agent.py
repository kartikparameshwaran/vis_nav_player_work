"""
Autonomous agent: localise with a Bayes filter over the exploration frames, follow the
shortest path on the topological map, then servo on the goal photos and check in.

    uv run source/my_agent.py --local 7                       # free, on your machine
    uv run source/my_agent.py --local 7 --descriptor dino     # DINOv2 descriptors
    uv run source/my_agent.py --challenge <id>                # one real attempt (spends it)

Pipeline per step (see PLAN.md):
    frame -> descriptor -> belief update -> estimated node -> next move on the shortest path
    near the goal (few hops) -> match the frame to the four goal photos with RANSAC:
        heading = sideways shift of the scene, distance = growth of the scene;
        turn / creep until both match the front photo, then CHECKIN.

Everything it uses is on the list in the original skeleton: exploration data, info.targets,
info.camera, info.limits, obs.image, obs.step, obs.steps_left.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from features import make as make_descriptor
from geometry import Matcher, scale_ratio, shift_deg
from navigator import BeliefFilter, TopoMap
from vis_nav_sdk import Action, Agent, Observation, Result, SessionInfo, run
from vlad import load_frames

TURN_DEG = 3.44
VIEW_OFFSET = (0.0, 90.0, 180.0, -90.0)  # front, left, back, right: how far "left" of front


def _intrinsic_focal(info: SessionInfo) -> float:
    cam = info.camera
    for name in ("K", "intrinsics", "matrix", "camera_matrix", "k"):
        k = getattr(cam, name, None)
        if k is not None and np.asarray(k).shape == (3, 3) and np.asarray(k)[0, 0] > 1:
            return float(np.asarray(k)[0, 0])
    for v in vars(cam).values():
        a = np.asarray(v) if isinstance(v, (list, tuple, np.ndarray)) else None
        if a is not None and a.shape == (3, 3) and a[0, 0] > 1:
            return float(a[0, 0])
    return 92.0  # the slides: f = 92 px


class MyAgent(Agent):
    def __init__(
        self,
        data_dir: Path,
        *,
        descriptor: str = "vlad",
        subsample: int = 1,
        n_clusters: int = 128,
        cache_dir: Path | None = None,
        head: Path | None = None,
        burst: int = 1,
        lookahead: int = 4,
        steer_deg: float = 5.0,
        scale_tol: float = 0.04,
        near_hops: int = 6,
        tau: float = 1.0,
        log_dir: Path | None = None,
        verbose: bool = False,
        budget: int | None = None,
    ) -> None:
        cache = cache_dir or Path("cache") / Path(data_dir).name
        self.frames, bounds = load_frames(Path(data_dir), subsample)
        print(f"{len(self.frames)} frames from {len(bounds)} trajectories")
        self.desc = make_descriptor(descriptor, cache, n_clusters, head)
        db = self.desc.database(self.frames)
        self.matcher = Matcher()
        self.topo = TopoMap(
            self.frames, bounds, db, cache=cache / f"shortcuts_{descriptor}_{subsample}.pkl",
            matcher=self.matcher,
        )
        self.belief = BeliefFilter(self.topo, tau=tau)
        self.burst, self.near_hops = burst, near_hops
        self.lookahead, self.steer_deg, self._feat_cache = lookahead, steer_deg, {}
        self.scale_tol = scale_tol
        self.log_dir, self.verbose, self.budget = log_dir, verbose, budget
        self.sign = 1  # +1: a LEFT turn moves scene content to the right in the image
        self._reset()

    def _reset(self) -> None:
        self.phase, self.log, self.t = "calib0", [], 0
        self.prev_small, self.prev_act, self.stuck, self.queue, self.n_stuck = None, None, 0, [], 0
        self.appr_steps, self.node, self.hops, self.state = 0, -1, -1, "init"
        self.best_cost, self.best_t, self.unstick, self.last_follow = np.inf, 0, 0, None
        self.jump_waits: dict = {}
        self.belief.reset()

    # -- session

    def setup(self, info: SessionInfo) -> None:
        # Runs twice (once as a pre-flight with random frames): derive everything from `info`.
        self._reset()
        self.focal = _intrinsic_focal(info)
        self.max_steps = info.limits.max_steps
        self.targets = info.targets
        self.tfeat = [self.matcher.features(t) for t in self.targets]
        # The goal photos are views from the goal in four directions; the exploration drive may
        # have passed the goal facing any way, so look for all four, keep the best geometric match.
        best = (-1, None, None)  # (inliers, node, view)
        for k in range(4):
            sims = self.topo.db @ self.desc.extract(self.targets[k])
            for i in np.argsort(-sims)[:6]:
                n = self.matcher.match(self.tfeat[k], self._node_feat(int(i))).n
                if n > best[0]:
                    best = (n, int(i), k)
        if best[0] >= 12:
            goal = best[1]
        else:  # weak evidence: fall back to the front view's top retrieval hit
            goal = int(np.argmax(self.topo.db @ self.desc.extract(self.targets[0])))
        self.topo.plan_to(goal)
        self.goal = goal
        print(f"goal: node {goal} (best match: view {best[2]}, {best[0]} inliers)")

    def finish(self, result: Result | None) -> None:
        print("result:", result)
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            out = self.log_dir / f"run_{int(time.time())}.json"
            out.write_text(json.dumps({"log": self.log, "result": repr(result)}, default=float))

    # -- decisions

    def act(self, obs: Observation) -> Action | tuple[Action, int]:
        self.t += 1
        left = obs.steps_left if self.budget is None else min(obs.steps_left, self.budget - obs.step)
        if left <= 12:
            return self._out(Action.CHECKIN, "budget")
        small = cv2.resize(cv2.cvtColor(obs.image, cv2.COLOR_BGR2GRAY), (40, 30)).astype(np.float32)

        if self.phase == "calib0":  # learn which way "LEFT" shifts the picture
            self.calib_img, self.phase = obs.image, "calib1"
            return self._out((Action.LEFT, 6), "calib")
        if self.phase == "calib1":
            m = self.matcher.match(self.matcher.features(self.calib_img), self.matcher.features(obs.image))
            s = shift_deg(m, self.focal)
            if m.n >= 8 and abs(s) > 3:
                self.sign = 1 if s > 0 else -1
            print(f"calibration: shift {s:+.1f} deg after LEFT x6 ({m.n} matches) -> sign {self.sign:+d}")
            self.phase = "calib2"
            return self._out((Action.RIGHT, 6), "calib")
        self.phase = "nav"

        stuck_now = (
            self.prev_act in ("FORWARD", "BACKWARD")
            and self.prev_small is not None
            and float(np.abs(small - self.prev_small).mean()) < 1.0
        )
        if self.prev_act in ("FORWARD", "BACKWARD"):
            # a blocked burst counts; a burst that moved clears it. Turns (steering, recovery)
            # leave the count alone: resetting on them hid a wall the robot kept pushing into.
            self.stuck = self.stuck + 1 if stuck_now else 0
        self.prev_small = small

        q = self.desc.extract(obs.image)
        self.node = self.belief.update(q)
        self.hops = self.topo.hops(self.node)

        if self.queue:  # recovery: back off the wall, then turn well away from it
            return self._out(self.queue.pop(0), "recover")
        if self.stuck >= 2:
            self.stuck, self.n_stuck = 0, self.n_stuck + 1
            turn = Action.RIGHT if self.n_stuck % 2 else Action.LEFT  # alternate sides
            self.queue = [(turn, min(8 + 3 * (self.n_stuck % 4), 20))]
            self.belief.reset()  # the pose estimate got us here; do not trust it
            return self._out((Action.BACKWARD, 4), "stuck")

        if self.hops <= self.near_hops:
            r = self._approach(obs)
            if r is not None:
                return r
        self.appr_steps = 0
        self._check_jump()
        cost = float(self.topo.dist[self.node])
        if cost < self.best_cost - 2:
            self.best_cost, self.best_t = cost, self.t
        if self.unstick > 0:  # watchdog: no progress for a long while -> drive on, then re-plan
            self.unstick -= 1
            return self._out((Action.FORWARD, 2), "unstick")
        if self.t - self.best_t > 60:
            self.best_t, self.unstick = self.t, 24
            self.belief.reset()
        name = self.topo.action_toward_goal(self.node, prefer=self.last_follow)
        self.last_follow = name
        if name in ("FORWARD", "BACKWARD"):
            steer = self._steer(obs, name)
            if steer is not None:
                return steer
        n = self.burst if name in ("FORWARD", "BACKWARD") else 2
        return self._out((Action[name], n), "follow")

    def _check_jump(self) -> None:
        """A shortcut we keep standing in front of without ever ending up on its far side was
        a false match: ban it and re-plan. (Verified shortcuts rarely are, but one is enough
        to trap the robot in a loop.)"""
        jump = self.topo.first_jump(self.node)
        if jump is None:
            self.jump_waits.clear()
            return
        self.jump_waits[jump] = self.jump_waits.get(jump, 0) + 1
        if self.jump_waits[jump] >= 15 and self.belief.mass_near(jump[1], 3, self.topo) < 0.25:
            print(f"[{self.t}] shortcut {jump} did not take: banned, re-planning")
            self.topo.ban(*jump)
            self.topo.plan_to(self.goal)
            self.jump_waits.clear()

    def _node_feat(self, i: int):
        if i not in self._feat_cache:
            self._feat_cache[i] = self.matcher.features(cv2.imread(str(self.frames[i].path)))
        return self._feat_cache[i]

    def _steer(self, obs: Observation, name: str):
        """On a straight stretch, match the camera to the frame ``lookahead`` hops ahead on the
        path and correct the heading when the scene has shifted sideways. This is what keeps
        the robot in the corridor despite motion noise (replaying recorded actions open-loop
        drifts into the walls). Turns are not pre-empted: only same-action stretches count."""
        tgt, cur = self.node, self.node
        for _ in range(self.lookahead):
            nxt = int(self.topo.next_hop[cur])
            if nxt < 0 or self.topo.edge_action(cur, nxt) != name:
                break
            tgt, cur = nxt, nxt
        if tgt == self.node:
            return None
        m = self.matcher.match(self._node_feat(tgt), self.matcher.features(obs.image))
        if m.n < 10:
            return None
        left = self.sign * shift_deg(m, self.focal)  # how far left of the reference we point
        self.state = f"steer n={m.n} left={left:+.1f}"
        if abs(left) > self.steer_deg:
            turn = Action.RIGHT if left > 0 else Action.LEFT
            return self._out((turn, int(np.clip(round(abs(left) / TURN_DEG), 1, 3))), "steer")
        return None

    def _approach(self, obs: Observation):
        feat = self.matcher.features(obs.image)
        best = None
        for k, tf in enumerate(self.tfeat):
            m = self.matcher.match(tf, feat)
            if best is None or m.n > best[1].n:
                best = (k, m)
        k, m = best
        if m.n < 12:
            return None  # not seeing the goal yet: keep following the map
        self.appr_steps += 1
        s = shift_deg(m, self.focal)
        left = self.sign * s + VIEW_OFFSET[k]  # how far left of the goal's heading we point
        r = scale_ratio(m)
        self.state = f"approach view{k} n={m.n} left={left:+.1f} scale={r:.2f}"
        turn = Action.RIGHT if left > 0 else Action.LEFT
        if k != 0 or abs(left) > 4:
            if k != 0 and abs(left) > 20:
                return self._out((turn, int(np.clip(round(abs(left) / TURN_DEG), 1, 8))), "approach-turn")
            if abs(left) > 4:
                return self._out((turn, int(np.clip(round(abs(left) / TURN_DEG), 1, 3))), "approach-turn")
        tol = self.scale_tol if self.appr_steps <= 25 else 2 * self.scale_tol  # loosen if dithering
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
        return [f"node {self.node} goal {getattr(self, 'goal', -1)} hops {self.hops}", self.state]


def main() -> None:
    import cli

    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data directory")
    parser.add_argument("--descriptor", default="vlad", choices=["vlad", "dino", "both"])
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--head", type=Path, default=None, help="projection from train_head.py")
    parser.add_argument("--burst", type=int, default=1)
    parser.add_argument("--budget", type=int, default=None, help="own step cap, e.g. 2000")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()
    data = cli.exploration_data(args, args.data)
    agent = MyAgent(
        data, descriptor=args.descriptor, subsample=args.subsample, head=args.head, burst=args.burst,
        tau=args.tau, log_dir=args.log_dir, verbose=args.verbose, budget=args.budget,
    )
    run(agent, cli.challenge(args), **cli.run_options(args))


if __name__ == "__main__":
    main()

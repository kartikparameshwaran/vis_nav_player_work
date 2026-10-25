"""A tiny textured-maze simulator, only for testing agent logic without the course simulator.

Same interface as the real task: 320x240 BGR frames, f = 92 px, FORWARD 0.03 m / LEFT 3.44 deg
per tick with motion noise, walls stop you, four goal photos (front, left, back, right),
exploration drives recorded as ``traj_<i>/data_info.json`` + jpgs every 5 ticks, and the
course's default rubric. It is NOT the real renderer: absolute scores here mean nothing,
only that the plumbing (localise -> plan -> approach -> check in) works.
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from vis_nav_sdk import Action, Camera, Limits, Observation, SessionInfo

W, H, F, CX, CY = 320, 240, 92.0, 160.0, 120.0
FWD_M = 0.03
TURN = math.radians(3.44)
CAM_H, WALL_H = 0.5, 1.0
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # heading 0, 90, 180, 270 deg (CCW)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class MiniWorld:
    def __init__(self, seed: int = 0, n: int = 7, cell: float = 1.0, noise: float = 0.1):
        self.rng = np.random.default_rng(seed)
        self.n, self.cell, self.noise = n, cell, noise
        self.open = self._carve()
        self.segs = self._segments()
        self.x = self.y = self.th = 0.0

    # -- maze
    def _carve(self):
        n, rng = self.n, self.rng
        op = {(i, j): set() for i in range(n) for j in range(n)}
        stack, seen = [(0, 0)], {(0, 0)}
        while stack:
            c = stack[-1]
            nb = [
                (c[0] + dx, c[1] + dy)
                for dx, dy in DIRS
                if (c[0] + dx, c[1] + dy) in op and (c[0] + dx, c[1] + dy) not in seen
            ]
            if not nb:
                stack.pop()
                continue
            o = nb[int(rng.integers(len(nb)))]
            op[c].add(o)
            op[o].add(c)
            seen.add(o)
            stack.append(o)
        for _ in range(n):  # a few loops, so shortcuts matter
            c = (int(rng.integers(n)), int(rng.integers(n)))
            d = DIRS[int(rng.integers(4))]
            o = (c[0] + d[0], c[1] + d[1])
            if o in op:
                op[c].add(o)
                op[o].add(c)
        return op

    def _segments(self):
        n, s = self.n, self.cell
        segs = []
        for i in range(n):
            for j in range(n):
                if (i + 1, j) not in self.open[(i, j)]:
                    segs.append((s * (i + 1), s * j, s * (i + 1), s * (j + 1)))
                if (i, j + 1) not in self.open[(i, j)]:
                    segs.append((s * i, s * (j + 1), s * (i + 1), s * (j + 1)))
        for k in range(n):
            segs += [(0, s * k, 0, s * (k + 1)), (s * k, 0, s * (k + 1), 0)]
        return np.array(segs, float)

    def center(self, c):
        return ((c[0] + 0.5) * self.cell, (c[1] + 0.5) * self.cell)

    def cell_of(self, x, y):
        return (int(x // self.cell), int(y // self.cell))

    # -- rendering
    def render(self, pose=None) -> np.ndarray:
        x, y, th = pose or (self.x, self.y, self.th)
        u = np.arange(W) + 0.5
        phi = np.arctan((u - CX) / F)
        ang = th - phi
        d = np.stack([np.cos(ang), np.sin(ang)], 1)
        a = self.segs[:, :2]
        e = self.segs[:, 2:] - a
        ap = a - np.array([x, y])
        denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
        cae = ap[:, 0] * e[:, 1] - ap[:, 1] * e[:, 0]
        cad = ap[None, :, 0] * d[:, None, 1] - ap[None, :, 1] * d[:, None, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cae[None] / denom
            s = cad / denom
        ok = (np.abs(denom) > 1e-9) & (t > 1e-3) & (s >= 0) & (s <= 1)
        t = np.where(ok, t, np.inf)
        k = np.argmin(t, 1)
        tt = t[np.arange(W), k]
        ss = s[np.arange(W), k]
        seglen = np.hypot(e[k, 0], e[k, 1])
        perp = np.maximum(tt * np.cos(phi), 0.05)
        half = F * CAM_H / perp
        top, bot = CY - half, CY + half
        rows = np.arange(H, dtype=float)[:, None]
        img = np.empty((H, W, 3), np.uint8)
        HR = int(CY)
        img[:HR] = 245
        fl = np.clip((rows[HR:] - CY) / (H - CY), 0, 1)
        floor = np.stack([232 - 10 * fl, 232 - 40 * fl, 232 - 60 * fl], -1)  # BGR: bluish
        img[HR:] = np.broadcast_to(floor, (H - HR, W, 3)).astype(np.uint8)
        inwall = (rows >= top[None]) & (rows < bot[None])
        v = np.clip((rows - top[None]) / np.maximum(bot - top, 1)[None], 0, 0.999)
        by = (v * 6).astype(np.int64)
        bx = (ss * seglen / 0.13).astype(np.int64)[None]
        sid = k.astype(np.int64)[None]
        h = (sid * 73856093) ^ (bx * 19349663) ^ (by * 83492791)
        h = (h * 2654435761) & 0xFFFFFFFF
        shade = np.clip(1.3 / (0.6 + 0.35 * perp), 0.3, 1.0)[None]
        for ch, sh in enumerate((0, 8, 16)):
            col = (((h >> sh) & 255) * shade).astype(np.uint8)
            img[..., ch] = np.where(inwall, col, img[..., ch])
        return img

    # -- dynamics
    def _dist_to_walls(self, x, y):
        a, b = self.segs[:, :2], self.segs[:, 2:]
        ab = b - a
        t = np.clip(((np.array([x, y]) - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
        p = a + ab * t[:, None]
        return np.hypot(p[:, 0] - x, p[:, 1] - y).min()

    def step(self, fwd: int, turn: int, n: int = 1) -> None:
        for _ in range(n):
            if turn:
                self.th = wrap(self.th + turn * TURN * (1 + self.noise * self.rng.normal()))
            if fwd:
                m = fwd * FWD_M * (1 + self.noise * self.rng.normal())
                nx, ny = self.x + m * math.cos(self.th), self.y + m * math.sin(self.th)
                if self._dist_to_walls(nx, ny) > 0.12:
                    self.x, self.y = nx, ny

    # -- exploration drives
    def _walk(self, length: int, start):
        path, cur, prev = [start], start, None
        for _ in range(length):
            nb = [c for c in self.open[cur] if c != prev] or list(self.open[cur])
            nxt = nb[int(self.rng.integers(len(nb)))]
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    def record(self, out: Path, routes: int = 3, length: int = 60, every: int = 5):
        out.mkdir(parents=True, exist_ok=True)
        visited = set()
        for r in range(routes):
            d = out / f"traj_{r}"
            d.mkdir(exist_ok=True)
            start = (int(self.rng.integers(self.n)), int(self.rng.integers(self.n)))
            path = self._walk(length, start)
            self.x, self.y = self.center(start)
            self.th = float(self.rng.integers(4)) * math.pi / 2
            recs = []
            for cell in path[1:]:
                tx, ty = self.center(cell)
                for _ in range(200):
                    want = math.atan2(ty - self.y, tx - self.x)
                    err = wrap(want - self.th)
                    close = math.hypot(tx - self.x, ty - self.y) < 0.12
                    if close:
                        break
                    if abs(err) > math.radians(9):
                        act, (f, t) = ("LEFT", (0, 1)) if err > 0 else ("RIGHT", (0, -1))
                    else:
                        act, (f, t) = "FORWARD", (1, 0)
                    k = len(recs)
                    cv2.imwrite(str(d / f"{k}.jpg"), self.render())
                    recs.append({"step": k, "image": f"{k}.jpg", "action": [act], "pose": [self.x, self.y, self.th]})
                    self.step(f, t, every)
                    visited.add(self.cell_of(self.x, self.y))
            (d / "data_info.json").write_text(json.dumps(recs))
        return sorted(visited)

    # -- episodes
    def oracle_steps(self, start_cell, goal_cell) -> int:
        q, prev = deque([start_cell]), {start_cell: None}
        while q:
            c = q.popleft()
            for o in self.open[c]:
                if o not in prev:
                    prev[o] = c
                    q.append(o)
        cells, c = [], goal_cell
        while c is not None:
            cells.append(c)
            c = prev[c]
        cells.reverse()
        turns = sum(
            1
            for a, b, c2 in zip(cells, cells[1:], cells[2:], strict=False)
            if (b[0] - a[0], b[1] - a[1]) != (c2[0] - b[0], c2[1] - b[1])
        )
        return int((len(cells) - 1) * self.cell / FWD_M + turns * 26 + 1)


def targets(world: MiniWorld, pose):
    x, y, th = pose
    return [world.render((x, y, wrap(th + k * math.pi / 2))) for k in range(4)]


def score(dist, heading, oracle, steps):
    if dist > 0.2 or heading > 20:
        return dict(points=0.0, spl=0.0, gate=False)
    pd = 3 if dist <= 0.10 else 2 if dist <= 0.15 else 1
    ph = 3 if heading <= 10 else 2 if heading <= 15 else 1
    spl = min(1.0, oracle / max(steps, 1))
    ps = 4 if spl >= 0.95 else 3 if spl >= 0.85 else 2 if spl >= 0.75 else 1 if spl >= 0.5 else 0
    return dict(points=float(pd + ph + ps), spl=spl, gate=True)


def make_episode(seed: int, data_dir: Path, n: int = 7, length: int = 60):
    """World + exploration data + (start, goal) poses. Returns (world, start_pose, goal_pose)."""
    world = MiniWorld(seed, n=n)
    visited = world.record(data_dir, length=length)
    rng = np.random.default_rng(seed + 1000)
    goal_cell = visited[int(rng.integers(len(visited)))]
    start_cell = goal_cell
    while start_cell == goal_cell:
        start_cell = visited[int(rng.integers(len(visited)))]
    gth = float(rng.integers(4)) * math.pi / 2
    goal = (*world.center(goal_cell), gth)
    start = (*world.center(start_cell), float(rng.integers(4)) * math.pi / 2)
    cv2.imwrite(str(data_dir / "target.jpg"), world.render(goal))
    return world, start, goal


def run_episode(agent, world: MiniWorld, start, goal, max_steps: int = 4000) -> dict:
    world.x, world.y, world.th = start
    cam = Camera(W, H, "bgr", "jpeg", np.array([[F, 0, CX], [0, F, CY], [0, 0, 1.0]]))
    info = SessionInfo("s", {"id": "mini"}, cam, Limits(max_steps, 8, 0, 3, 120, 300, 3600), targets(world, goal))
    agent.setup(info)
    steps, obs = 0, Observation(world.render(), 0, max_steps)
    oracle = world.oracle_steps(world.cell_of(*start[:2]), world.cell_of(*goal[:2]))
    out = dict(checked_in=False, steps=0, oracle=oracle, points=0.0, spl=0.0, dist=None, heading=None)
    while steps < max_steps:
        r = agent.act(obs)
        a, n = r if isinstance(r, tuple) else (r, 1)
        if a == Action.QUIT:
            break
        if a == Action.CHECKIN:
            dist = math.hypot(world.x - goal[0], world.y - goal[1])
            hd = abs(math.degrees(wrap(world.th - goal[2])))
            out.update(checked_in=True, steps=steps, dist=dist, heading=hd)
            out.update(score(dist, hd, oracle, steps))
            return out
        fwd = int(bool(a & Action.FORWARD)) - int(bool(a & Action.BACKWARD))
        turn = int(bool(a & Action.LEFT)) - int(bool(a & Action.RIGHT))
        world.step(fwd, turn, n)
        steps += n
        obs = Observation(world.render(), steps, max_steps - steps)
    out["steps"] = steps
    return out

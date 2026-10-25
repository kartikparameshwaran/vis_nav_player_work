"""Top view of a finished local run: the maze, the robot's true path, the goal, and how close it got.

    uv run --no-sync tools/top_view.py logs/sptm7_side.txt
    uv run --no-sync tools/top_view.py logs/sptm7_side.txt --maze 7 --motion-seed 2664226132

The agent never knows where the robot really is (the server never says). This script finds out
afterwards by REPLAYING the run: the local simulator is deterministic for a given maze and motion
seed, and a --verbose log has every move the agent sent ("[t] node ... (<Action.LEFT: 4>, 3) ..."),
so sending the same moves again drives the robot along exactly the same path. After every move it
asks the simulator what a check-in there would have been scored on (distance, heading), and puts
that next to what the agent BELIEVED (its map node and hops to the goal node) at the same moment.

Writes logs/<log name>_topview.png: the whole maze (left), a zoom around the goal (right), and
below them the goal photo, the robot's real view when it believed it was nearest the goal, and the
map photo it thought it was at. Only reads the log; changes nothing, spends nothing, and does not
touch the scoring.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import vis_nav_sim as sim  # noqa: E402

MOVE = re.compile(r"^\[(\d+)\] node (-?\d+) hops (\d+) .*?\(<Action\.\w+: (\d+)>, (\d+)\)")
FREE, WALL, ROUTE = (238, 238, 238), (70, 70, 70), (80, 175, 80)
GOAL, NEAR, BELIEF, INK = (40, 40, 220), (200, 40, 200), (200, 150, 0), (20, 20, 20)
GOAL_M, GOAL_DEG = 0.2, 20.0  # the rubric's goal gate (vis_nav_sdk/local.py), only reported here
ZOOM_M = 0.75  # half-size of the zoom box around the goal
NEAR_HOPS = 3  # the agent's near_hops: where it starts its goal approach


def read_log(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):  # Tee-Object in Windows PowerShell writes UTF-16
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("log", type=Path, help="console log of a --verbose --local run (the Tee-Object file)")
ap.add_argument("--maze", type=int, default=None, help="maze seed (default: read from the log, else 7)")
ap.add_argument("--motion-seed", type=int, default=None, help="default: read from the log")
ap.add_argument("--data", type=Path, default=None, help="exploration data (default: data/local-<maze>)")
ap.add_argument("--out", type=Path, default=None, help="default: logs/<log name>_topview.png")
args = ap.parse_args()

# ---- the moves, from the log: (agent step, believed node, believed hops, action, ticks)
text = read_log(args.log)
start = text.find("starting the local simulator")
if start < 0:
    sys.exit("no local run in this log (looked for 'starting the local simulator')")
body = text[start:]  # the '[1] ...' line above this is the SDK's sample-frame check, not a move
moves = [tuple(int(v) for v in m.groups()) for m in map(MOVE.match, body.splitlines()) if m]
if not moves:
    sys.exit("no moves in this log: was the run made with --verbose?")
found = re.search(r"Local maze (\d+)", body)
maze = args.maze if args.maze is not None else int(found[1]) if found else 7
found = re.search(r"motion_seed=(\d+)", body)
motion_seed = args.motion_seed if args.motion_seed is not None else int(found[1]) if found else None
if motion_seed is None:
    sys.exit("no motion seed in the log; pass --motion-seed (the run's 'replay with motion_seed=...')")
found = re.search(r"nav_steps=(\d+)", body)
run_ticks = int(found[1]) if found else None

# ---- replay in the simulator (same noise model as a local session: DEFAULT_NOISE, no bias)
world = sim.Simulator(sim.Textures.find(), maze, motion_noise=sim.DEFAULT_NOISE, motion_seed=motion_seed)
track = []  # (ticks so far, x, y, yaw, distance to goal, heading off goal); track[k] = before move k


def snap() -> None:
    x, y, yaw = world.pose()
    d, h = world.distance_heading()
    track.append((world.steps, x, y, yaw, d, h))


snap()
for t, node, hops, action, n in moves:
    n = max(1, min(n, sim.MAX_REPEAT))  # as the run loop clips it
    world.step_ticks(action, n, [n - 1])
    snap()
ticks, xs, ys, yaws, dist, head = np.array(track, float).T
step = np.array([0] + [m[0] for m in moves])  # agent step after which each pose holds
b_node = np.array([m[1] for m in moves])  # belief when deciding move k, made at pose track[k]
b_hops = np.array([m[2] for m in moves])
if run_ticks is not None and run_ticks != world.steps:
    print(f"WARNING: the replay took {world.steps} ticks but the run took {run_ticks}: the log is "
          "missing moves (run with --verbose?), so the path drawn may not be the real one")
else:
    print(f"replay OK: {len(moves)} moves, {world.steps} ticks" + (" (same as the run)" if run_ticks else ""))

# ---- the maze grid (True = wall) and how metres map onto it
occ = world.occupancy
rows, cols = occ.shape
cs = world.cell_size
sx, sy, _ = world.spawn
gx, gy, gyaw = world.target


def to_cell(x, y, g):
    """Fractional (col, row) of metres (x, y) under mapping g; whole numbers are cell centres."""
    swap, ax, ay, oc, orow = g
    u, v = (y, x) if swap else (x, y)
    return ax * np.asarray(u) / cs + oc, ay * np.asarray(v) / cs + orow


def wall_hits(g) -> int:
    c, r = (np.round(a).astype(int) for a in to_cell(xs, ys, g))
    inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    return int((~inside).sum() + occ[r[inside], c[inside]].sum())


def fit_grid():
    """Solve the metres -> cell mapping from the spawn and the goal (the simulator gives both in
    metres and as cells), and keep the solution under which the replayed path stays out of walls."""
    found = []
    for order in ((0, 1), (1, 0)):  # cells given as (row, col) or (col, row)
        sr, sc = world.spawn_cell[order[0]], world.spawn_cell[order[1]]
        tr, tc = world.target_cell[order[0]], world.target_cell[order[1]]
        for swap in (False, True):  # col along x (as documented) or along y
            for ax in (1, -1):
                for ay in (1, -1):
                    u, v = (sy, sx) if swap else (sx, sy)
                    g = (swap, ax, ay, sc - ax * u / cs, sr - ay * v / cs)
                    gc, gr = to_cell(gx, gy, g)
                    err = abs(float(gc) - tc) + abs(float(gr) - tr)
                    found.append((err > 1.0, wall_hits(g), err, order, g))
    bad, hits, _, order, g = min(found, key=lambda f: f[:3])
    return g, order, hits, bad


G, ORDER, HITS, BAD = fit_grid()
print(f"grid fit: the path touches wall cells at {HITS} of {len(xs)} poses"
      + (" -- WARNING: spawn/goal cells did not fit, the walls may be drawn off" if BAD else ""))
_swap, _ax, _ay = G[:3]
FLIP = (-_ax * _ay if _swap else _ax * _ay) > 0  # row 0 at the bottom iff that keeps x right / y up (unmirrored)


def cell_top(r):
    """Page row (cell units, downwards) of grid row r."""
    return rows - 1 - r if FLIP else r


def disp(x, y):
    """Metres -> drawing units: one unit per cell, right and down on the page."""
    c, r = to_cell(x, y, G)
    return c + 0.5, (rows - r - 0.5) if FLIP else (r + 0.5)


def route() -> list[tuple[float, float]]:
    """The oracle's shortest route through cell centres, if the simulator gives one."""
    try:
        cells = [tuple(int(v) for v in c) for c in world.shortest_path["cells"]]
    except Exception as exc:  # noqa: BLE001 -- only a drawing aid
        print(f"(no shortest route drawn: {exc!r})")
        return []
    return [(c[ORDER[1]] + 0.5, cell_top(c[ORDER[0]]) + 0.5) for c in cells]


ROUTE_UV = route()
cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
COL = [tuple(int(v) for v in c) for c in
       cv2.applyColorMap(np.linspace(0, 255, len(xs)).astype(np.uint8)[:, None], cmap)[:, 0]]

# ---- how close it got, and what the agent believed meanwhile
i_min = int(np.argmin(dist))
facing = np.where(head <= GOAL_DEG, dist, np.inf)
i_face = int(np.argmin(facing))
zone = np.nonzero((dist <= GOAL_M) & (head <= GOAL_DEG))[0]
k_bel = int(np.argmin(b_hops))  # first move made at the agent's fewest hops to the goal node
near = np.nonzero(b_hops <= NEAR_HOPS)[0]
lines = [
    f"maze {maze}, motion seed {motion_seed}: {len(moves)} moves, {world.steps} ticks, replayed in the simulator",
    f"closest to the goal: {dist[i_min]:.2f} m at step {step[i_min]} (tick {int(ticks[i_min])}), "
    f"heading {head[i_min]:.0f} deg off the goal heading",
    (f"closest while facing within {GOAL_DEG:.0f} deg of the goal heading: {dist[i_face]:.2f} m at step "
     f"{step[i_face]}") if np.isfinite(facing[i_face]) else
    f"never faced within {GOAL_DEG:.0f} deg of the goal heading",
    (f"INSIDE the scoring zone (<= {GOAL_M} m and <= {GOAL_DEG:.0f} deg) after steps "
     + ", ".join(str(step[i]) for i in zone[:8]) + (" ..." if len(zone) > 8 else "")
     + ": a check-in there would have scored") if len(zone) else
    f"never inside the scoring zone (<= {GOAL_M} m and <= {GOAL_DEG:.0f} deg at the same moment)",
    f"steps spent within 0.5 m of the goal: {int((dist <= 0.5).sum())}   |   "
    f"at the end: {dist[-1]:.2f} m from the goal, {head[-1]:.0f} deg off its heading",
]
try:
    lines.append(f"shortest route: {world.shortest_path['metres']:.1f} m, {world.oracle_steps} ticks for the oracle; "
                 f"the agent's plan at step {moves[0][0]}: {b_hops[0]} hops to the goal node")
except Exception:  # noqa: BLE001
    pass
lines.append(f"agent BELIEVED it was nearest at step {moves[k_bel][0]}: {b_hops[k_bel]} hops (map node "
             f"{b_node[k_bel]}); it was really {dist[k_bel]:.2f} m from the goal")
if len(near):
    lines.append(f"moves made believing it was within {NEAR_HOPS} hops: {len(near)}; real distance then "
                 f"{dist[near].min():.2f} - {dist[near].max():.2f} m")


# ---- drawing
def view(box, ppu: float):
    """Drawing of box = (left, top, right, bottom) in cell units, ppu pixels per cell."""
    l0, t0, r1, b1 = box
    img = np.full((int(round((b1 - t0) * ppu)), int(round((r1 - l0) * ppu)), 3), FREE, np.uint8)

    def px(u: float, v: float) -> tuple[int, int]:
        return int(round((u - l0) * ppu)), int(round((v - t0) * ppu))

    for r, c in zip(*np.nonzero(occ)):
        cv2.rectangle(img, px(c, cell_top(r)), px(c + 1, cell_top(r) + 1), WALL, -1)
    return img, px


def arrow(img, px, x, y, yaw, length, color, thick):
    cv2.arrowedLine(img, px(*disp(x, y)), px(*disp(x + length * np.cos(yaw), y + length * np.sin(yaw))),
                    color, thick, cv2.LINE_AA, tipLength=0.3)


def label(img, xy, s, color, scale=0.5):
    cv2.putText(img, s, (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, s, (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(box, ppu: float, zoom: bool) -> np.ndarray:
    img, px = view(box, ppu)
    ppm = ppu / cs  # pixels per metre
    if ROUTE_UV:
        cv2.polylines(img, [np.array([px(*p) for p in ROUTE_UV], np.int32)], False, ROUTE,
                      max(2, int(ppm * 0.02)), cv2.LINE_AA)
    pts = [px(*disp(x, y)) for x, y in zip(xs, ys)]
    lw = max(2, int(ppm * 0.012))
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], COL[i], lw, cv2.LINE_AA)
    if zoom:  # one dot per agent step: piles of dots = time spent dithering there
        for i, p in enumerate(pts):
            cv2.circle(img, p, 3, COL[i], -1, cv2.LINE_AA)
    g = px(*disp(gx, gy))
    for rad in (0.1, GOAL_M):  # 0.1 m = full distance points, 0.2 m = edge of the goal
        cv2.circle(img, g, int(rad * ppm), GOAL, 2, cv2.LINE_AA)
    arrow(img, px, gx, gy, gyaw, 0.3 if zoom else 0.35, GOAL, 2)
    label(img, g, "goal", GOAL, 0.6)
    cv2.circle(img, pts[i_min], 9 if zoom else 7, NEAR, 2, cv2.LINE_AA)
    arrow(img, px, xs[i_min], ys[i_min], yaws[i_min], 0.2 if zoom else 0.3, NEAR, 2)
    label(img, pts[i_min], f"closest {dist[i_min]:.2f} m", NEAR)
    if not zoom:
        cv2.drawMarker(img, pts[k_bel], BELIEF, cv2.MARKER_DIAMOND, 16, 2, cv2.LINE_AA)
        label(img, (pts[k_bel][0], pts[k_bel][1] + 26), f"believed {b_hops[k_bel]} hops", BELIEF)
        s = px(*disp(sx, sy))
        cv2.rectangle(img, (s[0] - 6, s[1] - 6), (s[0] + 6, s[1] + 6), INK, -1)
        label(img, s, "start", INK)
        arrow(img, px, xs[-1], ys[-1], yaws[-1], 0.3, INK, 2)
        label(img, pts[-1], "end", INK)
    return img


def titled(img: np.ndarray, title: str, ppu: float, bar_m: float) -> np.ndarray:
    strip = np.full((44, img.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK, 1, cv2.LINE_AA)
    x1 = img.shape[1] - 12  # scale bar, right-aligned
    x0 = x1 - int(bar_m / cs * ppu)
    cv2.line(strip, (x0, 30), (x1, 30), INK, 3)
    cv2.putText(strip, f"{bar_m:g} m", (x0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def tile(img: np.ndarray, top: str, bottom: str) -> np.ndarray:
    im = cv2.resize(img, (320, 240)).copy()
    for k, s in enumerate((top, bottom)):
        label(im, (0, 26 + 22 * k), s, INK, 0.5)
    return im


H = 720  # pixels for the map side
half = ZOOM_M / cs
gu, gv = disp(gx, gy)
left = titled(draw((0, 0, cols, rows), H / rows, False), f"whole maze {maze} ({cols * cs:.1f} m square)",
              H / rows, 1.0)
right = titled(draw((gu - half, gv - half, gu + half, gv + half), H / (2 * half), True),
               "around the goal: rings 0.1 / 0.2 m, one dot per step", H / (2 * half), 0.2)
h = max(left.shape[0], right.shape[0])
pad = lambda im, hh: np.vstack([im, np.full((hh - im.shape[0], im.shape[1], 3), 255, np.uint8)])  # noqa: E731
top = np.hstack([pad(left, h), np.full((h, 12, 3), 255, np.uint8), pad(right, h)])
W = top.shape[1]

# what the agent was looking at when it believed it was nearest the goal
tiles = [tile(world.targets()[0], "goal photo (front)", "where it had to get to"),
         tile(world.render(xs[k_bel], ys[k_bel], yaws[k_bel]), f"real view at step {moves[k_bel][0]}",
              f"{dist[k_bel]:.1f} m from the goal")]
try:
    from vlad import load_frames

    frames, _ = load_frames(args.data or Path(f"data/local-{maze}"))
    tiles.append(tile(cv2.imread(str(frames[b_node[k_bel]].path)), f"map node {b_node[k_bel]}",
                      f"where it thought it was ({b_hops[k_bel]} hops)"))
except Exception as exc:  # noqa: BLE001 -- only a picture
    print(f"(no map photo: {exc!r})")
row = np.hstack(tiles)
row = np.hstack([row, np.full((row.shape[0], W - row.shape[1], 3), 255, np.uint8)])
cap = np.full((40, W, 3), 255, np.uint8)
cv2.putText(cap, "what the robot saw when it believed it was nearest the goal", (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK, 1, cv2.LINE_AA)

legend = ["path colour = time (blue at the start -> red at the end); green = shortest route; "
          "blue diamond = where the agent believed it was nearest",
          "red arrow = the heading to arrive with; magenta = closest approach, arrow = the robot's heading there"]
text_lines = legend + lines
foot = np.full((16 + 26 * len(text_lines), W, 3), 255, np.uint8)
for k, s in enumerate(text_lines):
    cv2.putText(foot, s, (10, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                INK if k >= len(legend) else (90, 90, 90), 1, cv2.LINE_AA)

out = args.out or Path("logs") / f"{args.log.stem}_topview.png"
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), np.vstack([top, cap, row, foot]))
print("\n".join(lines))
print(f"saved {out}")

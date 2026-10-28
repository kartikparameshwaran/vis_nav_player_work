"""
A map in metres from the exploration photos (Change 42, --map-links of sptm_agent.py): every photo gets
a position and a heading, and photos the map puts at one place are linked, so routes stop retracing
the drives' wandering. Checked on the local mazes against the true positions by
tools/odometry_check.py (the map) and tools/route_check.py (the routes).

1. Heading. The maze is a grid, so walls and floor tiles run in only 4 directions. Straight lines next
   to the plain floor or ceiling meet the horizon at vanishing points that give the heading modulo
   90 deg from one photo (lecture 3: "a calibrated camera is a protractor"). The rotation between
   consecutive photos (SuperPoint + SuperGlue: homography and essential matrix) picks the quarter
   turn, chosen for a whole drive at once (Viterbi). Local mazes: heading error <= 1 deg.
2. Positions. With the headings fixed, a LINEAR least-squares problem (lecture 8, graph SLAM): each
   step moves its label's average distance along its heading, drives start at the spawn (drive 0's
   first photo is the origin), and the agent's verified shortcuts say two photos are one place
   (Cauchy-weighted, lecture 5). Local mazes: median error 0.3-0.8 m.
3. Links. Photos within LINK_R of each other on the map, facing within LINK_TURN, that SuperGlue
   confirms with >= LINK_MIN inliers (ORB-SLAM-style loop closing: map AND images must agree), so a
   look-alike place far away in the maze cannot be linked. Every edge then costs what driving it
   takes, in ticks. Local mazes: planned routes 9-37% shorter.

A label only names the move right after a photo; the rest of the ticks to the next photo can be
something else, and some photo gaps hide 45-50 deg turns -- hence the images.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

FOCAL, W, H = 92.0, 320, 240
K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1]])
# Average true motion per photo by label, and the spread of its distance, over the five local mazes
# 7, 8, 21, 37, 52 (cache/truth-*: 15,783 photos): (forward m, turn rad, distance spread m).
# No drive used BACKWARD; it mirrors FORWARD.
LABEL_MOTION = {
    "FORWARD": (0.0944, math.radians(0.25), 0.0567),
    "BACKWARD": (-0.0944, 0.0, 0.0567),
    "LEFT": (0.0139, math.radians(10.94), 0.0277),
    "RIGHT": (0.0134, math.radians(-11.02), 0.0274),
    "IDLE": (0.0642, math.radians(0.33), 0.0652),
    "MIXED": (0.05, 0.0, 0.3),
}
STEP_MAX = math.radians(60)  # most photo steps turn <= ~17 deg (5 ticks), but some photo gaps are
# longer: true turns of 45-50 deg between consecutive photos occur (often labelled FORWARD)
COMPASS_MIN = 0.5  # compass confidence (0..1, how well the lines agree) needed to use it
TURN_SIGMA = math.radians(5)  # rotation error per step, for weighing quarter-turn choices
SPIKE = math.radians(12)  # a compass reading this far from both neighbours is dropped
SIGMA_STEP, SIGMA_SAME = 0.05, 0.15  # m: default error of one step's distance; of "the same place"
SIGMA_GAP = 0.3  # m: a step that turned more than LONG_TURN hides a longer photo gap: distance unknown
LONG_TURN = math.radians(25)
ROBUST = 0.3  # m: a closure still this far apart after a solve counts half (Cauchy)
TICK_M, TICK_RAD = 0.03, math.radians(3.44)  # one tick forward, one tick of turning
LINK_R, LINK_TURN, LINK_MIN = 0.3, math.radians(20), 40
MIN_GAP = 20  # photos apart on one drive before a link between them counts as new
PER_PHOTO = 2  # candidate partners checked per photo (its nearest on the map)
SWITCH_TICKS = 15.0  # extra cost of an edge that jumps to other photos (Change 33's SWITCH_COST in
# ticks): following another drive's photos is where L gets confused, so only switch when it pays


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---- the exploration data, all photos (the agent's map skips IDLE ones)


def load_drives(data: Path) -> list[dict]:
    """Drives of an exploration dataset: labels, photo paths, and true ``poses`` when the folder has
    them (local mazes recorded with truth), else None."""
    drives = []
    for t in sorted(Path(data).glob("traj_*"), key=lambda p: int(p.name[5:])):
        recs = json.loads((t / "data_info.json").read_text())
        truth = t / "poses.json"
        poses = np.array(json.loads(truth.read_text()), float) if truth.exists() else None
        labels = [r["action"][0] if len(r["action"]) == 1 else "MIXED" for r in recs]
        drives.append(dict(labels=labels, poses=poses, images=[t / r["image"] for r in recs]))
    return drives


def frame_index(frames, drives: list[dict]) -> list[tuple[int, int]]:
    """(drive, photo) of each map frame (objects with .trajectory and .path, e.g. the agent's)."""
    index = {(d, p.name): k for d, dr in enumerate(drives) for k, p in enumerate(dr["images"])}
    return [(f.trajectory, index[(f.trajectory, f.path.name)]) for f in frames]


# ---- 1. heading


def _plain(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Plain floor or ceiling at these pixels: white / light grey, or the floor's light blue."""
    h, w = img.shape[:2]
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    out = np.zeros(len(pts), bool)
    bgr = img[pts[inside, 1].astype(int), pts[inside, 0].astype(int)].astype(int)
    b, g, r = bgr[:, 0], bgr[:, 1], bgr[:, 2]
    white = (bgr.min(1) > 200) & (bgr.max(1) - bgr.min(1) < 25)
    blue = (b > 170) & (b - r > 25) & (g > 140) & (r > 110)
    out[inside] = white | blue
    return out


_lsd = cv2.createLineSegmentDetector()


def compass(img: np.ndarray) -> tuple[float, float]:
    """Heading modulo 90 deg (rad, in -45..45 deg) and confidence 0..1 from one photo: line segments
    with plain floor/ceiling on exactly one side, each turned into a direction by its vanishing point
    on the horizon row, then a length-weighted vote modulo 90 deg."""
    segs = _lsd.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0]
    if segs is None:
        return math.nan, 0.0
    s = segs.reshape(-1, 4).astype(float)
    p1, p2 = s[:, :2], s[:, 2:]
    d = p2 - p1
    length = np.hypot(d[:, 0], d[:, 1])
    steep = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))  # 90 = vertical: a wall corner
    keep = (length >= 12) & (steep < 70)
    keep &= ~((np.abs(p1[:, 1] - K[1, 2]) < 6) & (np.abs(p2[:, 1] - K[1, 2]) < 6))  # on the horizon
    mid = (p1 + p2) / 2
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(length, 1e-6)[:, None]
    keep &= _plain(img, mid + 3 * nrm) ^ _plain(img, mid - 3 * nrm)
    if not keep.any():
        return math.nan, 0.0
    line = np.cross(np.c_[p1[keep], np.ones(keep.sum())], np.c_[p2[keep], np.ones(keep.sum())])
    vp = np.cross(line, np.array([0.0, 1.0, -K[1, 2]]))  # where the line meets the horizon
    psi = np.arctan2(vp[:, 0] - K[0, 2] * vp[:, 2], FOCAL * vp[:, 2])
    w = length[keep]
    z = np.sum(w * np.exp(4j * psi))  # x4: the 4 grid directions vote together
    return float(np.angle(z) / 4), float(abs(z) / w.sum())


def image_turns(sg, images: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation between consecutive photos (rad, + = turned left: it matched the true turn on 98-100%
    of clear turns of every local drive): homography, essential matrix, and inliers."""
    from geometry import yaw_from_e

    hom, ess, inl = [], [], []
    prev = sg.features(cv2.imread(str(images[0])))
    for path in images[1:]:
        cur = sg.features(cv2.imread(str(path)))
        m = sg.match(prev, cur, FOCAL)
        inl.append(m.n)
        h_yaw = e_yaw = np.nan
        if m.n >= 8:
            h, _ = cv2.findHomography(m.a, m.b, cv2.RANSAC, 2.0)
            if h is not None:
                r = np.linalg.inv(K) @ h @ K
                u, _, vt = np.linalg.svd(r)  # nearest rotation
                r = u @ vt * np.sign(np.linalg.det(u @ vt))
                h_yaw = math.atan2(r[0, 2], r[2, 2])
            y = yaw_from_e(m, FOCAL)
            e_yaw = math.radians(y) if y is not None else np.nan
        hom.append(h_yaw)
        ess.append(e_yaw)
        prev = cur
    return np.array(hom), np.array(ess), np.array(inl)


def measurements(sg, images: list[Path], cache: Path | None = None) -> dict[str, np.ndarray]:
    """Step rotations between consecutive photos (homography, essential) and every photo's compass;
    without ``sg`` only the compass. Cached in ``cache`` (SuperGlue takes ~1 min per drive)."""
    key = f"{images[0].parent.parent.name}_{images[0].parent.name}_{len(images)}_v1"
    path = Path(cache) / f"{key}.npz" if cache is not None else None
    if sg is not None and path is not None and path.exists():
        return dict(np.load(path))
    print(f"map: measuring {images[0].parent.name} ({len(images)} photos, about a minute per 800; "
          f"cached afterwards) ...", flush=True)
    comp, conf = np.array([compass(cv2.imread(str(p))) for p in images]).T
    nan = np.full(len(images) - 1, np.nan)
    out = dict(hom=nan, ess=nan, inl=np.zeros(len(images) - 1), comp=comp, conf=conf)
    if sg is not None:
        out["hom"], out["ess"], out["inl"] = image_turns(sg, images)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **out)
    return out


def fuse_turns(prior: np.ndarray, hom: np.ndarray, ess: np.ndarray) -> np.ndarray:
    """Per-step rotation: homography and essential agreeing within 2 deg -> their mean, else the one
    closer to the label's average turn; never more than STEP_MAX; the label when there is none."""
    both = ~np.isnan(hom) & ~np.isnan(ess)
    agree = both & (np.abs(hom - ess) < math.radians(2))
    hc = np.where(np.isnan(hom), np.inf, np.abs(hom - prior))
    ec = np.where(np.isnan(ess), np.inf, np.abs(ess - prior))
    turn = np.where(agree, (hom + ess) / 2, np.where(hc <= ec, hom, ess))
    return np.where(np.isnan(turn) | (np.abs(turn) > STEP_MAX), prior, turn)


def compass_headings(start: float, turns: np.ndarray, comp: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Headings along a drive. Each confident compass reading fixes the heading up to a quarter turn;
    the quarter turns of the WHOLE drive are chosen together (Viterbi, lecture 8's HMM view) so they
    agree best with the step rotations -- a step-by-step choice let one bad reading start a runaway.
    Isolated readings that disagree with both neighbours are dropped; between readings, the step
    rotations are followed and the mismatch at the next reading is spread evenly."""
    q = math.pi / 2
    cum = np.concatenate([[0.0], np.cumsum(turns)])  # dead-reckoned rotation since photo 0
    idx = [t for t in range(1, len(comp)) if conf[t] >= COMPASS_MIN and not math.isnan(comp[t])]
    if not idx:
        return start + cum
    cands = [comp[t] + q * np.arange(4) for t in idx]  # the 4 physical headings each reading allows
    prev_t, cost = 0, wrap(cands[0] - start - cum[idx[0]]) ** 2 / (TURN_SIGMA**2 * idx[0])
    prev_c, back = cands[0], []
    for t, c in zip(idx[1:], cands[1:]):
        d = wrap(c[None, :] - prev_c[:, None] - (cum[t] - cum[prev_t]))  # [from, to]
        total = cost[:, None] + d**2 / (TURN_SIGMA**2 * (t - prev_t))
        back.append(np.argmin(total, 0))
        cost, prev_c, prev_t = total.min(0), c, t
    pick = [int(np.argmin(cost))]
    for b in reversed(back):
        pick.append(int(b[pick[-1]]))
    anchors_t, anchors_h = [0] + idx, [start]
    for t_prev, t, c, k in zip(anchors_t, idx, cands, reversed(pick)):  # unwrap into one sequence
        pred = anchors_h[-1] + cum[t] - cum[t_prev]
        anchors_h.append(pred + wrap(c[k] - pred))
    keep = [True] * len(anchors_t)
    for i in range(1, len(anchors_t) - 1):  # drop spikes: off from both neighbours
        a, b, c = anchors_t[i - 1], anchors_t[i], anchors_t[i + 1]
        e1 = anchors_h[i] - anchors_h[i - 1] - (cum[b] - cum[a])
        e2 = anchors_h[i + 1] - anchors_h[i] - (cum[c] - cum[b])
        keep[i] = not (abs(e1) > SPIKE and abs(e2) > SPIKE and e1 * e2 < 0)
    at = [t for t, k in zip(anchors_t, keep) if k]
    ah = [h for h, k in zip(anchors_h, keep) if k]
    out = start + cum.copy()
    for (i, hi), (j, hj) in zip(zip(at, ah), zip(at[1:], ah[1:])):
        miss = hj - hi - (cum[j] - cum[i])
        steps = np.arange(i, j + 1)
        out[i:j + 1] = hi + (cum[steps] - cum[i]) + miss * (steps - i) / (j - i)
    out[at[-1]:] = ah[-1] + cum[at[-1]:] - cum[at[-1]]
    return out


# ---- 2. positions


def solve_positions(heads: list[np.ndarray], dists: list[np.ndarray], closures: list,
                    sigmas: list[np.ndarray] | None = None) -> tuple:
    """Least-squares (x, y) of every photo, headings fixed; ``sigmas``: each step's distance error (m).
    Returns per-drive positions and the final distance between each closure's two photos."""
    from scipy.sparse import coo_matrix, vstack
    from scipy.sparse.linalg import lsqr

    base = np.cumsum([0] + [len(h) for h in heads])
    nvar = 2 * base[-1]
    rows, cols, vals, rhs = [], [], [], []

    def eq(terms, b, w):
        r = len(rhs)
        for c, v in terms:
            rows.append(r)
            cols.append(c)
            vals.append(v * w)
        rhs.append(b * w)

    for d, (h, s) in enumerate(zip(heads, dists)):
        sig = sigmas[d] if sigmas is not None else np.full(len(s), SIGMA_STEP)
        for k, step in enumerate(s):
            i = base[d] + k
            for ax, u in ((0, math.cos(h[k])), (1, math.sin(h[k]))):
                eq([(2 * (i + 1) + ax, 1.0), (2 * i + ax, -1.0)], step * u, 1 / sig[k])
    for ax in (0, 1):
        eq([(ax, 1.0)], 0.0, 1e3)  # drive 0's first photo is the origin
        for d in range(1, len(heads)):  # every drive starts at the spawn
            eq([(2 * base[d] + ax, 1.0), (ax, -1.0)], 0.0, 1 / SIGMA_STEP)
    fixed = coo_matrix((vals, (rows, cols)), shape=(len(rhs), nvar)).tocsr()
    fixed_b = np.array(rhs)
    pa = np.array([base[d] + k for (d, k), _ in closures], int)
    pb = np.array([base[d] + k for _, (d, k) in closures], int)
    w = np.full(len(closures), 1 / SIGMA_SAME)
    xy, gap = None, np.zeros(0)
    for _ in range(6):  # iteratively re-weighted least squares
        r = np.arange(2 * len(closures))
        c_rows = np.r_[r, r]
        c_cols = np.r_[2 * pa, 2 * pa + 1, 2 * pb, 2 * pb + 1]
        c_vals = np.r_[w, w, -w, -w]
        loop = coo_matrix((c_vals, (c_rows, c_cols)), shape=(2 * len(closures), nvar))
        sol = lsqr(vstack([fixed, loop]), np.r_[fixed_b, np.zeros(2 * len(closures))], atol=1e-10, btol=1e-10)[0]
        xy = sol.reshape(-1, 2)
        gap = np.hypot(*(xy[pa] - xy[pb]).T) if len(closures) else np.zeros(0)
        w = (1 / SIGMA_SAME) / np.sqrt(1 + (gap / ROBUST) ** 2)
    return [xy[base[d]:base[d + 1]] for d in range(len(heads))], gap


def build_map(drives: list[dict], meas: list[dict], closures: list, start: float, sg=None,
              motion: dict | None = None) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """(x, y) and heading of every photo of every drive in one frame. ``start``: drive 0's first heading
    (it fixes the frame; drive 0's first photo is the origin). The other drives start at the spawn,
    turned by the image rotation between their first photo and drive 0's, snapped to the compass.
    ``closures``: ((drive, photo), (drive, photo)) pairs that are one place. ``motion``: per label
    (forward m, turn rad, distance spread m), default LABEL_MOTION. Returns positions and headings
    per drive, and each closure's final gap (m)."""
    motion = motion or LABEL_MOTION
    heads, dists, sigmas = [], [], []
    for d, (dr, m) in enumerate(zip(drives, meas)):
        lab = dr["labels"][:-1]
        turns = fuse_turns(np.array([motion[x][1] for x in lab]), m["hom"], m["ess"])
        h0 = start
        if d > 0:
            if sg is not None:
                h1, e1, _ = image_turns(sg, [drives[0]["images"][0], dr["images"][0]])
                h0 += float(fuse_turns(np.zeros(1), h1, e1)[0])
            if m["conf"][0] >= COMPASS_MIN and not math.isnan(m["comp"][0]):
                h0 = m["comp"][0] + (math.pi / 2) * round((h0 - m["comp"][0]) / (math.pi / 2))
        heads.append(compass_headings(h0, turns, m["comp"], m["conf"]))
        dists.append(np.array([motion[x][0] for x in lab]))
        sigmas.append(np.where(np.abs(turns) > LONG_TURN, SIGMA_GAP, [motion[x][2] for x in lab]))
    xy, gap = solve_positions(heads, dists, closures, sigmas)
    return xy, heads, gap


# ---- 3. links and costs


def step_cost(pa, ha: float, pb, hb: float) -> float:
    """Ticks to get from pose a to pose b: drive the distance, turn the heading change."""
    return float(np.hypot(*(np.asarray(pb) - np.asarray(pa))) / TICK_M + abs(wrap(hb - ha)) / TICK_RAD)


def edge_cost(frames, pos: np.ndarray, head: np.ndarray, a: int, b: int) -> float:
    """What a planner edge between map frames a and b costs: its ticks, plus SWITCH_TICKS unless the
    two are consecutive photos of one drive."""
    same = frames[a].trajectory == frames[b].trajectory and abs(a - b) == 1
    return step_cost(pos[a], head[a], pos[b], head[b]) + (0.0 if same else SWITCH_TICKS)


def propose_links(frames, pos: np.ndarray, head: np.ndarray, sg, linked) -> tuple[list, int, int]:
    """New same-place links: frames within LINK_R m and LINK_TURN of each other on the map (not
    neighbours on one drive, not ``linked(a, b)`` already), confirmed by SuperGlue with >= LINK_MIN
    inliers. Returns the links (a, b, inliers), how many candidates there were and how many were checked."""
    from scipy.spatial import cKDTree

    traj = np.array([f.trajectory for f in frames])
    pairs = cKDTree(pos).query_pairs(LINK_R, output_type="ndarray")
    if not len(pairs):
        return [], 0, 0
    ok = np.abs(wrap(head[pairs[:, 0]] - head[pairs[:, 1]])) < LINK_TURN
    ok &= ~((traj[pairs[:, 0]] == traj[pairs[:, 1]]) & (np.abs(pairs[:, 0] - pairs[:, 1]) < MIN_GAP))
    ok &= np.array([not linked(int(a), int(b)) for a, b in pairs])
    pairs = pairs[ok]
    count, keep = np.zeros(len(pos), int), []
    for k in np.argsort(np.hypot(*(pos[pairs[:, 0]] - pos[pairs[:, 1]]).T)):  # nearest first
        a, b = pairs[k]
        if count[a] < PER_PHOTO and count[b] < PER_PHOTO:
            keep.append((int(a), int(b)))
            count[a] += 1
            count[b] += 1
    feats: dict = {}  # SuperPoint features of a few photos at a time (they live on the GPU)

    def feat(i: int):
        if i not in feats:
            if len(feats) > 300:
                feats.clear()
            feats[i] = sg.features(cv2.imread(str(frames[i].path)))
        return feats[i]

    links = []
    for a, b in sorted(keep):
        n = sg.match(feat(a), feat(b), FOCAL).n
        if n >= LINK_MIN:
            links.append((a, b, n))
    return links, len(pairs), len(keep)

"""How well does the metric map (source/metric_map.py, the agent's --map-links) place every exploration
photo? Checked against the true positions of the local mazes.

    uv run --no-sync tools/odometry_check.py                  # every maze with saved truth
    uv run --no-sync tools/odometry_check.py --mazes 7,8 --plot

(A) labels: each photo's move label stands for the average true motion of that label, learned from
    the OTHER mazes (no maze grades itself), added up from the drive's first photo.
(B) images: the rotation between consecutive photos measured with SuperPoint + SuperGlue, as a
    homography (lecture 5 model selection: the right model for turning on the spot) and from the
    essential matrix (VO tutorial), compared with the true rotation. Then positions with (B)'s
    headings and (A)'s distances, and, as the limit of label distances, with the true headings.
(C) grid compass: the maze is a grid, so walls and floor tiles run in only 4 directions. Straight
    lines next to the plain floor or ceiling (wall bottoms and tops, tile edges) meet the horizon at
    vanishing points that give the heading modulo 90 deg from one photo (lecture 3: "a calibrated
    camera is a protractor"). (B)'s rotations pick which quarter turn.
(D) loop closures: with the compass headings fixed, every photo's position follows from a LINEAR
    least-squares problem (lecture 8's graph SLAM): each step moves its label distance along its
    heading; every drive starts at the spawn, drive 0's first photo is the origin ("we must fix the
    initial condition"); each of the agent's own verified shortcuts says two photos are one place.
    Closures that stay far off are down-weighted (Cauchy weights, lecture 5). Only drive 0's first pose
    is taken from the truth (it fixes the frame); the other drives get their start heading from the
    image rotation between their first photos and drive 0's.

A label only names the move right after the photo; the rest of the ticks to the next photo can be
something else (a "RIGHT" step turned 17 deg or 10 deg, an "IDLE" step moved 6 cm), which is why (B)
and (C) are measured at all. Reads cache/truth-<maze>/ (written by tools/run_check.py: the same photos
as data/local-<maze>, plus poses.json). Changes nothing; --plot writes logs/odometry_<maze>.png.
"""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

import cv2  # noqa: E402
import metric_map as mm  # noqa: E402
import numpy as np  # noqa: E402
from metric_map import COMPASS_MIN, ROBUST, wrap  # noqa: E402

LABELS = ("FORWARD", "BACKWARD", "LEFT", "RIGHT", "IDLE")
AT = (100, 300)  # report position errors after this many photos, and at the end of each drive
CACHE = ROOT / "cache" / "odometry"


def build_agent(data: Path):
    """The agent's map of ``data`` exactly as a run builds it (AnyLoc + SuperGlue, default settings)."""
    from anyloc import AnyLocScorer
    from sg_matcher import SGMatcher
    from sptm_agent import SptmAgent
    from sptm_models import TorchScorer

    scorer = AnyLocScorer(TorchScorer(ROOT / "models" / "sptm_r.pt", ROOT / "models" / "sptm_l.pt"),
                          cache_dir=ROOT / "cache")
    scorer.fit(data)
    return SptmAgent(data, scorer, goal_matcher=SGMatcher(), goal_min=20)


def agent_closures(agent, drives: list[dict]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """The agent's own verified shortcuts (AnyLoc proposals, 8-pair sequence check, SuperGlue) as
    ((drive, photo), (drive, photo)) pairs: the loop closures."""
    where = mm.frame_index(agent.frames, drives)
    return [(where[a], where[b]) for a, b in agent.shortcuts]


def load(maze: int) -> list[dict]:
    """Drives of one local maze: labels, true poses (x, y, yaw), photo paths."""
    return mm.load_drives(ROOT / "cache" / f"truth-{maze}")


def true_steps(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per step k -> k+1: distance along the heading at k (m) and change of heading (rad)."""
    d = np.diff(poses[:, :2], axis=0)
    fwd = d[:, 0] * np.cos(poses[:-1, 2]) + d[:, 1] * np.sin(poses[:-1, 2])
    return fwd, wrap(np.diff(poses[:, 2]))


def label_model(mazes: dict[int, list[dict]], skip: int) -> dict[str, tuple[float, float]]:
    """Average true (forward m, turn rad) per label over every maze except ``skip``."""
    acc = {lab: ([], []) for lab in LABELS + ("MIXED",)}
    for maze, drives in mazes.items():
        if maze == skip:
            continue
        for dr in drives:
            fwd, turn = true_steps(dr["poses"])
            for lab, f, a in zip(dr["labels"], fwd, turn):
                acc[lab][0].append(f)
                acc[lab][1].append(a)
    return {lab: (float(np.mean(f)) if f else 0.0, float(np.mean(a)) if a else 0.0) for lab, (f, a) in acc.items()}


def label_spread(mazes: dict[int, list[dict]], skip: int) -> dict[str, float]:
    """Standard deviation of the true forward motion per label (m), over every maze except ``skip``:
    how uncertain each step's distance is (lecture 7: weigh constraints by their uncertainty)."""
    acc = {lab: [] for lab in LABELS + ("MIXED",)}
    for maze, drives in mazes.items():
        if maze != skip:
            for dr in drives:
                for lab, f in zip(dr["labels"], true_steps(dr["poses"])[0]):
                    acc[lab].append(f)
    return {lab: max(float(np.std(f)), 0.01) if f else mm.SIGMA_STEP for lab, f in acc.items()}


def leave_one_out(mazes: dict[int, list[dict]], skip: int) -> dict[str, tuple[float, float, float]]:
    """metric_map's per-label motion (forward m, turn rad, spread m), learned without maze ``skip``."""
    model, spread = label_model(mazes, skip), label_spread(mazes, skip)
    return {lab: (*model[lab], spread[lab]) for lab in model}


def integrate(start: np.ndarray, fwd: np.ndarray, turn: np.ndarray) -> np.ndarray:
    """Poses from a start pose and per-step (forward, turn); the step moves along the heading at k."""
    out = [start.copy()]
    for f, a in zip(fwd, turn):
        x, y, yaw = out[-1]
        out.append(np.array([x + f * math.cos(yaw), y + f * math.sin(yaw), yaw + a]))
    return np.array(out)


def errors(est: np.ndarray, true: np.ndarray) -> tuple[list[float], list[float]]:
    """Position (m) and heading (deg) error after AT photos and at the end."""
    idx = [min(k, len(true) - 1) for k in AT] + [len(true) - 1]
    pos = [float(np.hypot(*(est[i, :2] - true[i, :2]))) for i in idx]
    head = [float(abs(math.degrees(wrap(est[i, 2] - true[i, 2])))) for i in idx]
    return pos, head


def draw(maze: int, drives: list[dict], paths: dict[str, list[np.ndarray]]) -> Path:
    scale, pad = 100, 20  # px per m
    colours = {"truth": (0, 160, 0), "labels": (0, 0, 220), "image headings": (220, 90, 0),
               "grid compass": (180, 0, 180), "compass, spawn start": (180, 0, 180),
               "+ loop closures": (0, 120, 255)}
    show = ("compass, spawn start", "+ loop closures") if paths.get("+ loop closures") else (
        "labels", "image headings", "grid compass")
    colours = {k: v for k, v in colours.items() if k == "truth" or (k in show and paths.get(k))}
    allp = np.vstack([dr["poses"][:, :2] for dr in drives]
                     + [p[:, :2] for k in colours if k != "truth" for p in paths[k]])
    lo = allp.min(0)
    size = (allp.max(0) - lo) * scale + 2 * pad
    img = np.full((int(size[1]), int(size[0]), 3), 255, np.uint8)
    for name, colour in colours.items():
        for p in ([dr["poses"] for dr in drives] if name == "truth" else paths[name]):
            pts = ((p[:, :2] - lo) * scale + pad).astype(np.int32)
            pts[:, 1] = img.shape[0] - pts[:, 1]
            cv2.polylines(img, [pts], False, colour, 1, cv2.LINE_AA)
    for i, (name, colour) in enumerate(colours.items()):
        cv2.putText(img, name, (pad, pad + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    out = ROOT / "logs" / f"odometry_{maze}.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mazes", default="", help="comma-separated; default: every cache/truth-<maze>")
    ap.add_argument("--no-images", action="store_true", help="labels only (A), no SuperGlue (fast)")
    ap.add_argument("--plot", action="store_true", help="write logs/odometry_<maze>.png")
    ap.add_argument("--no-closures", dest="closures", action="store_false",
                    help="skip step 2 (it builds the agent's map for the loop closures: ~1 min a maze)")
    args = ap.parse_args()
    found = sorted(int(p.name[6:]) for p in (ROOT / "cache").glob("truth-*") if p.name[6:].isdigit())
    mazes = {m: load(m) for m in found}  # the label model learns from every maze but the one checked
    todo = [int(m) for m in args.mazes.split(",")] if args.mazes else found
    sg = None
    if not args.no_images:
        from sg_matcher import SGMatcher

        sg = SGMatcher()
    at = " / ".join(f"{k} photos" for k in AT) + " / end"
    for maze in todo:
        drives, model = mazes[maze], label_model(mazes, maze)
        print(f"\n=== maze {maze}: {len(drives)} drives, {sum(len(d['labels']) for d in drives)} photos")
        print("label model (from the other mazes), mean true motion per photo: " + ", ".join(
            f"{lab} {f:+.3f} m {math.degrees(a):+.1f} deg" for lab, (f, a) in model.items() if lab in LABELS))
        rows = {name: [] for name in ("labels", "image headings", "grid compass", "compass, spawn start",
                                      "+ loop closures", "true headings")}
        paths = {name: [] for name in rows if name != "true headings"}
        summary = {}
        turn_err = {"homography": [], "essential": []}
        comp_err, comp_conf = [], []
        meas = []  # per drive: step rotations and compass (cached), shared with step 2
        for dr in drives:
            poses = dr["poses"]
            t_fwd, t_turn = true_steps(poses)
            l_fwd = np.array([model[lab][0] for lab in dr["labels"][:-1]])
            l_turn = np.array([model[lab][1] for lab in dr["labels"][:-1]])
            est = integrate(poses[0], l_fwd, l_turn)
            rows["labels"].append(errors(est, poses))
            paths["labels"].append(est)
            rows["true headings"].append(errors(integrate(poses[0], l_fwd, t_turn), poses))
            m = mm.measurements(sg, dr["images"], CACHE)
            meas.append(m)
            ok = m["conf"] >= COMPASS_MIN
            comp_err.append(np.degrees(np.abs((m["comp"][ok] - poses[ok, 2] + np.pi / 4) % (np.pi / 2) - np.pi / 4)))
            comp_conf.append(ok)
            turns = mm.fuse_turns(l_turn, m["hom"], m["ess"])  # the labels' turns when there are no images
            if sg is not None:
                # sign is fixed, + = turn left: it matched the true turn on 98-100% of clear turns of
                # every drive checked, so the truth is only used to score, never to correct
                for name, yaw in (("homography", m["hom"]), ("essential", m["ess"])):
                    turn_err[name].append(np.degrees(np.abs(wrap(yaw - t_turn)))[~np.isnan(yaw)])
                est = integrate(poses[0], l_fwd, turns)
                rows["image headings"].append(errors(est, poses))
                paths["image headings"].append(est)
            heads = mm.compass_headings(poses[0, 2], turns, m["comp"], m["conf"])
            est = integrate(poses[0], l_fwd, np.diff(heads))
            rows["grid compass"].append(errors(est, poses))
            paths["grid compass"].append(est)
        if args.closures:  # step 2: one map for all drives, without and with the loop closures
            origin = drives[0]["poses"][0]  # fixes the frame: the only truth used here
            motion = leave_one_out(mazes, maze)
            closures = agent_closures(build_agent(ROOT / "data" / f"local-{maze}"), drives)
            true_gap = np.array([np.hypot(*(drives[a]["poses"][i, :2] - drives[b]["poses"][j, :2]))
                                 for (a, i), (b, j) in closures])
            for name, cl in (("compass, spawn start", []), ("+ loop closures", closures)):
                xy, heads_sp, gap = mm.build_map(drives, meas, cl, origin[2], sg, motion)
                all_err = []
                for dr, p, h in zip(drives, xy, heads_sp):
                    true = dr["poses"][:, :2] - origin[:2]
                    rows[name].append(errors(np.c_[p, h], np.c_[true, dr["poses"][:, 2]]))
                    paths[name].append(np.c_[p + origin[:2], h])
                    all_err.append(np.hypot(*(p - true).T))
                e = np.concatenate(all_err)
                summary[name] = (f"median {np.median(e):.2f} m, 90% within {np.percentile(e, 90):.2f} m, "
                                 f"max {e.max():.2f} m; per drive median/max: " + ", ".join(
                                     f"{np.median(x):.2f}/{x.max():.2f}" for x in all_err))
            print(f"loop closures (the agent's verified shortcuts): {len(closures)}; true distance between "
                  f"the two photos: median {np.median(true_gap):.2f} m, max {true_gap.max():.2f} m; still "
                  f"apart > {ROBUST} m after solving (down-weighted): {int((gap > ROBUST).sum())}")
        if sg is not None:
            for name, errs in turn_err.items():
                e = np.concatenate(errs)
                print(f"rotation per photo from images ({name}): median error {np.median(e):.2f} deg, "
                      f"90% within {np.percentile(e, 90):.2f} deg, over {len(e)} steps")
        e = np.concatenate(comp_err)
        print(f"grid compass, heading mod 90 deg from one photo: median error {np.median(e):.2f} deg, "
              f"90% within {np.percentile(e, 90):.2f} deg, usable on {np.concatenate(comp_conf).mean():.0%} of photos")
        print(f"position error, median over drives, after {at} (heading error in brackets):")
        for name, r in rows.items():
            if not r:
                continue
            pos = np.median([x[0] for x in r], axis=0)
            head = np.median([x[1] for x in r], axis=0)
            print(f"  {name:21s} " + " / ".join(f"{p:5.2f} m ({h:4.0f} deg)" for p, h in zip(pos, head)))
        for name, text in summary.items():
            print(f"  every photo, {name}: {text}")
        if args.plot:
            print(f"plot: {draw(maze, drives, paths)}")


if __name__ == "__main__":
    main()

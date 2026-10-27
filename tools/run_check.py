"""Run the SPTM agent on a local maze and check the run against the truth -- one command.

    uv run --no-sync tools/run_check.py --name c31                  # maze 7, the usual motion seed
    uv run --no-sync tools/run_check.py --name m8 --maze 8
    uv run --no-sync tools/run_check.py --name c31_nosteer --agent-args=--no-steer   # note the "="
    uv run --no-sync tools/run_check.py --name c31_al --agent-args="--place anyloc --no-steer"
    uv run --no-sync tools/run_check.py --log logs/sptm7_c29.txt    # only check a finished run

1. runs source/sptm_agent.py --local <maze> --superglue --verbose and writes logs/<name>.txt
   (UTF-8, also printed live: no Tee-Object, no buffering, no encoding crash at the end)
2. runs tools/top_view.py on that log (logs/<name>_topview.png)
3. re-records the maze's exploration data WITH true poses (once, into cache/truth-<maze>/, the same
   recording cli.record_exploration_data makes) and replays the run, so it knows where the robot
   really was at every step -- and where every map node really is
4. prints a summary to read or paste, also saved as logs/<name>_summary.txt

Local runs only; the truth is never available on the server. Changes nothing in source/.
"""
import argparse
import collections
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

import numpy as np  # noqa: E402
import vis_nav_sim as sim  # noqa: E402
from vlad import load_frames  # noqa: E402

MOVE = re.compile(r"^\[(\d+)\] node (-?\d+) hops (\d+) (\S+) .*?\(<Action\.\w+: (\d+)>, (\d+)\)")
EVENTS = {  # log line pattern -> label in the summary
    r"\] node .* reverse ": "reverse steps (Change 28)",
    r"detour around": "detours (Change 29)",
    r"dropped shortcut": "dropped shortcuts (Change 25c)",
    r"jump .* accepted": "far jumps accepted (Change 27)",
    r"jump .* rejected": "far jumps rejected (Change 27)",
    r"\] node .* stuck ": "stuck recoveries",
}


def read_log(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):  # Tee-Object in Windows PowerShell writes UTF-16
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


def run_agent(args) -> Path:
    log = ROOT / "logs" / f"{args.name}.txt"
    log.parent.mkdir(exist_ok=True)
    cmd = [sys.executable, "-u", "source/sptm_agent.py", "--local", str(args.maze), "--superglue", "--verbose",
           "--motion-seed", str(args.motion_seed), "--budget", str(args.budget), "--log-dir", f"logs/{args.name}",
           "--yes", "--no-browser", *shlex.split(args.agent_args)]
    print("running:", " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    if args.hide:  # no viewer window (closing the window ends the run unscored: t_m37, t_m52)
        env.update(SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    with log.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            out.write(line)
        proc.wait()
    print(f"\nlog: {log}  (exit code {proc.returncode})")
    return log


def truth_poses(maze: int) -> np.ndarray:
    """True (x, y, yaw) of every map node, in the agent's node order (vlad.load_frames)."""
    d = ROOT / "cache" / f"truth-{maze}"
    if not (d / "traj_0" / "poses.json").exists():
        print(f"recording maze {maze} with true poses into {d} (once) ...", flush=True)
        world = sim.Simulator(sim.Textures.find(), maze, motion_noise=sim.DEFAULT_NOISE, motion_seed=maze)
        world.record(d, routes=3, seed=maze, capture_every=5, student_files=False)
    frames, bounds = load_frames(d)
    pose_of = {}
    for t in range(len(bounds)):
        td = d / f"traj_{t}"
        for r, p in zip(json.loads((td / "data_info.json").read_text()), json.loads((td / "poses.json").read_text()),
                        strict=True):
            pose_of[r["image"], t] = p
    used, _ = load_frames(ROOT / "data" / f"local-{maze}")
    if len(used) != len(frames):
        print(f"WARNING: data/local-{maze} has {len(used)} frames, the truth recording {len(frames)}: node "
              "positions may not line up")
    return np.array([pose_of[f.path.name, f.trajectory] for f in frames])


def check(log: Path, top_view_out: str) -> str:
    text = read_log(log)
    body = text[text.find("starting the local simulator"):]
    lines = body.splitlines()
    maze = int(m[1]) if (m := re.search(r"Local maze (\d+)", body)) else 7
    seed = int(m[1]) if (m := re.search(r"motion_seed=(\d+)", body)) else None
    moves = [m.groups() for m in map(MOVE.match, lines) if m]
    if seed is None or not moves:
        return "no local --verbose run found in this log"
    P = truth_poses(maze)
    world = sim.Simulator(sim.Textures.find(), maze, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed)
    gx, gy, _ = world.target
    rows = []  # (step, node, hops, why, true x, true y, distance to goal, belief error)
    for t, node, hops, why, action, n in moves:
        x, y, _ = world.pose()
        node = int(node)
        err = float(np.hypot(*(P[node, :2] - (x, y)))) if 0 <= node < len(P) else float("nan")
        rows.append((int(t), node, int(hops), why, x, y, float(np.hypot(gx - x, gy - y)), err))
        n = max(1, min(int(n), sim.MAX_REPEAT))
        world.step_ticks(int(action), n, [n - 1])
    err = np.array([r[7] for r in rows])
    dist = np.array([r[6] for r in rows])
    k = int(np.argmin(dist))

    wrong = []  # far jumps that moved the belief > 0.5 m further from the truth
    for a, b in zip(rows, rows[1:]):
        if a[1] >= 0 and b[1] >= 0 and abs(b[1] - a[1]) > 20 and b[7] > a[7] + 0.5:
            wrong.append(f"step {b[0]}: node {a[1]} -> {b[1]}, belief error {a[7]:.1f} -> {b[7]:.1f} m")

    still, best, start = 0, (0, 0), 0  # longest stretch with the robot within 0.1 m of one spot
    for i in range(1, len(rows)):
        if np.hypot(rows[i][4] - rows[start][4], rows[i][5] - rows[start][5]) < 0.1:
            still = i - start
            best = max(best, (still, start))
        else:
            start = i
    s_len, s_at = best
    nodes_there = collections.Counter(r[1] for r in rows[s_at: s_at + s_len + 1]).most_common(3)

    result = re.search(r"result: Result\(goal_tier='(\w+)', nav_steps=(\d+).*?points=([\d.]+)", body)
    out = [f"=== {log.name}: maze {maze}, motion seed {seed}, {len(moves)} decisions ===",
           f"result: {result[1]}, {result[3]} points, {result[2]} ticks" if result else "result: (not found)",
           f"closest to the goal: {dist[k]:.2f} m at step {rows[k][0]} (believed hops {rows[k][2]}); "
           f"at the end {dist[-1]:.2f} m",
           f"belief error: median {np.nanmedian(err):.2f} m, more than 1 m off in {np.nanmean(err > 1):.0%} of steps",
           f"wrong far jumps: {len(wrong)}" + "".join(f"\n    {w}" for w in wrong[:6]),
           f"longest time not moving (within 0.1 m): {s_len} steps from step {rows[s_at][0]}, believed nodes "
           + ", ".join(f"{n} x{c}" for n, c in nodes_there),
           "decisions: " + ", ".join(f"{w} {c}" for w, c in collections.Counter(r[3] for r in rows).most_common()),
           "events: " + ", ".join(f"{label} {sum(bool(re.search(p, ln)) for ln in lines)}" for p, label in EVENTS.items())]
    tv = [ln for ln in top_view_out.splitlines() if ln.startswith(("closest", "never", "steps spent", "saved"))]
    final = world.pose()  # after the last move: where it checked in, or where the budget ran out
    return "\n".join(out + ["top view: " + ln for ln in tv] + diagnose(rows, lines, body, final, world.target, err))


def diagnose(rows, lines, body, final, target, err) -> list[str]:
    """Name the likely cause(s) of a failed run, from the replay against the truth."""
    gx, gy, gyaw = target
    found = []
    dist = np.array([r[6] for r in rows])
    if re.search(r"\] node .* arrived ", body):  # the agent checked in: how far off, and in which direction?
        x, y, yaw = final
        dx, dy = x - gx, y - gy
        along = dx * np.cos(gyaw) + dy * np.sin(gyaw)
        side = -dx * np.sin(gyaw) + dy * np.cos(gyaw)
        head = abs((np.degrees(yaw - gyaw) + 180) % 360 - 180)
        miss = [f"{name} {v:+.2f} m" for name, v in (("along the goal heading", along), ("sideways", side))
                if abs(v) > 0.1] + ([f"heading {head:.0f} deg"] if head > 10 else [])
        found.append("LAST METRE: checked in " + (", ".join(miss) if miss else "inside the best bands")
                     + (" -- see the approach-* lines before 'arrived'" if miss else ""))
    elif dist.min() < 0.5:
        found.append(f"LAST METRE: got within {dist.min():.2f} m but never checked in -- see the approach-* lines")
    flips = [(a, b) for a, b in zip(rows, rows[1:])
             if b[2] - a[2] > 100 and a[7] < 0.6 and b[7] < 0.6]  # hops jump up while both beliefs are right
    if flips:
        ex = ", ".join(f"step {b[0]}: {a[1]}->{b[1]} hops {a[2]}->{b[2]}" for a, b in flips[:3])
        n_link = sum("linked" in ln for ln in lines)
        found.append(f"MAP GAP: the plan flipped {len(flips)}x while the belief was right ({ex}); "
                     f"two drives of the same place are not linked ({n_link} run-time links made)")
    if np.nanmean(err > 1) > 0.10:
        found.append(f"LOCALISATION: belief > 1 m off in {np.nanmean(err > 1):.0%} of steps")
    best, start = (0, 0), 0  # longest stay within 0.1 m, and what the agent was doing there
    for i in range(1, len(rows)):
        if np.hypot(rows[i][4] - rows[start][4], rows[i][5] - rows[start][5]) < 0.1:
            best = max(best, (i - start, start))
        else:
            start = i
    n, s = best
    if n > 200:
        kinds = [r[3] for r in rows[s: s + n + 1]]
        top = collections.Counter(kinds).most_common(2)
        switches = sum(a != b for a, b in zip(kinds, kinds[1:]))
        fight = f"; '{top[0][0]}' and '{top[1][0]}' alternate every {n / max(switches, 1):.1f} decisions" \
            if len(top) > 1 and switches > n / 6 else ""
        found.append(f"STALL: {n} decisions within 0.1 m from step {rows[s][0]} "
                     f"({', '.join(f'{k} {c}' for k, c in top)}){fight}")
    if not re.search(r"\] node .* arrived ", body) and len(rows) > 50:
        tail = [r[2] for r in rows[-len(rows) // 5:]]
        if tail[-1] < tail[0] - 20:
            found.append(f"BUDGET: still progressing at the end (hops {tail[0]} -> {tail[-1]} in the last 20%): "
                         "give it more ticks (the challenge allows 60,000)")
    return ["likely causes:"] + [f"  - {f}" for f in found] if found else ["likely causes: none found"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="check", help="log name: logs/<name>.txt (default: check)")
    ap.add_argument("--maze", type=int, default=7)
    ap.add_argument("--motion-seed", type=int, default=2664226132, help="the same seed = a fair before/after test")
    ap.add_argument("--budget", type=int, default=8000, help="ticks; keeps a failing local run short")
    ap.add_argument("--agent-args", default="",
                    help='extra agent options; write it with "=": --agent-args="--no-steer --min-run 6"')
    ap.add_argument("--log", type=Path, default=None, help="only check this finished run, do not run the agent")
    ap.add_argument("--hide", action="store_true",
                    help="no viewer window. With the window shown, minimise it -- closing it ends the run unscored")
    args = ap.parse_args()
    log = args.log if args.log is not None else run_agent(args)
    tv = subprocess.run([sys.executable, "tools/top_view.py", str(log)], cwd=ROOT, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    summary = check(Path(log), tv.stdout)
    print("\n" + summary)
    out = Path(log).with_name(Path(log).stem + "_summary.txt")
    out.write_text(summary + "\n", encoding="utf-8")
    print(f"\nsummary saved: {out}")


if __name__ == "__main__":
    main()

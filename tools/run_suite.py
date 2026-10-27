"""Run the agent on several local mazes and compare -- one command, one table.

    uv run --no-sync tools/run_suite.py --name base --mazes 7,21,37,52
    uv run --no-sync tools/run_suite.py --name bayes --mazes 7,21,37,52 --agent-args="--localiser bayes"

Runs tools/run_check.py on every maze (same motion seed and budget, so two suites are a fair
comparison of two versions of the agent), then prints a table and writes logs/<name>_suite.md:
points, closest approach, belief error, wrong jumps, longest stall, and each failure's likely causes.
Mazes the code was never tuned on say whether a change GENERALISES; one maze never does.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {  # summary line pattern -> column
    "result": r"result: (\w+), ([\d.]+) points, (\d+) ticks",
    "closest": r"closest to the goal: ([\d.]+) m",
    "belief": r"more than 1 m off in (\d+)%",
    "jumps": r"wrong far jumps: (\d+)",
    "stall": r"longest time not moving \(within 0.1 m\): (\d+) steps",
}


def parse(summary: str) -> dict:
    row = {}
    for key, pat in FIELDS.items():
        m = re.search(pat, summary)
        row[key] = m.groups() if m else None
    row["causes"] = [ln.strip()[2:].split(":")[0] for ln in summary.splitlines() if ln.startswith("  - ")]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="suite name: logs/<name>_m<maze>.txt and logs/<name>_suite.md")
    ap.add_argument("--mazes", default="7,21,37,52", help="comma-separated maze seeds")
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--agent-args", default="", help='e.g. --agent-args="--localiser bayes"')
    args = ap.parse_args()
    rows = []
    for maze in [int(m) for m in args.mazes.split(",")]:
        name = f"{args.name}_m{maze}"
        t0 = time.time()
        print(f"maze {maze}: running ({name}) ...", flush=True)
        subprocess.run([sys.executable, "tools/run_check.py", "--name", name, "--maze", str(maze),
                        "--budget", str(args.budget), f"--agent-args={args.agent_args}"],
                       cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        summary_file = ROOT / "logs" / f"{name}_summary.txt"
        summary = summary_file.read_text(encoding="utf-8") if summary_file.exists() else ""
        row = parse(summary)
        row["maze"] = maze
        rows.append(row)
        res = row["result"]
        print(f"maze {maze}: {res[1] if res else '?'} points, closest {row['closest'][0] if row['closest'] else '?'} m, "
              f"causes: {', '.join(row['causes']) or '-'}  ({(time.time() - t0) / 60:.0f} min)", flush=True)

    head = "| maze | points | ticks | closest (m) | belief >1 m off | wrong jumps | longest stall | likely causes |"
    lines = [f"# suite {args.name}  (budget {args.budget}, agent args: `{args.agent_args or '-'}`)", "",
             head, "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = (lambda k, i=0: r[k][i] if r[k] else "?")  # noqa: E731
        lines.append(f"| {r['maze']} | {g('result', 1)} | {g('result', 2)} | {g('closest')} | {g('belief')}% | "
                     f"{g('jumps')} | {g('stall')} | {', '.join(r['causes']) or '-'} |")
    scored = sum(1 for r in rows if r["result"] and float(r["result"][1]) > 0)
    near = sum(1 for r in rows if r["closest"] and float(r["closest"][0]) < 0.5)
    lines += ["", f"scored on {scored}/{len(rows)} mazes; within 0.5 m of the goal on {near}/{len(rows)}"]
    out = ROOT / "logs" / f"{args.name}_suite.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines) + f"\n\nsaved {out}")


if __name__ == "__main__":
    main()

"""Check bearing_from_e on exploration pairs where the robot only drove FORWARD.

For a pair (i -> i+1) the robot moved forward, so:
  A = frame i+1 (taken further ahead), B = frame i  -> A's spot is AHEAD of B:  bearing ~ 0
  A = frame i,  B = frame i+1                      -> A's spot is BEHIND B:   bearing ~ +-180
If the "ahead" column is near 0 and the "behind" column near +-180, the sign is right and
bearing_from_e can steer the robot toward the spot a goal photo was taken from.
Only reads data; changes nothing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import cv2  # noqa: E402
from geometry import bearing_from_e  # noqa: E402
from sg_matcher import SGMatcher  # noqa: E402
from vlad import load_frames  # noqa: E402

frames, _ = load_frames(Path("data/local-7"))
sg = SGMatcher()
shown = 0
for i in range(len(frames) - 1):
    f, g = frames[i], frames[i + 1]
    if f.trajectory != g.trajectory or f.action != "FORWARD":
        continue
    fa, fb = sg.features(cv2.imread(str(f.path))), sg.features(cv2.imread(str(g.path)))
    ahead = bearing_from_e(sg.match(fb, fa, focal=92.0), 92.0)  # A = i+1 (ahead), B = i
    behind = bearing_from_e(sg.match(fa, fb, focal=92.0), 92.0)  # A = i (behind), B = i+1
    fmt = lambda x: "none" if x is None else f"{x:+7.1f}"  # noqa: E731
    print(f"FORWARD {i:4d}->{i + 1:4d}   ahead (expect ~0): {fmt(ahead)}   behind (expect ~+-180): {fmt(behind)}")
    shown += 1
    if shown >= 12:
        break

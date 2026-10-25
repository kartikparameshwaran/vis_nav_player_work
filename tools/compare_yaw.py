"""Check yaw_from_e against shift_deg on exploration pairs where the robot only turned."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import cv2  # noqa: E402
from geometry import Matcher, scale_ratio, shift_deg, yaw_from_e
  # noqa: E402
from sg_matcher import SGMatcher  # noqa: E402
from vlad import load_frames  # noqa: E402

frames, _ = load_frames(Path("data/local-7"))
sg = SGMatcher()
shown = {"LEFT": 0, "RIGHT": 0}
for i in range(len(frames) - 1):
    f, g = frames[i], frames[i + 1]
    if f.trajectory != g.trajectory or f.action not in shown or shown[f.action] >= 8:
        continue
    m = sg.match(sg.features(cv2.imread(str(f.path))), sg.features(cv2.imread(str(g.path))), focal=92.0)
    y = yaw_from_e(m, 92.0)
    ys = "none" if y is None else f"{y:+6.1f}"
    print(f"{f.action:5s} {i:4d}->{i + 1:4d}  n={m.n:3d}  shift_deg={shift_deg(m, 92.0):+6.1f}  yaw_E={ys}")
    shown[f.action] += 1
    if all(v >= 8 for v in shown.values()):
        break

"""Save map frames (goal node and where the agent thinks it is) under the goal photos, to check by eye.

    uv run --no-sync tools/show_nodes.py                 # nodes 1569 1568 1437 983
    uv run --no-sync tools/show_nodes.py 1569 1700 1313  # any nodes you like

Writes logs/goal_check.jpg: the four goal photos (data/local-7/target.jpg) on top, the chosen
map frames below, each labelled with its node number. Only reads data; changes nothing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from vlad import load_frames  # noqa: E402

frames, _ = load_frames(Path("data/local-7"))
nodes = [int(a) for a in sys.argv[1:]] or [1569, 1568, 1437, 983]
tiles = []
for n in nodes:
    im = cv2.resize(cv2.imread(str(frames[n].path)), (320, 240))
    cv2.putText(im, f"node {n}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    tiles.append(im)
row = np.hstack(tiles)
target = cv2.imread("data/local-7/target.jpg")
target = cv2.resize(target, (row.shape[1], int(target.shape[0] * row.shape[1] / target.shape[1])))
Path("logs").mkdir(exist_ok=True)
cv2.imwrite("logs/goal_check.jpg", np.vstack([target, row]))
print("saved logs/goal_check.jpg  nodes:", nodes)

"""
Mid-term version: the mid-term run's code and flags (version 4 = my_agent_1348.py, plus --goal-place and
--confirm-checkin; 6/10 in 9,095 steps, 0.022 m) with ONE change: --subsample 3 (Change 47).

Why: the mid-term exploration photos are one tick apart -- a recorded turn photo is 3.4 deg (SuperGlue,
106 pairs) against 10.1-10.6 deg on the Development and local mazes, and the maze has 16,061 records, what a
normal maze gives when photographed every tick (local maze 25 that way: 15,751). Every hop count in the agent
(waypoint look-ahead, undo size, localisation window, the L network's training distance) was tuned on the
~3x coarser photos, so on the mid-term steering cost 7-10 ticks per photo of progress. Keeping every 3rd
photo gives the agent the spacing it was tuned on.
Logs go to logs/my_agent_midterm/.

    uv run source/my_agent_midterm.py               # one real attempt (VIS_NAV_CHALLENGE in .env)
    uv run source/my_agent_midterm.py --local 25    # free, on your machine
"""

from __future__ import annotations

import sys

from my_agent import MyAgent, with_defaults
from sptm_agent import main

FLAGS = ("--smooth-steer", "--verbose", "--steer-hold", "--detour-once", "--map-ticks", "--think-confused",
         "--goal-place", "--confirm-checkin")
SUBSAMPLE = "3"

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not any(a == "--subsample" or a.startswith("--subsample=") for a in argv):
        argv += ["--subsample", SUBSAMPLE]
    sys.argv[1:] = with_defaults(argv, flags=FLAGS, log_dir="logs/my_agent_midterm")
    main(MyAgent)

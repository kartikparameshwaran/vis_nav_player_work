"""
Version 2: 6/10 in 2,767 steps on the Development Challenge (SPL 0.27); version 1 took 3,073.
Version 1 (my_agent_3073.py) plus two rules that make the robot commit instead of re-deciding:

    --steer-hold   measured steering, once in control, keeps it down to 30 SuperGlue inliers
                   (40 to take control), so L no longer undoes measured turns (Change 40)
    --detour-once  a stall makes each route edge costlier at most once, so detours cannot stack
                   up and flip the plan between neighbouring photos (Change 41)

Logs go to logs/my_agent_2767/.

    uv run source/my_agent_2767.py               # one real attempt (VIS_NAV_CHALLENGE in .env)
    uv run source/my_agent_2767.py --local 8     # free, on your machine
"""

from __future__ import annotations

import sys

from my_agent import MyAgent, with_defaults
from sptm_agent import main

FLAGS = ("--smooth-steer", "--verbose", "--steer-hold", "--detour-once")

if __name__ == "__main__":
    sys.argv[1:] = with_defaults(sys.argv[1:], flags=FLAGS, log_dir="logs/my_agent_2767")
    main(MyAgent)

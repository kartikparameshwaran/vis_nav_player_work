"""
Version 4: 6/10 in 1,820 steps on the Development Challenge (SPL 0.42, 2026-10-28), against version 3's
1,872 (my_agent_1872.py) -- version 3 plus thinking when confused (Change 44, --think-confused).

When the robot has believed it is at the same photo for 3 decisions in a row, it checks its view with
SuperGlue against the next 5 photos of its route and believes one that clearly matches better: never
backwards, never off the route, and not within 8 hops of the goal. This costs time, not steps. The same
version-3 code alone gave 1,872, 2,247 and one stuck run on the dev maze, so one run is not yet proof.
Logs go to logs/my_agent_1820/.

    uv run source/my_agent_1820.py               # one real attempt (VIS_NAV_CHALLENGE in .env)
    uv run source/my_agent_1820.py --local 8     # free, on your machine
"""

from __future__ import annotations

import sys

from my_agent import MyAgent, with_defaults
from sptm_agent import main

FLAGS = ("--smooth-steer", "--verbose", "--steer-hold", "--detour-once", "--map-ticks", "--think-confused")

if __name__ == "__main__":
    sys.argv[1:] = with_defaults(sys.argv[1:], flags=FLAGS, log_dir="logs/my_agent_1820")
    main(MyAgent)

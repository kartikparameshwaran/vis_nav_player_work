"""
Version 3: 6/10 in 1,872 steps on the Development Challenge (SPL 0.41, 0.009 m, 1.6 deg), against
version 2's 2,767 (my_agent_2767.py) -- version 2 planning on a map in metres (Changes 42-43).

Before connecting -- a few extra minutes, which cost no steps -- it places every exploration photo on
a map (grid compass + SuperGlue rotations + loop closures, source/metric_map.py), links photos the map
AND SuperGlue agree are one place, and costs every route edge in ticks (--map-ticks: what this run
used). A link that leads into a bump, or nowhere for 30 decisions, is dropped at once (Change 43).
On local maze 8 the same version needed 5,406 ticks against version 2's 2,618 (both 6/10): one dev
run is not yet proof. Logs go to logs/my_agent_1872/.

    uv run source/my_agent_1872.py               # one real attempt (VIS_NAV_CHALLENGE in .env)
    uv run source/my_agent_1872.py --local 8     # free, on your machine
"""

from __future__ import annotations

import sys

from my_agent import MyAgent, with_defaults
from sptm_agent import main

FLAGS = ("--smooth-steer", "--verbose", "--steer-hold", "--detour-once", "--map-ticks")

if __name__ == "__main__":
    sys.argv[1:] = with_defaults(sys.argv[1:], flags=FLAGS, log_dir="logs/my_agent_1872")
    main(MyAgent)

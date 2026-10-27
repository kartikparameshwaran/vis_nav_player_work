"""
Version 1, the Development Challenge dry run: 6/10 in 3,073 steps (SPL 0.25; git tag dryrun-6pts).
The SPTM agent with --smooth-steer, exactly what my_agent.py runs; logs go to logs/my_agent_3073/.

    uv run source/my_agent_3073.py               # one real attempt (VIS_NAV_CHALLENGE in .env)
    uv run source/my_agent_3073.py --local 8     # free, on your machine
"""

from __future__ import annotations

import sys

from my_agent import MyAgent, with_defaults
from sptm_agent import main

if __name__ == "__main__":
    sys.argv[1:] = with_defaults(sys.argv[1:], log_dir="logs/my_agent_3073")
    main(MyAgent)

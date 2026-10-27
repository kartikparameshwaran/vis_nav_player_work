"""
The challenge entry point. The course runs ``uv run source/my_agent.py``: this is the SPTM agent
(source/sptm_agent.py, class SptmAgent) unchanged, started with the settings of its best local
runs (git tag best-s1, mazes 8 and 52 solved): --smooth-steer, --verbose and a log directory.

    uv run source/my_agent.py --local 8      # free, on your machine
    uv run source/my_agent.py                # one real attempt (VIS_NAV_CHALLENGE in .env)

Every other sptm_agent.py option still works here (see --help). The Bayes-filter agent that used
to live in this file is source/bayes_agent.py.
"""

from __future__ import annotations

import sys

from sptm_agent import SptmAgent, main


class MyAgent(SptmAgent):
    """The course's name for "your agent": the SPTM agent, with nothing changed."""


def with_defaults(
    argv: list[str],
    flags: tuple[str, ...] = ("--smooth-steer", "--verbose"),
    log_dir: str = "logs/my_agent",
) -> list[str]:
    """The best runs' flags, added unless already given: a forgotten flag cannot cost an attempt.
    my_agent_3073.py and my_agent_2767.py pass their own flags and log directory."""
    out = list(argv)
    for flag in flags:
        if flag not in out:
            out.append(flag)
    if not any(a == "--log-dir" or a.startswith("--log-dir=") for a in out):
        out += ["--log-dir", log_dir]
    return out


if __name__ == "__main__":
    sys.argv[1:] = with_defaults(sys.argv[1:])
    main(MyAgent)

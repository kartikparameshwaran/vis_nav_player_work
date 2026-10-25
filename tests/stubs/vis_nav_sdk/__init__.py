"""Minimal stand-in for ``vis_nav_sdk``, only so the agent logic can be unit-tested where the
real SDK cannot be installed. Mirrors the names and constructor signatures used in
tests/test_agents.py. Never used when the real SDK is importable (see tests/conftest.py)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class Action(enum.IntFlag):
    IDLE = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 4
    RIGHT = 8
    CHECKIN = 16
    QUIT = 32


WAIT = (Action.IDLE, 1)


class Agent:
    viewer = None

    def setup(self, info) -> None: ...
    def finish(self, result) -> None: ...
    def hud(self) -> list[str]:
        return []

    def panel(self):
        return None


@dataclass
class Camera:
    width: int
    height: int
    color: str
    encoding: str
    K: np.ndarray


@dataclass
class Limits:
    max_steps: int
    max_repeat: int = 8
    attempts_used: int = 0
    attempts_allowed: int = 3
    silence_timeout_s: int = 120
    grace_s: int = 300
    session_timeout_s: int = 3600


@dataclass
class SessionInfo:
    session_id: str
    challenge: dict
    camera: Camera
    limits: Limits
    targets: list = field(default_factory=list)


@dataclass
class Observation:
    image: np.ndarray
    step: int
    steps_left: int


@dataclass
class Arrival:
    distance_m: float
    heading_deg: float


@dataclass
class Result:
    goal_tier: str
    nav_steps: int
    spl: float
    points: float
    max_points: float
    awarded: dict
    job_id: str
    session_id: str
    arrival: Arrival


SimError = RuntimeError


def run(*a, **k):  # pragma: no cover
    raise NotImplementedError("stub SDK: use tests/minisim.run_episode")

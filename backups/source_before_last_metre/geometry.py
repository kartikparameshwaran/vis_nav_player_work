"""Two-view geometry helpers: RootSIFT matching with RANSAC, plus the two numbers the agent
steers by near the goal: how far the scene shifted sideways (heading) and how much it grew or
shrank (distance along the view direction)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Feat:
    pts: np.ndarray  # (n, 2) float32, x y
    des: np.ndarray  # (n, 128) float32, RootSIFT


@dataclass
class Match:
    a: np.ndarray  # inlier points in image A (m, 2)
    b: np.ndarray  # inlier points in image B (m, 2)

    @property
    def n(self) -> int:
        return len(self.a)


class Matcher:
    def __init__(self, n_features: int = 600, ratio: float = 0.8, ransac_px: float = 1.5):
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio = ratio
        self.ransac_px = ransac_px

    def features(self, image: np.ndarray) -> Feat:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(kp) == 0:
            return Feat(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
        des = np.sqrt(des / (des.sum(1, keepdims=True) + 1e-9)).astype(np.float32)
        return Feat(np.array([k.pt for k in kp], np.float32), des)

    def match(self, a: Feat, b: Feat, focal: float | None = None) -> Match:
        """Geometrically verified matches. Fundamental-matrix RANSAC by default (permissive:
        any match along an epipolar line passes); with ``focal`` the essential matrix under the
        known intrinsics, which is much stricter and is what shortcut verification uses."""
        empty = Match(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))
        if len(a.des) < 8 or len(b.des) < 8:
            return empty
        pairs = self.bf.knnMatch(a.des, b.des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        if len(good) < 8:
            return empty
        pa = a.pts[[m.queryIdx for m in good]]
        pb = b.pts[[m.trainIdx for m in good]]
        if focal is not None:
            k = np.array([[focal, 0, 160.0], [0, focal, 120.0], [0, 0, 1.0]])
            _, mask = cv2.findEssentialMat(pa, pb, k, cv2.RANSAC, 0.999, 1.0)
        else:
            _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, self.ransac_px, 0.99)
        if mask is None:
            return empty
        keep = mask.ravel().astype(bool)
        return Match(pa[keep], pb[keep])


def shift_deg(m: Match, focal: float) -> float:
    """Median horizontal motion of the scene from A to B, as the yaw that would cause it.
    Positive: content in B sits to the right of where A has it. 0 when there is no match."""
    if m.n < 6:
        return 0.0
    return math.degrees(math.atan(float(np.median(m.b[:, 0] - m.a[:, 0])) / focal))


def scale_ratio(m: Match, pairs: int = 300, seed: int = 0) -> float:
    """Median size ratio of B over A (> 1: B is closer to the scene). 1.0 when unknown."""
    if m.n < 6:
        return 1.0
    rng = np.random.default_rng(seed)
    i = rng.integers(0, m.n, pairs)
    j = rng.integers(0, m.n, pairs)
    da = np.linalg.norm(m.a[i] - m.a[j], axis=1)
    db = np.linalg.norm(m.b[i] - m.b[j], axis=1)
    ok = da > 15
    return float(np.median(db[ok] / da[ok])) if ok.sum() >= 5 else 1.0

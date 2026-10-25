"""SuperPoint + SuperGlue (Sarlin et al., CVPR 2020) behind the same interface as
geometry.Matcher: features(image) -> features, match(a, b, focal=None) -> Match.
Official code in third_party/SuperGlue (magicleap/SuperGluePretrainedNetwork).
After SuperGlue matches, the same RANSAC check as geometry.Matcher keeps only
geometrically consistent pairs, so Match.n stays "number of inliers"."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from geometry import Match

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "SuperGlue"))
from models.superglue import SuperGlue  # noqa: E402
from models.superpoint import SuperPoint  # noqa: E402


@dataclass
class SPFeat:
    pts: np.ndarray  # (n, 2) float32, x y
    kpts: torch.Tensor  # (1, n, 2)
    scores: torch.Tensor  # (1, n)
    desc: torch.Tensor  # (1, 256, n)
    shape: tuple  # (1, 1, H, W) of the image


class SGMatcher:
    def __init__(self, weights: str = "indoor", max_keypoints: int = 512,
                 match_threshold: float = 0.2, ransac_px: float = 1.5, device: str | None = None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sp = SuperPoint({"max_keypoints": max_keypoints}).eval().to(self.dev)
        self.sg = SuperGlue({"weights": weights, "match_threshold": match_threshold}).eval().to(self.dev)
        self.ransac_px = ransac_px

    @torch.no_grad()
    def features(self, image: np.ndarray) -> SPFeat:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x = torch.from_numpy(gray).float().div(255.0)[None, None].to(self.dev)
        out = self.sp({"image": x})
        k = out["keypoints"][0]
        return SPFeat(k.cpu().numpy().astype(np.float32), k[None], out["scores"][0][None],
                      out["descriptors"][0][None], tuple(x.shape))

    @torch.no_grad()
    def match(self, a: SPFeat, b: SPFeat, focal: float | None = None) -> Match:
        empty = Match(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))
        if len(a.pts) < 8 or len(b.pts) < 8:
            return empty
        data = {
            "keypoints0": a.kpts, "scores0": a.scores, "descriptors0": a.desc,
            "keypoints1": b.kpts, "scores1": b.scores, "descriptors1": b.desc,
            "image0": torch.empty(a.shape, device=self.dev), "image1": torch.empty(b.shape, device=self.dev),
        }
        m0 = self.sg(data)["matches0"][0].cpu().numpy()
        ok = m0 > -1
        pa, pb = a.pts[ok], b.pts[m0[ok]]
        if len(pa) < 8:
            return empty
        if focal is not None:  # essential matrix with known intrinsics (stricter)
            k = np.array([[focal, 0, 160.0], [0, focal, 120.0], [0, 0, 1.0]])
            _, mask = cv2.findEssentialMat(pa, pb, k, cv2.RANSAC, 0.999, 1.0)
        else:  # fundamental matrix, same as geometry.Matcher's default
            _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, self.ransac_px, 0.99)
        if mask is None:
            return empty
        keep = mask.ravel().astype(bool)
        return Match(pa[keep], pb[keep])

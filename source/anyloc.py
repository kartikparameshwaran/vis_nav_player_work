"""
AnyLoc-style place recognition for the SPTM agent (Change 19, fix 1C).

AnyLoc (Keetha, Mishra et al., IEEE RA-L 2023) describes an image by dense DINOv2 patch
features aggregated with VLAD, using a vocabulary built from images of the target
environment, with NO training on that environment. That is what we need: our learned R
memorised the 32 training mazes' wall textures (recall@1 ~0.73 on unseen mazes).

Here, per image:
  DINOv2 ViT-S/14 patch tokens (official facebookresearch/dinov2 via torch.hub)
  -> hard-assign each patch to one of K cluster centres (k-means on exploration patches)
  -> VLAD: sum of residuals per centre, intra-normalised, flattened, L2-normalised
  -> PCA to ``dims`` (keeps memory small for the agent's pairwise scoring), L2-normalised
Differences from the paper, on purpose: final-layer patch tokens of the small ViT instead of
an intermediate-layer facet of ViT-G (speed on a laptop GPU), and the PCA step.

The class is a drop-in SPTM "scorer": embed / prob / locomote, plus ``sim`` (raw cosine) and
``sim_scale`` for the margin check (fix 1A). ``prob`` maps cosine to 0..1 with a calibration
taken from the exploration drives themselves (consecutive frames = same place, frames far
apart = usually different), so the agent's thresholds keep their meaning. ``locomote`` is
delegated to the existing scorer (the L network), which AnyLoc does not replace.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from vlad import load_frames

SIZE = (224, 294)  # (h, w), multiples of the 14-px ViT patch: 16 x 21 = 336 patches
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class AnyLocScorer:
    def __init__(self, base, device: str | None = None, model: str = "dinov2_vits14",
                 k: int = 32, dims: int = 256, cache_dir: Path = Path("cache")):
        self.base = base  # keeps L (locomote) from the trained scorer
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name, self.k, self.dims, self.cache_dir = model, k, dims, cache_dir
        self.net = torch.hub.load("facebookresearch/dinov2", model, trust_repo=True).eval().to(self.dev)
        self.centres = self.pca_mean = self.pca_comp = None
        self.c_lo, self.c_hi = 0.0, 1.0
        print(f"AnyLoc scorer: {model} on {self.dev}, K={k}, PCA {dims}")

    # ---- features

    @torch.no_grad()
    def _patches(self, images: list[np.ndarray]) -> np.ndarray:
        """(B, P, C) L2-normalised DINOv2 patch tokens for BGR uint8 images."""
        h, w = SIZE
        x = np.stack([cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1] for im in images])
        x = (x.astype(np.float32) / 255.0 - _MEAN) / _STD
        t = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.dev)
        tok = self.net.forward_features(t)["x_norm_patchtokens"]
        tok = torch.nn.functional.normalize(tok, dim=-1)
        return tok.float().cpu().numpy()

    def _vlad(self, pf: np.ndarray) -> np.ndarray:
        """VLAD of one image's (P, C) patch features: residuals per centre, intra-normalised."""
        assign = np.argmax(pf @ self.centres.T, axis=1)  # cosine: features and centres unit-norm
        v = np.zeros_like(self.centres)
        for c in range(self.k):
            sel = pf[assign == c]
            if len(sel):
                v[c] = (sel - self.centres[c]).sum(0)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9  # intra-normalisation
        v = v.reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)

    def _raw(self, images: list[np.ndarray], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch):
            out += [self._vlad(pf) for pf in self._patches(images[i:i + batch])]
        return np.asarray(out, np.float32)

    def _project(self, raw: np.ndarray) -> np.ndarray:
        z = ((raw - self.pca_mean) @ self.pca_comp.T).astype(np.float32)  # PCA projection
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

    # ---- fitting on the exploration data (no labels)

    def fit(self, data_dir: Path) -> None:
        frames, bounds = load_frames(Path(data_dir))
        key = f"anyloc_{Path(data_dir).name}_{self.model_name}_k{self.k}_d{self.dims}_{len(frames)}.npz"
        cache = self.cache_dir / key
        if cache.exists():
            z = np.load(cache)
            self.centres, self.pca_comp, self.pca_mean = z["centres"], z["components"], z["mean"]
            self.c_lo, self.c_hi = float(z["c_lo"]), float(z["c_hi"])
            print(f"AnyLoc: loaded vocabulary + calibration from {cache}")
            return
        imgs = [cv2.imread(str(f.path)) for f in frames]
        rng = np.random.default_rng(0)
        print(f"AnyLoc: building a {self.k}-word vocabulary from {len(imgs)} exploration frames ...")
        sample = []
        for i in range(0, len(imgs), 32):  # pass 1: a random 40 patches per frame for k-means
            for pf in self._patches(imgs[i:i + 32]):
                sample.append(pf[rng.choice(len(pf), 40, replace=False)])
        km = MiniBatchKMeans(self.k, random_state=0, n_init=3, batch_size=4096).fit(np.vstack(sample))
        c = km.cluster_centers_.astype(np.float32)
        self.centres = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
        raw = self._raw(imgs)  # pass 2: VLAD for every frame
        pca = PCA(self.dims, random_state=0).fit(raw)
        self.pca_comp, self.pca_mean = pca.components_.astype(np.float32), pca.mean_.astype(np.float32)
        db = self._project(raw)
        # calibration from the drives' own timeline: neighbours in a drive = same place,
        # frames >= 60 apart in a drive = usually a different place
        traj = np.concatenate([[t] * (e - s) for t, (s, e) in enumerate(bounds)])
        same = [float(db[i] @ db[i + 1]) for i in range(len(db) - 1) if traj[i] == traj[i + 1]]
        a = rng.integers(0, len(db), 20000)
        b = rng.integers(0, len(db), 20000)
        far = (traj[a] == traj[b]) & (np.abs(a - b) >= 60)
        diff = np.sum(db[a[far]] * db[b[far]], axis=1)
        self.c_hi = float(np.percentile(same, 25))  # most same-place pairs score at least this
        self.c_lo = float(np.percentile(diff, 95))  # few different-place pairs score above this
        if self.c_hi <= self.c_lo + 0.05:  # overlap: fall back to the medians
            self.c_lo, self.c_hi = float(np.median(diff)), float(np.median(same))
        print(f"AnyLoc calibration: different places ~ {self.c_lo:.3f}, same place ~ {self.c_hi:.3f}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache, centres=self.centres, components=self.pca_comp, mean=self.pca_mean,
                 c_lo=self.c_lo, c_hi=self.c_hi)

    # ---- the SPTM scorer interface

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        return self._project(self._raw(list(images)))

    def sim(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Raw cosine similarity of row pairs (unsaturated: used by the margin check)."""
        return np.sum(a * b, axis=1)

    @property
    def sim_scale(self) -> float:
        return max(self.c_hi - self.c_lo, 1e-3)

    def prob(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine mapped to 0..1: 0 at a typical different place, 1 at a typical same place."""
        return np.clip((self.sim(a, b) - self.c_lo) / self.sim_scale, 0.0, 1.0)

    def locomote(self, cur: np.ndarray, wp: np.ndarray, node: int = -1) -> str:
        return self.base.locomote(cur, wp, node)

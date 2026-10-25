"""Global image descriptors behind one interface.

    desc.database(frames) -> (N, d) unit vectors, cached on disk
    desc.extract(image)   -> (d,) unit vector

``vlad``  : the course baseline (RootSIFT + VLAD), no GPU needed.
``dino``  : DINOv2 patch features: CLS + GeM + a coarse grid so left/right layout survives.
            Needs torch + transformers (``uv pip install transformers``). On the cluster,
            compute the database once on a GPU; the cache file then travels with the data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from vlad import Frame, VLADExtractor


def _key(frames: list[Frame]) -> str:
    h = hashlib.md5("|".join(f"{f.trajectory}/{f.path.name}" for f in frames).encode())
    return f"{len(frames)}_{h.hexdigest()[:8]}"


class VladDescriptor:
    name = "vlad"

    def __init__(self, n_clusters: int = 128, cache_dir: Path = Path("cache")):
        self.ex = VLADExtractor(n_clusters, cache_dir)

    def database(self, frames: list[Frame]) -> np.ndarray:
        return self.ex.fit(frames)

    def extract(self, image: np.ndarray) -> np.ndarray:
        return self.ex.extract(image)


class DinoDescriptor:
    def __init__(
        self,
        model: str = "facebook/dinov2-small",
        cache_dir: Path = Path("cache"),
        device: str | None = None,
        grid: tuple[int, int] = (2, 4),
        size: tuple[int, int] = (224, 294),  # h, w, both multiples of 14
        batch: int = 64,
        net=None,
        head: Path | None = None,
    ):
        import torch

        self.torch = torch
        self.name = "dino_" + model.split("/")[-1]
        self.cache_dir, self.grid, self.size, self.batch = cache_dir, grid, size, batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if net is None:
            from transformers import AutoModel

            net = AutoModel.from_pretrained(model)
        self.net = net.eval().to(self.device)
        self.head = None
        if head is not None:  # projection learned by source/train/train_head.py
            self.head = torch.from_numpy(np.load(head)).to(self.device)
            self.name += "_ft"
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _prep(self, images: list[np.ndarray]):
        h, w = self.size
        x = np.stack([cv2.resize(i, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1] for i in images])
        x = self.torch.from_numpy(x.copy()).to(self.device).permute(0, 3, 1, 2).float() / 255
        return (x - self.mean) / self.std

    @staticmethod
    def _l2(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            tok = self.net(pixel_values=self._prep(images)).last_hidden_state
            cls, patches = tok[:, 0], tok[:, 1:]
            gh, gw = self.size[0] // 14, self.size[1] // 14
            fmap = patches.reshape(len(images), gh, gw, -1).permute(0, 3, 1, 2)
            gem = fmap.clamp(min=1e-6).pow(3).mean((2, 3)).pow(1 / 3)
            cells = torch.nn.functional.adaptive_avg_pool2d(fmap, self.grid).flatten(2)
            parts = [self._l2(cls), self._l2(gem)] + [self._l2(cells[:, :, k]) for k in range(cells.shape[2])]
            out = self._l2(torch.cat(parts, dim=1))
            if self.head is not None:
                out = self._l2(out @ self.head)
        return out.cpu().numpy().astype(np.float32)

    def database(self, frames: list[Frame]) -> np.ndarray:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        f = self.cache_dir / f"{self.name}_{_key(frames)}.npy"
        if f.exists():
            return np.load(f)
        out = []
        for i in tqdm(range(0, len(frames), self.batch), desc=self.name):
            imgs = [cv2.imread(str(fr.path)) for fr in frames[i : i + self.batch]]
            out.append(self.embed(imgs))
        db = np.concatenate(out)
        np.save(f, db)
        return db

    def extract(self, image: np.ndarray) -> np.ndarray:
        return self.embed([image])[0]


class ConcatDescriptor:
    """Concatenate several descriptors (each already unit-norm) and renormalise: cosine
    similarity becomes the average of the parts' similarities."""

    def __init__(self, parts: list):
        self.parts = parts
        self.name = "+".join(p.name for p in parts)

    def database(self, frames):
        return self._norm(np.concatenate([p.database(frames) for p in self.parts], axis=1))

    def extract(self, image):
        return self._norm(np.concatenate([p.extract(image) for p in self.parts]))

    @staticmethod
    def _norm(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def make(kind: str, cache_dir: Path, n_clusters: int = 128, head: Path | None = None):
    if kind == "vlad":
        return VladDescriptor(n_clusters, cache_dir)
    if kind == "dino":
        return DinoDescriptor(cache_dir=cache_dir, head=head)
    if kind == "both":
        return ConcatDescriptor([VladDescriptor(n_clusters, cache_dir), DinoDescriptor(cache_dir=cache_dir, head=head)])
    raise ValueError(f"unknown descriptor {kind!r} (vlad | dino | both)")

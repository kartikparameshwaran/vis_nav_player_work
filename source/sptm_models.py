"""
SPTM networks (PyTorch). Needs torch + torchvision; the agent file does not.

R  (retrieval): siamese ResNet-18. Each frame -> 512-d embedding once (so the whole
                exploration database is embedded once), then a small MLP on the pair
                (e_a * e_b, |e_a - e_b|) -> P(same place). Symmetric in a and b.
L  (locomotion): ResNet-18 on the two frames stacked as 6 channels -> FORWARD / LEFT / RIGHT.
Frames are resized to 160x120 (the paper's size) and fed as RGB, ImageNet-normalised.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

SIZE = (160, 120)  # (w, h)
ACTIONS = ("FORWARD", "LEFT", "RIGHT")
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prep(images: list[np.ndarray]) -> np.ndarray:
    """BGR uint8 frames of any size -> (B, 120, 160, 3) uint8 BGR."""
    return np.stack([cv2.resize(im, SIZE, interpolation=cv2.INTER_AREA) for im in images])


def to_tensor(arr: np.ndarray, device) -> torch.Tensor:
    """(B, 120, 160, 3) uint8 BGR -> normalised (B, 3, 120, 160) RGB float."""
    x = torch.from_numpy(np.ascontiguousarray(arr[..., ::-1])).to(device).permute(0, 3, 1, 2).float() / 255.0
    return (x - _MEAN.to(device)) / _STD.to(device)


def _resnet18(in_ch: int, pretrained: bool) -> nn.Module:
    net = None
    if pretrained:
        try:
            net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:  # no internet on a compute node, say
            print(f"warning: ImageNet weights unavailable ({exc!r}); training from scratch")
    if net is None:
        net = torchvision.models.resnet18(weights=None)
    if in_ch != 3:  # two frames stacked: copy the RGB filters for each, halved
        old = net.conv1
        conv = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv.weight.copy_(old.weight.repeat(1, in_ch // 3, 1, 1) / (in_ch // 3))
        net.conv1 = conv
    net.fc = nn.Identity()
    return net


class RNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.enc = _resnet18(3, pretrained)
        self.head = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def pair_logit(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([ea * eb, (ea - eb).abs()], dim=1)).squeeze(1)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        return self.pair_logit(self.embed(xa), self.embed(xb))


class LNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.enc = _resnet18(6, pretrained)
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, len(ACTIONS)))

    def forward(self, cur: torch.Tensor, wp: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(torch.cat([cur, wp], dim=1)))


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchScorer:
    """What SptmAgent talks to: embed(frames), prob(emb_a, emb_b), locomote(cur, wp, node)."""

    def __init__(self, r_path: Path, l_path: Path, device: str | None = None):
        self.device = pick_device(device)
        self.r = RNet(pretrained=False)
        self.l = LNet(pretrained=False)
        self.r.load_state_dict(torch.load(r_path, map_location="cpu")["state"])
        self.l.load_state_dict(torch.load(l_path, map_location="cpu")["state"])
        self.r.to(self.device).eval()
        self.l.to(self.device).eval()
        print(f"SPTM scorer on {self.device}")

    @torch.no_grad()
    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(images), 128):
            x = to_tensor(prep(images[i : i + 128]), self.device)
            out.append(self.r.embed(x).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def prob(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(a), 8192):
            ea = torch.from_numpy(np.ascontiguousarray(a[i : i + 8192])).to(self.device)
            eb = torch.from_numpy(np.ascontiguousarray(b[i : i + 8192])).to(self.device)
            out.append(torch.sigmoid(self.r.pair_logit(ea, eb)).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def locomote(self, cur: np.ndarray, wp: np.ndarray, node: int = -1) -> str:
        x = to_tensor(prep([cur, wp]), self.device)
        logits = self.l(x[0:1], x[1:2])
        return ACTIONS[int(logits.argmax(1))]

    @torch.no_grad()
    def locomote_probs(self, cur: np.ndarray, wp: np.ndarray, node: int = -1) -> np.ndarray:
        """Change 38: L's probabilities for ACTIONS (FORWARD, LEFT, RIGHT), not only the winner, so
        the agent can tell 'turn left' from 'left and right about equal: the waypoint is ahead'."""
        x = to_tensor(prep([cur, wp]), self.device)
        return torch.softmax(self.l(x[0:1], x[1:2]), 1)[0].float().cpu().numpy()

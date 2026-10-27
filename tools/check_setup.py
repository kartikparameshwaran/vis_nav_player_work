"""Check that everything the agent needs is in place -- run this first on a new machine.

    uv run tools/check_setup.py

Checks PyTorch and the GPU, the two trained networks in models/, SuperGlue in third_party/,
the local simulator, and .env (the API key is never printed). Each missing piece comes with the
command that fixes it. Exits with 1 if the agent could not start. Changes nothing.
"""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

TORCH_FIX = (
    "uv pip install torch==2.6.0 torchvision==0.21.0 "
    "--index-url https://download.pytorch.org/whl/cu124   (no NVIDIA GPU: .../whl/cpu)"
)
MODELS_FIX = "download sptm_l.pt and sptm_r.pt from the repo's Releases page (models-v1) to models/"
SUPERGLUE_FIX = (
    "git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git third_party/SuperGlue"
)

missing = 0


def report(ok: bool | None, what: str, fix: str = "") -> None:
    """ok: True = fine, False = the agent cannot start without it, None = a warning only."""
    global missing
    tag = {True: " OK ", False: "MISS", None: "WARN"}[ok]
    print(f"[{tag}] {what}" + (f"\n         fix: {fix}" if fix and not ok else ""))
    missing += ok is False


# PyTorch: needed by the networks, AnyLoc and SuperGlue; not installed by `uv sync` (see README)
try:
    import torch
    import torchvision

    report(True, f"PyTorch {torch.__version__}, torchvision {torchvision.__version__}")
    if torch.cuda.is_available():
        report(True, f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        report(None, "no CUDA GPU: the agent runs on the CPU, several times slower per step")
    if (Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main").is_dir():
        report(True, "DINOv2 (AnyLoc) is downloaded")
    else:
        report(None, "DINOv2 is not downloaded yet: the first run fetches it (needs internet)")
except ImportError:
    report(False, "PyTorch is not installed", TORCH_FIX)

for name in ("sptm_l.pt", "sptm_r.pt"):
    path = ROOT / "models" / name
    if path.is_file():
        report(True, f"models/{name} ({path.stat().st_size / 1e6:.0f} MB)")
    else:
        report(False, f"models/{name} is missing", MODELS_FIX)

sg = ROOT / "third_party" / "SuperGlue" / "models"
sg_files = ["superglue.py", "superpoint.py", "weights/superglue_indoor.pth"]
lost = [f for f in sg_files + ["weights/superpoint_v1.pth"] if not (sg / f).is_file()]
detail = f" (missing: {', '.join(lost)})" if lost else ""
report(not lost, "SuperGlue in third_party/SuperGlue" + detail, SUPERGLUE_FIX)

if importlib.util.find_spec("vis_nav_sim") is None:
    report(None, "the local simulator is not installed: --local runs need it (run `uv sync`)")
else:
    report(True, "local simulator (for --local runs)")

import cli  # noqa: E402  (after sys.path)
import vis_nav_sdk  # noqa: E402

report(True, f"vis-nav-sdk {vis_nav_sdk.__version__}")
cli.load_dotenv()
if not (ROOT / ".env").is_file():
    report(None, ".env is missing: server runs need it (copy .env.example to .env, fill it in)")
else:
    key = bool(os.environ.get("VIS_NAV_API_KEY"))
    report(key or None, "API key is set" if key else "API key is empty in .env (for server runs)")
    cid = os.environ.get("VIS_NAV_CHALLENGE")
    report(bool(cid) or None, f"challenge: {cid}" if cid else "VIS_NAV_CHALLENGE is empty in .env")

print("\nReady: the agent can start." if not missing else f"\n{missing} missing: fix them first.")
sys.exit(1 if missing else 0)

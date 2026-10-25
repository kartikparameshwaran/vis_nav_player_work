#!/bin/bash
# Run ONCE on a LOGIN node (compute nodes usually have no internet).
# Installs the project into $SCRATCH, and pre-downloads everything that needs the network:
# the SDK + simulator wheels, the texture pack (123 MB), and DINOv2 weights.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$SCRATCH/vis_nav_player}"
cd "$REPO_DIR"

export UV_CACHE_DIR="$SCRATCH/.uv-cache"
export HF_HOME="$SCRATCH/.hf"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync                          # SDK + local simulator (course package index)
uv pip install torch transformers --extra-index-url https://download.pytorch.org/whl/cu128
# first local run downloads the texture pack and records exploration data for maze 7
uv run --no-sync python - <<'PY'
import sys; sys.path.insert(0, "source")
import cli
print(cli.record_exploration_data(7))
PY
# DINOv2 weights into HF_HOME so compute nodes can run with HF_HUB_OFFLINE=1
uv run --no-sync python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-small'); print('dinov2-small cached')"
echo "done. Next: sbatch hpc/eval_array.sbatch"

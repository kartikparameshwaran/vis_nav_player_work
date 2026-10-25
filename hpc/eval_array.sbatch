#!/bin/bash
# One local maze per array task; CPU only. Usage:
#   DESC=vlad sbatch hpc/eval_array.sbatch
#   DESC=dino sbatch --gres=gpu:1 hpc/eval_array.sbatch      # GPU only speeds up DINO
# Then: uv run source/summarize.py results/*.csv
#SBATCH --job-name=visnav-eval
#SBATCH --array=7-26%10
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%A_%a.out
##SBATCH --account=YOUR_ACCOUNT        # uncomment / set if your cluster needs it
##SBATCH --partition=YOUR_PARTITION
set -euo pipefail
cd "${REPO_DIR:-$SCRATCH/vis_nav_player}"
mkdir -p logs results
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SCRATCH/.uv-cache"
export HF_HOME="$SCRATCH/.hf" HF_HUB_OFFLINE=1 SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
DESC="${DESC:-vlad}"
uv run --no-sync source/eval_local.py --seeds "$SLURM_ARRAY_TASK_ID" --descriptor "$DESC" \
    ${EXTRA:-} --out "results/${TAG:-$DESC}_${SLURM_ARRAY_TASK_ID}.csv"

# Running on NYU HPC

I could not read NYU's current HPC docs from where this was written, so partition / account /
module names are left for you to fill in; check `sinfo`, `sacctmgr show assoc user=$USER format=account%30,partition`,
and the NYU HPC docs (a new cluster, "Torch", was announced on 2026-10-19; Greene may be
changing, so follow whatever the docs say for students today).

What the cluster is worth here, in order:
1. **Evaluation sweeps** (`eval_array.sbatch`): 20 mazes in parallel, so every idea gets a
   mean-points number in ~15 minutes instead of an evening on a laptop. This is the main use.
2. **Descriptor extraction** with DINOv2 on a GPU (seconds instead of minutes); the cache file
   travels with the data.
3. **Fine-tuning a descriptor** (`source/train/`), only if the sweep shows retrieval is the
   bottleneck. See PLAN.md.

Order of operations
    git clone <your fork> $SCRATCH/vis_nav_player
    bash hpc/setup_login_node.sh            # login node: needs internet
    DESC=vlad sbatch hpc/eval_array.sbatch  # baseline number first
    DESC=dino sbatch --gres=gpu:1 hpc/eval_array.sbatch
    uv run source/summarize.py results/*.csv

Do not run heavy work on the login node. Do not put the API key in the repo: server attempts
are run from your laptop, not from the cluster.

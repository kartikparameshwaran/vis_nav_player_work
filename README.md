# Robot Perception Challenge — SPTM navigation agent

Our agent for the **Visual Navigation Challenge** of NYU **ROB-GY 6203 Robot Perception**
(run by the [AI4CE Lab](https://ai4ce.github.io/)), built on the course's
[starter kit](https://github.com/ai4ce/vis_nav_player).

A robot sits in a maze on the course server. Before the run we get about 3,000 photos from
earlier drives through the same maze (each labelled with the move made after it); when the
run starts we get four photos taken at the goal. The agent picks every move from the camera
frame alone and checks in when it believes it has arrived. The run is scored on distance,
heading and path efficiency. The course site has the full rules and a tutorial:
<https://visual-navigation-challenge.ai4ce.dev/>.

## Results so far

| run | result |
|---|---|
| Development Challenge, version 1 (`my_agent_3073.py`, dry run) | **6 / 10**: 0.049 m from the goal, 0.2° off, SPL 0.25, 3,073 of 10,000 steps |
| Development Challenge, version 2 (`my_agent_2767.py`) | **6 / 10**: 0.035 m from the goal, 0.5° off, SPL 0.27, 2,767 of 10,000 steps |
| Development Challenge, version 3 (`my_agent_1872.py`) | **6 / 10**: 0.009 m from the goal, 1.6° off, SPL 0.41, 1,872 of 10,000 steps |
| Development Challenge, version 4 (`my_agent_1820.py`) | **6 / 10**: SPL 0.42, 1,820 of 10,000 steps (one run; version 3 alone later gave 2,247 and one stuck run) |
| local maze 8, version 2 / version 3 | 6 / 10 in 2,618 / 5,406 steps (one run each: version 3 is not yet proven better) |
| local maze 8 | 6 / 10 (4,595 steps) |
| local maze 52 | 6 / 10 (8,581 steps) |
| local maze 7 | 0 (an earlier run without `--smooth-steer` scored 6 / 10) |

Git tag `dryrun-6pts` marks the exact code of the Development Challenge run.

## Setup (once per machine)

Tested on Windows 11 with an NVIDIA RTX 4060 laptop GPU. Linux works the same way; without
an NVIDIA GPU everything still runs, several times slower. On Windows use **PowerShell**, not WSL.

1. **Install uv** (Python and all packages, nothing global):
   ```powershell
   irm https://astral.sh/uv/install.ps1 | iex              # Windows
   curl -LsSf https://astral.sh/uv/install.sh | sh         # macOS, Linux
   ```
2. **Clone and install** (the repo is private: ask the owner for access first):
   ```powershell
   git clone https://github.com/kartikparameshwaran/Robot-Perception-Challenge.git
   cd Robot-Perception-Challenge
   uv sync
   ```
3. **Install PyTorch.** It is not part of `uv sync`, so install it into the same environment:
   ```powershell
   uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
   # no NVIDIA GPU: the same with --index-url https://download.pytorch.org/whl/cpu
   ```
   ⚠ A later plain `uv sync` removes PyTorch again (see [Troubleshooting](#troubleshooting)).
4. **Get SuperGlue** (Magic Leap's code and weights; not redistributed here):
   ```powershell
   git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git third_party/SuperGlue
   ```
5. **Get the two trained networks** (`sptm_l.pt`, `sptm_r.pt`, about 45 MB each; too large
   for the repo itself). Download both from this repo's **Releases** page (release `models-v1`)
   into a `models/` folder, or with the GitHub CLI:
   ```powershell
   gh release download models-v1 --repo kartikparameshwaran/Robot-Perception-Challenge --dir models
   ```
   To train them yourself instead, see [Retraining the networks](#retraining-the-networks).
6. **Credentials** (only for runs on the server): copy `.env.example` to `.env` and fill in
   your API key (course site, account menu) and the challenge id (the last part of the
   challenge page's URL). git ignores `.env`; never commit or share the key.
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```
7. **Check everything** — it lists anything missing with the command that fixes it:
   ```powershell
   uv run tools/check_setup.py
   ```

## Running

On Windows, run this once per terminal first; it stops a crash when the result is printed:
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

**On your own machine (free, unlimited):** the course simulator runs locally; a seed is a
maze, the same on every machine. The first run downloads the textures (123 MB) and records
that maze's exploration photos into `data/local-<seed>/`.
```powershell
uv run source/my_agent.py --local 8
```

**On the server (spends an attempt):** with the challenge id in `.env`,
```powershell
uv run source/my_agent.py
```
What happens: the exploration data downloads into `data/<challenge id>/`, the agent builds its
map (1–2 minutes on a new maze), checks itself on random frames, and then asks
`Start attempt N of M on <challenge>?`. Nothing is spent before that question.

Rules that protect your attempts:
- **Answer `y` only when you mean it.** Pressing Enter also means yes.
- **Never pipe the output** (`| Tee-Object`, `| tee`) and never pass `--yes` on the server:
  either one skips the question and starts the attempt.
- **Keep the viewer window open** until the result box appears. Closing it or Ctrl-C ends the
  run unscored, and the attempt still counts.
- A session lasts **one hour**. Use a stable connection; after a drop the SDK reconnects by
  itself for up to 5 minutes.
- Every run's decisions and result are saved to `logs/my_agent/sptm_<time>.json`.

`my_agent.py` is the course's entry point. It runs the agent in `source/sptm_agent.py`
unchanged, with the flags of its best runs (`--smooth-steer --verbose --log-dir logs/my_agent`).
All other options: `uv run source/sptm_agent.py --help`.

Two named versions, run the same way (`uv run source/my_agent_3073.py` or `uv run source/my_agent_2767.py`):
- `my_agent_3073.py`: version 1, exactly what `my_agent.py` runs (3,073 steps on the Development Challenge).
- `my_agent_2767.py`: version 2, which adds `--steer-hold` (measured steering keeps control down to 30
  inliers once it has it, so L no longer undoes its turns) and `--detour-once` (stalls cannot stack
  detours on one stretch of route): 2,767 steps on the Development Challenge. Logs: `logs/my_agent_2767/`.
- `my_agent_1872.py`: version 3, version 2 planning on a map in metres (`--map-ticks`,
  `source/metric_map.py`): before connecting it places every exploration photo on a map (a grid
  compass from vanishing points, SuperGlue rotations, loop closures), links photos the map and
  SuperGlue agree are one place, and costs every route edge in ticks; a link that leads into a bump
  is dropped at once. 1,872 steps on the Development Challenge. Logs: `logs/my_agent_1872/`.
- `my_agent_1820.py`: version 4, version 3 that thinks when confused (`--think-confused`): after 3
  decisions at the same photo it checks its view with SuperGlue against the next 5 photos of its route
  and believes one that clearly matches better (only forward along the route, never within 8 hops of
  the goal). It costs time, not steps. 1,820 steps on the Development Challenge. Logs: `logs/my_agent_1820/`.

The starter kit's own tools still work: `uv run source/keyboard_agent.py --local 7` (drive by
hand: arrows move, space checks in, esc quits) and `uv run source/baseline_agent.py --local 7`.

## How the agent works

It is **SPTM** (Semi-Parametric Topological Memory, Savinov et al., ICLR 2018), with geometric
checks wherever look-alike walls could fool it. In order:

1. **Map, before connecting.** Every exploration photo is a node; consecutive photos are joined
   by the move between them; the drives are joined at the spawn, where they all start.
   Shortcuts between drives are proposed by **AnyLoc** (DINOv2 features + VLAD), kept only if
   they lie on a run of ≥ 8 matching pairs (SeqSLAM-style) and pass **SuperPoint + SuperGlue**
   matching with a RANSAC essential-matrix check.
2. **Goal, at session start.** AnyLoc proposes map photos for each goal photo; SuperGlue picks
   the best. Dijkstra gives the route to the goal from every node.
3. **Every step.** Localise with AnyLoc near the last node (far jumps need SuperGlue proof);
   pick the furthest route photo still in view as the waypoint; steer by the rotation measured
   against it (essential matrix), or by the trained **L network** (smoothed over time) when the
   match is weak. Look around if the waypoint is lost; back off when blocked; re-plan when stuck.
4. **Last metre.** Match the view against the four goal photos: heading from how far the scene
   is shifted, distance from how big it looks; line up front and sideways, then check in.

Details, with the reason behind every setting, are in the comments of `source/sptm_agent.py`.

## What is in the repository

**Needed to run the agent** (do not delete or rename):

| file | role |
|---|---|
| `source/my_agent.py` | entry point: class `MyAgent`, the SPTM agent with its best flags |
| `source/my_agent_3073.py`, `source/my_agent_2767.py`, `source/my_agent_1872.py`, `source/my_agent_1820.py` | version 1 (= `my_agent.py`, 3,073 steps), version 2 (+ `--steer-hold --detour-once`, 2,767 steps), version 3 (+ `--map-ticks`, 1,872 steps), version 4 (+ `--think-confused`, 1,820 steps) |
| `source/metric_map.py` | the map in metres for version 3: grid compass, step rotations, loop closures, verified links |
| `source/sptm_agent.py` | the agent: map, localisation, planning, steering, last metre |
| `source/sptm_models.py` | the trained networks (R: same place?, L: which move?) |
| `source/anyloc.py` | AnyLoc place recognition (DINOv2 + VLAD) |
| `source/sg_matcher.py` | SuperPoint + SuperGlue matching |
| `source/geometry.py` | essential matrix, rotation, bearing, scale |
| `source/bayes_loc.py`, `source/bayes_agent.py`, `source/navigator.py`, `source/features.py`, `source/vlad.py` | shared helpers (the earlier Bayes-filter agent's code is imported by the SPTM agent) |
| `source/cli.py` | command line, `.env`, `--local` |
| `pyproject.toml`, `uv.lock`, `.python-version` | the Python environment |

**Not in git, but needed** (see Setup): `models/sptm_l.pt`, `models/sptm_r.pt`,
`third_party/SuperGlue/`, `.env`. Created automatically: `data/`, `cache/`, `logs/`.

**Tools for developing** (not used by the agent):

| file | what |
|---|---|
| `tools/check_setup.py` | checks a new machine has everything |
| `tools/run_check.py` | runs the agent on a local maze and compares the run with the true positions |
| `tools/run_suite.py` | the same on several mazes, one table: is a change better everywhere? |
| `tools/top_view.py` | picture of a local run: the maze, the true path, the goal |
| `tools/odometry_check.py` | how accurately the map places every photo, against the true positions (`--plot` draws it) |
| `tools/route_check.py` | how long the planned routes are, with and without the map links, against the true shortest route; `--dev` for the Development maze |
| `source/sptm_train.py`, `source/train/` | retraining the networks |
| `source/keyboard_agent.py`, `source/baseline_agent.py` | the starter kit's manual driver and baseline |
| `source/eval_local.py`, `source/summarize.py`, `source/rl/`, `hpc/` | starter kit extras: batch evaluation, an RL demo, NYU HPC scripts |
| `tests/` | `uv run pytest -q` |
| `PLAN.md` | the early plan (historical) |

## Testing a change before spending an attempt

Locally the true robot position is known, so a run can be checked against it:
```powershell
uv run --no-sync tools/run_check.py --name test1 --maze 8 --agent-args=--smooth-steer
uv run --no-sync tools/run_suite.py --name test1 --mazes 7,8,52 --agent-args=--smooth-steer
```
`run_check` prints a summary (closest approach, how often the agent was lost, likely causes)
and draws `logs/<name>_topview.png`. `run_suite` runs several mazes with the same motion noise,
so two versions of the agent are compared fairly. Note the `=` in `--agent-args=...`.

## Retraining the networks

Training uses drives with true positions from the local simulator (many mazes; seeds that are
multiples of 5 are held out for validation). Needs PyTorch and preferably a GPU:
```powershell
uv run source/train/gen_data.py --mazes 100-139 --out data_train --ticks 3000
uv run source/sptm_train.py r --data data_train --out models/sptm_r.pt --iters 30000
uv run source/sptm_train.py l --data data_train --out models/sptm_l.pt --iters 30000
```

## Troubleshooting

| problem | fix |
|---|---|
| `No module named 'torch'` | PyTorch was removed by a plain `uv sync`, or by the SDK's own update (it runs `uv sync`). Install it again (Setup step 3). `uv run …` itself never removes it. |
| `UnicodeEncodeError` when the result is printed | `$env:PYTHONIOENCODING = "utf-8"` before running |
| `Failed to spawn: source/my_agent.py` | wrong folder: `cd` into the repository first |
| `--agent-args` error | write it with `=`: `--agent-args=--smooth-steer` |
| run ended "viewer window closed" | the window was closed; minimise it instead |
| "a newer SDK is out" | say yes (the course asks for it), then reinstall PyTorch (Setup step 3) |
| GitHub says the checks failed | the style check (`ruff`) only reports; the tests decide |

## Credits

Course starter kit, simulator and SDK: AI4CE Lab, NYU. SPTM: Savinov, Dosovitskiy, Koltun
(ICLR 2018). AnyLoc: Keetha et al. (RA-L 2023), on Meta's DINOv2. SuperPoint and SuperGlue:
Magic Leap (DeTone et al. 2018; Sarlin et al. 2020), under their own non-commercial licence,
which is why they are downloaded rather than included. SDK reference:
<https://visual-navigation-challenge.ai4ce.dev/sdk>.

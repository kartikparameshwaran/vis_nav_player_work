# Visual Navigation Challenge: plan for the Oct 28 midterm

Written Fri Oct 23, 00:30. Midterm challenge: **Wed Oct 28** (10% of grade, plus the 5% mid-term
report). You have 2-3 working days, so the plan is built around one rule:
**never spend a server attempt on something you have not measured on local mazes first.**

## What is being scored

Gate: distance <= 0.2 m **and** heading <= 20 deg at the moment you CHECKIN, otherwise 0.
Then distance (3) + heading (3) + SPL (4) = 10. SPL = shortest-route ticks / your ticks, so wandering
to localise costs points even when you arrive. Typical successful run ~1500 ticks; a missed
check-in, QUIT or running out of budget is also 0. Leaderboard: points, then SPL, then distance.

Check the challenge page for the real numbers (`max_steps`, attempts, this challenge's rubric);
the slides say "each challenge shows its own".

## What you are given / what is hidden

Given: current 320x240 frame, four goal photos (front/left/back/right, taken from the goal
position), intrinsics (f = 92 px), ~3000 exploration frames from 3 drives with the action taken after
each, steps left. Hidden: pose, map, goal position, collisions, how far you missed by.
So the task is **place recognition + planning on a topological map + a last-metre alignment**.

## The agent in this repo (`source/my_agent.py`)

1. **Calibrate turn direction** (12 ticks): LEFT x6, measure which way the picture shifts, RIGHT x6.
2. **Localise** with a Bayes filter over the exploration frames (`navigator.BeliefFilter`): last
   belief diffused along the map, times appearance likelihood. One aliased match cannot teleport you.
3. **Plan**: one Dijkstra from the goal frame over a *directed* graph (drive edges cost 1, driving
   them backwards 1.25, geometrically verified visual shortcuts ~1.5); next move chosen with
   hysteresis so ties between equal routes do not make the robot shuttle.
4. **Approach** when within a few hops of the goal: RANSAC-match the frame to the four goal
   photos. Sideways shift of the scene -> heading error; growth of the scene -> distance along
   the view. Turn, creep, then CHECKIN.
5. **Safety nets**: stuck detector (frame unchanged after FORWARD -> turn), no-progress watchdog,
   budget guard.

Descriptor is pluggable: `--descriptor vlad|dino|both` (`dino` needs `torch transformers`).

## Status when this was handed over (Fri Oct 23, ~02:00)

- Code written: agent, filter, planner, approach controller, batch evaluator, HPC scripts,
  optional descriptor-training stage. Lints clean; plumbing tests pass on a stand-in maze.
- **Never run against the real simulator or server.** On six stand-in mazes (random block
  textures, 1 m corridors) it reached the gate in 4/6 (points 4-6 when it did), with poor SPL
  (0.2-0.5): it wanders. These numbers say the loop works, not how it will score on the real
  thing. Bugs found and fixed on the stand-in, each of which will probably recur on the real maze:
  false visual shortcuts trapping the robot (now verified with the essential matrix, banned at run
  time if they never take), tie between equal routes making it shuttle (directed costs + hysteresis),
  open-loop action replay drifting into walls (visual steering on straight stretches).
- Known weak spot: lateral offset in the last metre (scene shift conflates sideways offset with
  heading). Expect to tune `approach` first on real local mazes.

## Schedule

| When | Do | Done when |
|---|---|---|
| **Fri (today)** | Install, drive maze 7 by hand, run `baseline_agent.py`, then `my_agent.py --local 7` and watch it. Run `eval_local.py --seeds 7-16` for the **baseline-quality number**. Fork the repo, push, run `hpc/setup_login_node.sh`. | You have a mean-points number over 10 mazes, on your machine and on HPC. |
| **Sat** | Failure analysis: for each failed maze, is it (a) never localised, (b) wrong place, (c) reached but missed the gate, (d) too slow? Fix the biggest bucket. Sweep `vlad` vs `dino` vs `both` on 20 mazes with the HPC array job. | One config beats the others by more than the noise (see below). |
| **Sun** | Tune approach (`near_hops`, scale/shift tolerances) against true arrival distance/heading. Only if retrieval is the bottleneck: generate data and train the head (`source/train/`). Practice challenge attempts (yours, not shared). | Gate rate >= 70% locally, no crashes over 20 mazes. |
| **Mon** | Freeze code. Dry-run on the server practice challenge. Write the mid-term report from the sweep tables. Check `.env`, API key, challenge id for the midterm. | Tagged commit you will submit with. |
| **Tue** | Buffer. Re-run the frozen code on 20 mazes once to confirm. Sleep. | |
| **Wed Oct 28** | Midterm run. Do not change code on the day. | |

Noise: with 20 mazes, one maze = 5% of the gate rate. Treat differences under ~10% as noise.

## HPC: what it is for

1. **Evaluation sweeps (main use).** `hpc/eval_array.sbatch`: one maze per array task. This is what
   lets you try an idea and see mean points in ~15 minutes.
2. **DINOv2 descriptor extraction** on a GPU, cached next to the data.
3. **Training** only if the sweep says place recognition is the bottleneck (wrong-place failures
   dominate). Then: `source/train/gen_data.py` (true-pose drives from the local simulator, many
   mazes) -> `train_head.py` (contrastive projection on frozen DINOv2; reports held-out
   recall@1 vs raw DINO so you can see whether it helps). If it does not beat raw DINO on
   held-out mazes, drop it: do not ship a model you cannot show helps.

Training a policy end to end with RL is **not** recommended in this window: the kit's own README
says the RL demo "learns to move, not to find the goal".

## Decision rules

- Change one thing at a time; keep `results/*.csv` for every config; compare with `summarize.py`.
- Only spend a real attempt when the local gate rate has stopped improving. Attempts spent on the
  *practice* challenge are yours; the competition one is shared by the team.
- If DINO does not beat VLAD by a visible margin locally, ship VLAD: fewer moving parts on the day,
  and DINO inference on a laptop CPU is slower per step.
- A CHECKIN with a bad pose scores 0, same as running out of budget; so near the end of the budget
  always check in rather than stop.

## Known unknowns (things I could not verify)

- I could not install the SDK or simulator where this was written (the course site is blocked there),
  so the agent was tested on a **stand-in maze simulator** (`tests/minisim.py`), not the real one.
  The plumbing works there; absolute scores mean nothing. The first thing to do on your machine is
  `uv run pytest` and then `eval_local.py` on real local mazes.
- `info.camera` attribute names, the maximum burst length (`max_repeat`), and whether "left"
  goal photo is a LEFT-turn from front: the code is defensive (falls back to f = 92 px, bursts <= 8,
  turn direction calibrated at run time), but check the first run's printed `calibration:` line.
- NYU HPC partition / account names (see `hpc/README.md`).
- The midterm challenge's own rubric and step budget.

## If you have a teammate

Split: one person owns localisation (descriptors, filter, shortcuts, training); the other owns the
last metre (approach controller, check-in rule) and the report. They only meet at `eval_local.py`.

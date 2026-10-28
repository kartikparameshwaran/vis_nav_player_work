"""Plan in metres: how long are the agent's routes, and how long with its --map-links planner
(Change 42: source/metric_map.py)? Checked against the simulator's true shortest route on local mazes.
No robot drives, nothing changes.

    uv run --no-sync tools/route_check.py                 # local mazes 7, 8, 21, 37, 52
    uv run --no-sync tools/route_check.py --mazes 8
    uv run --no-sync tools/route_check.py --dev           # the Development Challenge maze (no truth)

A route is a sequence of the agent's map photos. Its cost is what driving through those photos takes:
metres / 0.03 m per tick + degrees turned / 3.44 deg per tick -- with the TRUE poses on local mazes,
with the map's own poses on the dev maze. Three routes from the start to the goal photo:
  current agent    the planner without --map-links (cost per photo hop, shortcuts cost extra)
  costs in steps   the same links, each costed in ticks from the map
  + map links      the --map-links planner itself: plus the links the map proposes and SuperGlue
                   confirms, costed like verified shortcuts (agent.add_map_links, with label motions
                   learned without this maze)
  links + ticks    --map-ticks: the links, and every edge costed in ticks (shortest on paper, but on
                   maze 8 it drove 2x longer: backwards along drives and hopping between them)
Every new link is also checked against the maze's true walls: a link whose two photos have a wall
between them would send the robot into that wall. The goal photo on a local maze is the map photo
nearest the true goal (facing within 45 deg).
"""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

import metric_map as mm  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import odometry_check as oc  # noqa: E402

DEV = "515b23a3c14a"


def route_cost(route: list[int], pos: np.ndarray, head: np.ndarray) -> float:
    return sum(mm.step_cost(pos[a], head[a], pos[b], head[b]) for a, b in zip(route, route[1:]))


def crosses_wall(world, p, q, samples: int = 20) -> bool:
    """True if the straight line between two true positions passes through a wall cell of the maze.
    Cells are centred on multiples of the cell size: the spawn (0.2, 2.6) is cell (row 13, col 1)."""
    occ, cell = world.occupancy, world.cell_size
    for t in np.linspace(0, 1, samples):
        x, y = (1 - t) * np.asarray(p) + t * np.asarray(q)
        r, c = int(round(y / cell)), int(round(x / cell))
        if 0 <= r < occ.shape[0] and 0 <= c < occ.shape[1] and occ[r, c]:
            return True
    return False


def way_around(world, p, q) -> float:
    """Length (m) of the shortest path through free cells between two true positions (4-connected,
    cell centre to cell centre), or inf. Positions just past a cell edge snap to the nearest free cell."""
    from collections import deque

    occ, cell = world.occupancy, world.cell_size

    def free_cell(pt):
        r, c = int(round(pt[1] / cell)), int(round(pt[0] / cell))
        if not occ[r, c]:
            return r, c
        near = [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if 0 <= r + dr < occ.shape[0] and 0 <= c + dc < occ.shape[1] and not occ[r + dr, c + dc]]
        return min(near, key=lambda rc: math.hypot(rc[0] * cell - pt[1], rc[1] * cell - pt[0]), default=None)

    a, b = free_cell(p), free_cell(q)
    if a is None or b is None:
        return math.inf
    seen, todo = {a: 0}, deque([a])
    while todo:
        r, c = todo.popleft()
        if (r, c) == b:
            return seen[b] * cell
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < occ.shape[0] and 0 <= nc < occ.shape[1] and not occ[nr, nc] and (nr, nc) not in seen:
                seen[(nr, nc)] = seen[(r, c)] + 1
                todo.append((nr, nc))
    return math.inf


BEHIND = 0.4  # m: a link whose way around (through free cells) is this much longer than the straight
# line has a wall between its photos -- following it would send the robot into that wall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mazes", default="7,8,21,37,52", help="local mazes with saved truth")
    ap.add_argument("--dev", action="store_true", help=f"the Development Challenge maze (data/{DEV}, no truth)")
    ap.add_argument("--goal", type=int, default=3616, help="--dev: goal photo (where the dry runs checked in)")
    args = ap.parse_args()
    found = sorted(int(p.name[6:]) for p in (ROOT / "cache").glob("truth-*") if p.name[6:].isdigit())
    mazes = {m: oc.load(m) for m in found}  # label motions learn from every maze but the one checked
    for maze in [None] if args.dev else [int(m) for m in args.mazes.split(",")]:
        data = ROOT / "data" / (DEV if maze is None else f"local-{maze}")
        agent = oc.build_agent(data)
        before = agent.G.copy()  # the planner without --map-links
        agent.add_map_links(data, mm.LABEL_MOTION if maze is None else oc.leave_one_out(mazes, maze))
        pos, head = agent.map_pos, agent.map_head
        steps = before.copy()
        for a, b in steps.edges:
            steps.edges[a, b]["weight"] = mm.edge_cost(agent.frames, pos, head, a, b)
        tpos = thead = oracle = world = None
        if maze is None:
            goal = args.goal
        else:
            import vis_nav_sim as sim

            world = sim.Simulator(sim.Textures.find(), maze)
            oracle = world.oracle_steps
            where = mm.frame_index(agent.frames, mazes[maze])
            tpos = np.array([mazes[maze][d]["poses"][k, :2] for d, k in where])
            thead = np.array([mazes[maze][d]["poses"][k, 2] for d, k in where])
            tx, ty, tyaw = world.target
            far = np.hypot(tpos[:, 0] - tx, tpos[:, 1] - ty)
            far += np.where(np.abs(mm.wrap(thead - tyaw)) < math.radians(45), 0.0, 100.0)
            goal = int(np.argmin(far))
        start = 0  # drive 0's first map photo: the spawn
        ticks = agent.G.copy()  # --map-ticks: the links, and every edge costed in ticks
        for a, b in ticks.edges:
            ticks.edges[a, b]["weight"] = mm.edge_cost(agent.frames, pos, head, a, b)
        routes = {
            "current agent": nx.dijkstra_path(before, start, goal, weight="weight"),
            "costs in steps": nx.dijkstra_path(steps, start, goal, weight="weight"),
            "+ map links": nx.dijkstra_path(agent.G, start, goal, weight="weight"),
            "links + ticks": nx.dijkstra_path(ticks, start, goal, weight="weight"),
        }
        title = "Development Challenge maze" if maze is None else f"maze {maze}"
        print(f"\n=== {title}: {len(agent.frames)} map photos, start photo {start}, goal photo {goal}"
              + (f"; true shortest route (oracle): {oracle} ticks" if oracle else ""))
        print(f"new map links: {len(agent.map_links)}")
        if tpos is not None and agent.map_links:
            gap = np.array([np.hypot(*(tpos[a] - tpos[b])) for a, b in agent.map_links])
            walls = [(a, b) for a, b in agent.map_links if crosses_wall(world, tpos[a], tpos[b])]
            behind = [(a, b) for a, b in walls
                      if way_around(world, tpos[a], tpos[b]) > np.hypot(*(tpos[a] - tpos[b])) + BEHIND]
            used = set(zip(routes["+ map links"], routes["+ map links"][1:]))
            on_route = [e for e in behind if e in used or e[::-1] in used]
            print(f"   their true distance: median {np.median(gap):.2f} m, max {gap.max():.2f} m; straight line "
                  f"touches a wall cell: {len(walls)}; WALL BETWEEN THE PHOTOS (way around > {BEHIND} m "
                  f"longer): {len(behind)} {behind[:8]} (on the planned route: {len(on_route)})")
        base_est = route_cost(routes["current agent"], pos, head)  # the percentages compare with this
        base_true = route_cost(routes["current agent"], tpos, thead) if tpos is not None else None
        traj = [f.trajectory for f in agent.frames]
        for name, r in routes.items():
            est = route_cost(r, pos, head)
            same = [traj[a] == traj[b] and abs(a - b) == 1 for a, b in zip(r, r[1:])]
            back = sum(s and b == a - 1 for s, (a, b) in zip(same, zip(r, r[1:])))
            line = (f"  {name:15s} {len(r) - 1:4d} photo hops ({len(same) - sum(same)} drive switches, {back} "
                    f"driven backwards), map estimate {est:6.0f} ticks ({est / base_est - 1:+.0%})")
            if tpos is not None:
                true = route_cost(r, tpos, thead)
                line += f", true {true:6.0f} ticks ({true / base_true - 1:+.0%}) = {true / oracle:.1f}x the shortest"
            print(line)


if __name__ == "__main__":
    main()

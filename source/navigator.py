"""Topological map over the exploration frames, a Bayes filter that says which frame the robot
is at, and the next-hop table toward the goal. Pure numpy/scipy/networkx; no SDK, no GPU.

Differences from the course baseline:
  * localisation is a *filter*: the previous belief is diffused along the map, then multiplied
    by the appearance likelihood, so one aliased frame cannot teleport the estimate;
  * visual shortcut edges are *geometrically verified* (RANSAC inliers), a few per node,
    instead of the 30 globally most similar pairs;
  * the next move is a table lookup (one Dijkstra from the goal), not a search per step.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import scipy.sparse as sp
from geometry import Matcher, scale_ratio, shift_deg
from tqdm import tqdm
from vlad import REVERSE, Frame

FOCAL = 92.0
TEMPORAL_W = 1.0
REVERSE_W = 1.25  # driving a recorded edge backwards: same ticks, but we face away from the path
SHORTCUT_W = 1.5


class TopoMap:
    def __init__(
        self,
        frames: list[Frame],
        bounds: list[tuple[int, int]],
        db: np.ndarray,
        *,
        shortcuts_per_node: int = 3,
        min_gap: int = 40,
        min_inliers: int = 25,
        stride: int = 2,
        max_shift_deg: float = 10.0,
        max_scale_dev: float = 0.25,
        cache: Path | None = None,
        matcher: Matcher | None = None,
        verify: bool = True,
    ):
        self.frames, self.bounds, self.db = frames, bounds, db
        self.n = len(frames)
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(self.n))
        for s, e in bounds:
            for i in range(s, e - 1):
                self.graph.add_edge(i, i + 1, weight=TEMPORAL_W)
                self.graph.add_edge(i + 1, i, weight=REVERSE_W)
        self.shortcuts = self._shortcuts(
            shortcuts_per_node, min_gap, min_inliers, stride, cache, matcher or Matcher(), verify,
            max_shift_deg, max_scale_dev,
        )
        for i, j, w in self.shortcuts:
            self.graph.add_edge(i, j, weight=SHORTCUT_W + w, visual=True)
            self.graph.add_edge(j, i, weight=SHORTCUT_W + w, visual=True)

    # -- shortcuts
    def _shortcuts(self, per_node, gap, min_inl, stride, cache, matcher, verify, max_shift, max_scale):
        key = (self.n, per_node, gap, min_inl, stride, verify, max_shift, max_scale)
        if cache is not None and cache.exists():
            saved = pickle.loads(cache.read_bytes())
            if saved["key"] == key:
                return saved["edges"]
        traj = np.concatenate([[t] * (e - s) for t, (s, e) in enumerate(self.bounds)])
        feats: dict[int, object] = {}

        def feat(i):
            if i not in feats:
                feats[i] = matcher.features(cv2.imread(str(self.frames[i].path)))
            return feats[i]

        edges, seen = [], set()
        for i in tqdm(range(0, self.n, stride), desc="shortcuts"):
            sim = self.db @ self.db[i]
            idx = np.arange(self.n)
            far = (traj != traj[i]) | (np.abs(idx - i) > gap)
            sim[~far] = -2
            for j in np.argsort(-sim)[:per_node]:
                j = int(j)
                pair = (min(i, j), max(i, j))
                if pair in seen:
                    continue
                seen.add(pair)
                if verify:
                    m = matcher.match(feat(i), feat(j), focal=FOCAL)
                    # same place AND same heading: the shortcut lets us borrow j's next move,
                    # which is only right if we point the way j pointed
                    if (
                        m.n < min_inl
                        or abs(shift_deg(m, FOCAL)) > max_shift
                        or abs(scale_ratio(m) - 1) > max_scale
                    ):
                        continue
                edges.append((pair[0], pair[1], float(max(0.0, 1 - sim[j]))))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps({"key": key, "edges": edges}))
        return edges

    # -- planning
    def ban(self, i: int, j: int) -> None:
        """Forget a shortcut that turned out not to connect the same place."""
        for a, b in ((i, j), (j, i)):
            if self.graph.has_edge(a, b) and self.graph[a][b].get("visual"):
                self.graph.remove_edge(a, b)

    def first_jump(self, node: int, horizon: int = 4):
        """The first visual-shortcut edge (i, j) within ``horizon`` hops along the plan."""
        cur = node
        for _ in range(horizon):
            nxt = int(self.next_hop[cur])
            if nxt < 0:
                return None
            if self.graph[cur][nxt].get("visual"):
                return cur, nxt
            cur = nxt
        return None

    def plan_to(self, goal: int) -> None:
        """Cost and hops from every node to ``goal`` along directed edges (one Dijkstra and
        one BFS on the reversed graph), plus the next hop."""
        self.goal = goal
        rev = self.graph.reverse(copy=False)
        dist, paths = nx.single_source_dijkstra(rev, goal, weight="weight")
        self.dist = np.full(self.n, np.inf)
        self.next_hop = np.full(self.n, -1)
        for node, d in dist.items():
            self.dist[node] = d
            if len(paths[node]) > 1:
                self.next_hop[node] = paths[node][-2]
        self.hop_table = np.full(self.n, 10**6)
        for node, h in nx.single_source_shortest_path_length(rev, goal).items():
            self.hop_table[node] = h

    def hops(self, node: int) -> int:
        """Graph hops from ``node`` to the goal (unweighted)."""
        return int(self.hop_table[node])

    def edge_action(self, a: int, b: int) -> str | None:
        """Movement that takes the robot from frame a toward frame b; None for a shortcut."""
        if b == a + 1 and self.graph[a][b].get("visual") is None:
            return self.frames[a].action
        if b == a - 1 and self.graph[a][b].get("visual") is None:
            return REVERSE[self.frames[b].action]
        return None

    def action_costs(self, node: int) -> dict[str, float]:
        """Cost-to-go if the next move from ``node`` is each action. Across a shortcut the
        move is the one the far end would make (we are at the same place)."""
        out: dict[str, float] = {}
        for nxt, data in self.graph[node].items():
            cost = data["weight"] + self.dist[nxt]
            if not np.isfinite(cost):
                continue
            if data.get("visual"):
                n2 = int(self.next_hop[nxt])
                if n2 < 0 or self.graph[nxt][n2].get("visual"):
                    continue
                act = self.edge_action(nxt, n2)
            else:
                act = self.edge_action(node, nxt)
            if act is not None and cost < out.get(act, np.inf):
                out[act] = float(cost)
        return out

    def action_toward_goal(self, node: int, prefer: str | None = None, margin: float = 2.5) -> str:
        """Cheapest next move; keeps ``prefer`` (the move we were making) unless another is
        better by more than ``margin``. This is what stops a tie between two equally short
        routes from making the robot shuttle back and forth."""
        costs = self.action_costs(node)
        if not costs:
            return "FORWARD"
        best = min(costs, key=costs.get)
        if prefer in costs and costs[prefer] <= costs[best] + margin:
            return prefer
        return best


class BeliefFilter:
    def __init__(self, topo: TopoMap, tau: float = 1.0, sigma: float = 1.5, stay: float = 0.4, floor: float = 1e-3):
        n = topo.n
        rows, cols, vals = [], [], []
        reach = int(np.ceil(3 * sigma))
        for s, e in topo.bounds:
            for i in range(s, e):
                for d in range(-reach, reach + 1):
                    j = i + d
                    if s <= j < e:
                        w = stay if d == 0 else (1 - stay) * np.exp(-0.5 * (d / sigma) ** 2)
                        rows.append(i), cols.append(j), vals.append(w)
        for i, j, _ in topo.shortcuts:  # being at i makes j plausible, and vice versa
            for a, b in ((i, j), (j, i)):
                rows.append(a), cols.append(b), vals.append(0.3)
        k = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        k = sp.diags(1 / np.asarray(k.sum(1)).ravel()) @ k
        self.kt = k.T.tocsr()  # b_new = K^T b
        self.db, self.tau, self.floor, self.n = topo.db, tau, floor, n
        self.reset()

    def reset(self) -> None:
        self.b = np.full(self.n, 1.0 / self.n)

    def update(self, q: np.ndarray, moved: bool = True) -> int:
        if moved:
            self.b = self.kt @ self.b
        sims = self.db @ q
        # Raw VLAD similarities are tiny (best ~0.1, rest ~0.0 at 16k dims), so a fixed
        # temperature in raw units is meaningless: score in standard deviations above the crowd.
        z = (sims - sims.mean()) / (sims.std() + 1e-9)
        like = np.exp((z - z.max()) / self.tau) + self.floor
        self.b = self.b * like
        self.b /= self.b.sum()
        self.last_sim = float(sims.max())
        return int(np.argmax(self.b))

    def mass_near(self, node: int, radius: int, topo: TopoMap) -> float:
        """Belief mass on frames within ``radius`` temporal steps of ``node`` (same drive)."""
        for s, e in topo.bounds:
            if s <= node < e:
                return float(self.b[max(s, node - radius) : min(e, node + radius + 1)].sum())
        return 0.0

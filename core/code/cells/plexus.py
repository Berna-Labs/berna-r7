#!/usr/bin/env python3
"""
R5-compliant Plexus.

Dynamic graph P = (V, E, W):
  V = nodes (cells)
  E = edges (co-activation links)
  W : E -> R+ (edge weights)

Update rule (R5 Section 3.3, Eq. 1):
  W_uv(t+1) = W_uv(t) + eta_e * (A_uv(t) - lambda_e * W_uv(t))

Constants (R5):
  eta_e    = 0.01
  lambda_e = 0.001
  convergence condition: 0 < eta_e * lambda_e < 2

K4 (reasoning depth) will use the graph:
  effective path length from this node to reachable nodes.
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ETA_E = 0.01
LAMBDA_E = 0.001


@dataclass
class Plexus:
    nodes: List[str] = field(default_factory=list)
    W: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def has_node(self, u: str) -> bool:
        return u in self.W

    def add_node(self, u: str):
        if u not in self.W:
            self.W[u] = {}
            self.nodes.append(u)

    def edge_weight(self, u: str, v: str) -> float:
        return self.W.get(u, {}).get(v, 0.0)

    def set_edge(self, u: str, v: str, w: float):
        self.add_node(u)
        self.add_node(v)
        self.W[u][v] = float(w)


def update_edge(plexus: Plexus, u: str, v: str, A_uv: float,
                eta_e: float = ETA_E, lambda_e: float = LAMBDA_E):
    """
    Single edge update per R5 Eq. 1:
      W_uv(t+1) = W_uv(t) + eta_e * (A_uv(t) - lambda_e * W_uv(t))
    """
    w_old = plexus.edge_weight(u, v)
    w_new = w_old + eta_e * (A_uv - lambda_e * w_old)
    plexus.set_edge(u, v, w_new)


def update_from_coactivations(plexus: Plexus, A: Dict[Tuple[str, str], float],
                              eta_e: float = ETA_E, lambda_e: float = LAMBDA_E):
    """
    Apply update to all co-activations in A (dict: (u,v) -> A_uv).
    """
    for (u, v), a in A.items():
        update_edge(plexus, u, v, a, eta_e=eta_e, lambda_e=lambda_e)


def effective_path_length(plexus: Plexus, start: str,
                          max_depth: int = 20) -> float:
    """
    Weighted-BFS average depth from `start` over edges with weight > 0.
    Returns K4 candidate (normalized by max_depth).
    """
    if not plexus.has_node(start):
        return 0.0
    visited = {start: 0}
    frontier = [start]
    total_depth = 0
    count = 0
    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        next_frontier = []
        for u in frontier:
            for v, w in plexus.W.get(u, {}).items():
                if w > 0 and v not in visited:
                    visited[v] = depth
                    next_frontier.append(v)
                    total_depth += depth
                    count += 1
        frontier = next_frontier
    if count == 0:
        return 0.0
    avg = total_depth / count
    return float(min(1.0, avg / max_depth))


def compute_K4(plexus: Plexus, cell_id: str) -> float:
    """K4 (D) reasoning depth for one cell."""
    return effective_path_length(plexus, cell_id)


def save_plexus(plexus: Plexus, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"nodes": plexus.nodes, "W": plexus.W}
    path.write_text(json.dumps(data, indent=2))
    return path


def load_plexus(path: Path) -> Plexus:
    data = json.loads(Path(path).read_text())
    return Plexus(nodes=list(data["nodes"]), W=data["W"])


if __name__ == "__main__":
    p = Plexus()
    p.add_node("cell-A")
    p.add_node("cell-B")
    p.add_node("cell-C")
    p.add_node("cell-D")

    # simulate 100 steps of co-activation
    import random
    rng = random.Random(42)
    A = {
        ("cell-A", "cell-B"): 0.8,
        ("cell-A", "cell-C"): 0.3,
        ("cell-B", "cell-C"): 0.6,
        ("cell-C", "cell-D"): 0.9,
    }
    for step in range(100):
        update_from_coactivations(p, A)

    print("Edge weights after 100 steps:")
    for (u, v) in A:
        print(f"  {u} -> {v}: W = {p.edge_weight(u, v):.4f}")

    print()
    print("K4 (depth) from cell-A:", f"{compute_K4(p, 'cell-A'):.3f}")
    print("K4 (depth) from cell-D:", f"{compute_K4(p, 'cell-D'):.3f}")

    out = Path("/tmp/plexus_test.json")
    save_plexus(p, out)
    print(f"\nSaved to {out}")

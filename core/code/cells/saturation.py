#!/usr/bin/env python3
"""
R5-compliant saturation module.

Cell = group of neurons in an MLP layer.
6D knowledge vector K = [L, W, H, D, T, E]

Dimensions (per R5 paper, Section 3.4):
  K1 (L) - sequential coverage (long-range reach)
  K2 (W) - domain breadth (number of domains activated)  [needs domain labels]
  K3 (H) - abstraction level (low entropy = focused)
  K4 (D) - reasoning depth (effective path length)       [needs plexus graph]
  K5 (T) - temporal freshness (exponential decay by age)
  K6 (E) - encompassment (full facet coverage)           [needs facets.json]

S = ||K||_omega (weighted L2, omega in simplex)
S* = 0.85 (split threshold), T_cons = 3 (consecutive steps)
"""
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import numpy as np

N_CELLS_PER_LAYER = 4
S_STAR = 0.85
T_CONSECUTIVE = 3
DECAY_LAMBDA = 0.001  # for K5 (temporal freshness)


@dataclass
class Cell:
    cell_id: str
    layer_idx: int
    neuron_start: int
    neuron_end: int
    K: list = field(default_factory=lambda: [0.0] * 6)
    omega: list = field(default_factory=lambda: [1/6] * 6)
    S: float = 0.0
    S_history: list = field(default_factory=list)
    state: str = "active"
    splits: int = 0
    parent_id: Optional[str] = None
    created_at: str = ""
    last_updated_step: int = 0

    def num_neurons(self) -> int:
        return self.neuron_end - self.neuron_start

def compute_K1_sequential_coverage(activations: np.ndarray) -> float:
    """
    K1 (L): sequential coverage / long-range reach.
    activations: shape [seq_len, num_neurons] for ONE sample/cell.
    Returns: average normalized range of positions where neurons are active.
    High K1 = cell responds to tokens spread across long range.
    Low K1 = cell responds only to nearby tokens.
    """
    if activations.size == 0:
        return 0.0
    a = np.abs(np.asarray(activations, dtype=np.float32))
    seq_len = a.shape[0]
    if seq_len < 2:
        return 0.0
    # per-neuron: measure span between first and last activation
    active = a > a.mean(axis=0, keepdims=True)
    spans = []
    for j in range(a.shape[1]):
        idx = np.where(active[:, j])[0]
        if len(idx) >= 2:
            spans.append((idx[-1] - idx[0]) / (seq_len - 1))
    if not spans:
        return 0.0
    return float(np.clip(np.mean(spans), 0.0, 1.0))

def compute_K3_abstraction(activations: np.ndarray) -> float:
    """
    K3 (H): abstraction level.
    Low entropy = focused (high K3).
    High entropy = diffuse (low K3).
    """
    if activations.size == 0:
        return 0.0
    a = np.abs(np.asarray(activations, dtype=np.float32))
    p = a.mean(axis=0)
    total = p.sum()
    if total < 1e-9:
        return 0.0
    p = p / total
    n = len(p)
    if n < 2:
        return 1.0
    H = -np.sum(p * np.log(p + 1e-12))
    H_max = math.log(n)
    if H_max < 1e-9:
        return 1.0
    return float(np.clip(1.0 - H / H_max, 0.0, 1.0))

def compute_K5_temporal_freshness(
    last_updated_step: int,
    current_step: int,
    decay_lambda: float = DECAY_LAMBDA,
) -> float:
    """
    K5 (T): temporal freshness.
    Exponential decay by age.
    Recently-updated cells: K5 near 1.
    Old cells: K5 near 0.
    """
    age = max(0, current_step - last_updated_step)
    return float(math.exp(-decay_lambda * age))

def compute_K2_domain_breadth(activations: np.ndarray, domains=None) -> float:
    """K2 (W): domain breadth. TODO: needs domain labels per sample."""
    return 0.0


def compute_K4_reasoning_depth(activations: np.ndarray, plexus=None) -> float:
    """K4 (D): reasoning depth. TODO: needs plexus graph."""
    return 0.0


def compute_K6_encompassment(activations: np.ndarray, facets=None) -> float:
    """K6 (E): encompassment. TODO: needs facets.json taxonomy."""
    return 0.0


def compute_S(K: list, omega: list) -> float:
    """Weighted L2 norm of K in the simplex."""
    k = np.asarray(K, dtype=np.float32)
    w = np.asarray(omega, dtype=np.float32)
    w = w / (w.sum() + 1e-9)
    return float(np.sqrt(np.sum(w * k * k)))


def save_cells(cells: list, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in cells], indent=2))
    return path


def load_cells(path: Path) -> list:
    data = json.loads(Path(path).read_text())
    return [Cell(**d) for d in data]


if __name__ == "__main__":
    a = np.random.randn(128, 1536).astype(np.float32)
    K1 = compute_K1_sequential_coverage(a)
    K3 = compute_K3_abstraction(a)
    K5 = compute_K5_temporal_freshness(0, 1000)
    print(f"K1 = {K1:.3f}")
    print(f"K3 = {K3:.3f}")
    print(f"K5 = {K5:.3f}")
    K = [K1, 0.0, K3, 0.0, K5, 0.0]
    print(f"S  = {compute_S(K, [1/6]*6):.3f}")


def load_facets(path=None) -> dict:
    """Load facets taxonomy from facets.json."""
    if path is None:
        path = Path(__file__).parent / "facets.json"
    data = json.loads(Path(path).read_text())
    return data.get("domains", {})


def compute_K6_encompassment_real(
    activated_facets: set,
    domain: str,
    facets_map: dict,
) -> float:
    """
    K6 (E) real implementation.
    activated_facets: set of facet names that the cell reacted to.
    domain: e.g. "medicine"
    facets_map: output of load_facets()
    Returns fraction of facets covered.
    """
    if domain not in facets_map:
        return 0.0
    all_facets = facets_map[domain]
    if not all_facets:
        return 0.0
    covered = set(all_facets) & set(activated_facets)
    return float(len(covered) / len(all_facets))


def compute_K4_from_plexus(cell_id: str, plexus) -> float:
    """K4 (D) via plexus graph. Wrapper for plexus.compute_K4."""
    if plexus is None:
        return 0.0
    try:
        from .plexus import compute_K4 as _k4
        return _k4(plexus, cell_id)
    except Exception:
        return 0.0

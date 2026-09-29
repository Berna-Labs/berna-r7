#!/usr/bin/env python3
"""
R5-compliant DNA Kernel.

100 chromosomes × 4 genes = 400 regulatory parameters per cell.

10 functional groups (each 10 chromosomes):
  0: language
  1: math
  2: code
  3: science
  4: sensory
  5: reasoning
  6: memory
  7: coordination
  8: adaptation
  9: growth

4 genes per chromosome (per R5 Section 3.5):
  a  = activation
  s  = sensitivity
  p  = persistence
  pi = plasticity

Maps (D_c, K_c, u_c) -> {route, grow, split, retain}
"""
import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

N_CHROMOSOMES = 100
N_GENES = 4
GENE_NAMES = ["a", "s", "p", "pi"]

GROUPS = [
    "language", "math", "code", "science", "sensory",
    "reasoning", "memory", "coordination", "adaptation", "growth",
]


@dataclass
class DNA:
    """DNA of one cell. 100 chromosomes x 4 genes."""
    cell_id: str
    genes: list = field(default_factory=list)   # shape [100][4]
    generation: int = 0
    parent_id: Optional[str] = None
    mutations: int = 0
    signature: str = ""

    def __post_init__(self):
        if not self.genes:
            self.genes = [[0.5, 0.5, 0.5, 0.5] for _ in range(N_CHROMOSOMES)]
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        raw = json.dumps(self.genes, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:32]


def random_dna(cell_id: str, seed: int = 42) -> DNA:
    rng = random.Random(seed)
    genes = [[rng.random() for _ in range(N_GENES)] for _ in range(N_CHROMOSOMES)]
    return DNA(cell_id=cell_id, genes=genes, generation=0)


def mutate_gene(g: float, rate: float = 0.01) -> float:
    """Small mutation; clamped to [0,1]."""
    delta = random.uniform(-rate, rate)
    return max(0.0, min(1.0, g + delta))


def mutate_dna(dna: DNA, rate: float = 0.01) -> DNA:
    """Copy with small random changes (used on split)."""
    new_genes = [[mutate_gene(g, rate) for g in row] for row in dna.genes]
    new = DNA(
        cell_id=dna.cell_id + "_m" + str(dna.mutations + 1),
        genes=new_genes,
        generation=dna.generation + 1,
        parent_id=dna.cell_id,
        mutations=dna.mutations + 1,
    )
    return new


def group_mean(dna: DNA, group_idx: int, gene_idx: int) -> float:
    """Mean of a specific gene across the 10 chromosomes of one group."""
    start = group_idx * 10
    end = start + 10
    vals = [dna.genes[i][gene_idx] for i in range(start, end)]
    return sum(vals) / len(vals)


def signal_route(dna: DNA) -> float:
    """Probability of routing to this cell. Mean of 'a' gene across all groups."""
    return sum(dna.genes[i][0] for i in range(N_CHROMOSOMES)) / N_CHROMOSOMES


def signal_grow(dna: DNA) -> float:
    """Readiness to grow. High plasticity (pi) + low persistence (p)."""
    pi = sum(dna.genes[i][3] for i in range(N_CHROMOSOMES)) / N_CHROMOSOMES
    p  = sum(dna.genes[i][2] for i in range(N_CHROMOSOMES)) / N_CHROMOSOMES
    return max(0.0, min(1.0, pi - p + 0.5))


def signal_split(dna: DNA, S: float, s_star: float = 0.85) -> float:
    """Readiness to split. Driven by S vs S* + sensitivity (s)."""
    s = sum(dna.genes[i][1] for i in range(N_CHROMOSOMES)) / N_CHROMOSOMES
    return max(0.0, min(1.0, (S - s_star) + s * 0.2))


def signal_retain(dna: DNA) -> float:
    """Readiness to retain (freeze). High persistence (p) + low plasticity (pi)."""
    p  = sum(dna.genes[i][2] for i in range(N_CHROMOSOMES)) / N_CHROMOSOMES
    pi = sum(dna.genes[i][3] for i in range(N_CHROMOSOMES)) / N_CHROMOSOMES
    return max(0.0, min(1.0, p - pi + 0.5))


def decide(dna: DNA, S: float, S_star: float = 0.85,
           t_consecutive: int = 3) -> dict:
    """
    (D_c, K_c, u_c) -> {route, grow, split, retain}
    Priority-based decision per R5 Section 3.5.

    Priority order (per R5):
      1. split  - if S >= S* sustained (handled by caller's S_history)
      2. retain - if persistence high (frozen state)
      3. grow   - if plasticity high and not saturated
      4. route  - default behavior
    """
    route  = signal_route(dna)
    grow   = signal_grow(dna)
    split  = signal_split(dna, S, S_star)
    retain = signal_retain(dna)

    signals = {"route": route, "grow": grow, "split": split, "retain": retain}

    # Priority-based decision (R5 Section 3.5)
    if S >= S_star:
        decision = "split"
    elif retain > 0.7:
        decision = "retain"
    elif grow > 0.6:
        decision = "grow"
    else:
        decision = "route"

    signals["decision"] = decision
    return signals


def save_dna(dna: DNA, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(dna), indent=2))


def load_dna(path: Path) -> DNA:
    return DNA(**json.loads(Path(path).read_text()))


if __name__ == "__main__":
    dna = random_dna("cell-0001", seed=42)
    print(f"DNA cell: {dna.cell_id}")
    print(f"Signature: {dna.signature}")
    print(f"Genes[0][0:4]: {dna.genes[0]}")
    print(f"Genes[50][0:4]: {dna.genes[50]}")
    print()
    for S in [0.5, 0.7, 0.85, 0.95]:
        d = decide(dna, S)
        print(f"S={S:.2f} -> route={d['route']:.3f} grow={d['grow']:.3f} "
              f"split={d['split']:.3f} retain={d['retain']:.3f} "
              f"=> {d['decision'].upper()}")
    print()
    m = mutate_dna(dna, rate=0.05)
    print(f"Mutated: {m.cell_id} (generation {m.generation})")

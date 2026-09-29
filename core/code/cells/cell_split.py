#!/usr/bin/env python3
"""
R5-compliant cell splitting.

Rule (R5 Hypothesis 1):
  If S(c, t) >= S* for T_cons >= 3 consecutive steps:
      split c into two new cells.

Split mechanics:
  - Parent cell's neurons are partitioned into two halves.
  - Each child inherits:
      * half the neurons
      * mutated DNA (via dna_kernel.mutate_dna)
      * K reset to [0]*6 (re-measure needed)
  - Parent state becomes 'split'.
  - Children registered in the cell registry.
"""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from .saturation import Cell, S_STAR, T_CONSECUTIVE
from .dna_kernel import DNA, mutate_dna, random_dna


def is_ready_to_split(cell: Cell, t_consecutive: int = T_CONSECUTIVE,
                      s_star: float = S_STAR) -> bool:
    """
    Check if cell has S >= S* for at least t_consecutive steps.
    Uses the cell's S_history (most recent t_consecutive entries).
    """
    if len(cell.S_history) < t_consecutive:
        return False
    recent = cell.S_history[-t_consecutive:]
    return all(s >= s_star for s in recent)


def can_split(cell: Cell, min_neurons: int = 2) -> bool:
    """Cell must have at least min_neurons to split."""
    return cell.num_neurons() >= min_neurons and cell.state == "active"


def split_cell(cell: Cell, dna: DNA, split_id: str) -> tuple:
    """
    Split a cell into two.
    Returns (child_a, child_b, dna_a, dna_b).
    """
    mid = (cell.neuron_start + cell.neuron_end) // 2

    # Children inherit halves of the parent's neuron range
    child_a = Cell(
        cell_id=f"{cell.cell_id}-a",
        layer_idx=cell.layer_idx,
        neuron_start=cell.neuron_start,
        neuron_end=mid,
        state="active",
        parent_id=cell.cell_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    child_b = Cell(
        cell_id=f"{cell.cell_id}-b",
        layer_idx=cell.layer_idx,
        neuron_start=mid,
        neuron_end=cell.neuron_end,
        state="active",
        parent_id=cell.cell_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # Mutate DNA for each child
    dna_a = mutate_dna(dna, rate=0.02)
    dna_a.cell_id = child_a.cell_id
    dna_b = mutate_dna(dna, rate=0.02)
    dna_b.cell_id = child_b.cell_id

    return child_a, child_b, dna_a, dna_b


def apply_split(cell: Cell, dna: DNA, registry_path: Optional[Path] = None) -> dict:
    """
    Check + split if ready.
    Returns dict with:
      - split: bool
      - reason: str
      - children: list of cell_ids (if split)
    """
    if not can_split(cell):
        return {"split": False, "reason": "cannot_split", "children": []}

    if not is_ready_to_split(cell):
        return {"split": False, "reason": "S_not_sustained", "children": []}

    child_a, child_b, dna_a, dna_b = split_cell(cell, dna, split_id=cell.cell_id)

    # Parent state -> 'split'
    cell.state = "split"
    cell.splits += 1

    # Save children + DNA if registry path is provided
    if registry_path:
        registry_path = Path(registry_path)
        registry_path.mkdir(parents=True, exist_ok=True)
        from .saturation import save_cells
        save_cells([child_a, child_b], registry_path / f"{cell.cell_id}_children.json")
        from .dna_kernel import save_dna
        save_dna(dna_a, registry_path / f"{dna_a.cell_id}.dna.json")
        save_dna(dna_b, registry_path / f"{dna_b.cell_id}.dna.json")

    return {
        "split": True,
        "reason": "S_sustained_above_star",
        "children": [child_a.cell_id, child_b.cell_id],
    }


if __name__ == "__main__":
    from .saturation import Cell
    from .dna_kernel import random_dna

    # Create a test cell with S history above S*
    c = Cell(cell_id="cell-test", layer_idx=0, neuron_start=0, neuron_end=1536)
    c.S_history = [0.5, 0.6, 0.87, 0.90, 0.92]  # last 3 above S*=0.85
    c.S = 0.92

    dna = random_dna(c.cell_id, seed=42)

    print(f"Cell: {c.cell_id}, neurons {c.neuron_start}-{c.neuron_end}")
    print(f"S_history (last 3): {c.S_history[-3:]}")
    print(f"Ready to split? {is_ready_to_split(c)}")
    print()

    result = apply_split(c, dna, registry_path=None)
    print(f"Result: {result}")
    print(f"Parent state after split: {c.state}")

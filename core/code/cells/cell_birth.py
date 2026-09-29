#!/usr/bin/env python3
"""
R5-compliant cell birth.

Initial partition:
  Each MLP layer's intermediate_size is divided into N_CELLS_PER_LAYER cells.
  Each cell gets:
    - a unique cell_id (model_uuid-cell-XXXX)
    - random DNA (or inherited from parent model)
    - K reset to [0]*6 (measured during training)
    - state = 'active'
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List

from .saturation import Cell, N_CELLS_PER_LAYER
from .dna_kernel import DNA, random_dna, load_dna


def birth_cells_for_layer(
    layer_idx: int,
    intermediate_size: int,
    n_cells: int = N_CELLS_PER_LAYER,
    model_uuid: str = "mother-001",
) -> List[Cell]:
    """
    Create n_cells cells for a single MLP layer.
    Each cell gets an equal share of the layer's neurons.
    """
    if n_cells < 1:
        raise ValueError("n_cells must be >= 1")
    if intermediate_size < n_cells:
        raise ValueError("intermediate_size must be >= n_cells")

    step = intermediate_size // n_cells
    cells = []
    for i in range(n_cells):
        start = i * step
        end = intermediate_size if i == n_cells - 1 else (i + 1) * step
        cell_id = f"{model_uuid}-cell-{layer_idx:02d}{i:02d}"
        c = Cell(
            cell_id=cell_id,
            layer_idx=layer_idx,
            neuron_start=start,
            neuron_end=end,
            state="active",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        cells.append(c)
    return cells


def initialize_model_cells(config: dict, model_uuid: str = "mother-001",
                            n_cells_per_layer: int = N_CELLS_PER_LAYER) -> List[Cell]:
    """
    Create cells for all MLP layers of a model.
    config must have 'num_hidden_layers' and 'intermediate_size'.
    """
    num_layers = int(config["num_hidden_layers"])
    interm = int(config["intermediate_size"])
    all_cells = []
    for layer_idx in range(num_layers):
        cells = birth_cells_for_layer(
            layer_idx=layer_idx,
            intermediate_size=interm,
            n_cells=n_cells_per_layer,
            model_uuid=model_uuid,
        )
        all_cells.extend(cells)
    return all_cells


def make_dna_for_cells(cells: List[Cell], seed: int = 42) -> dict:
    """Create random DNA for every cell. Returns {cell_id: DNA}."""
    out = {}
    for i, c in enumerate(cells):
        out[c.cell_id] = random_dna(c.cell_id, seed=seed + i)
    return out


def save_birth(cells: List[Cell], dnas: dict, out_dir: Path) -> Path:
    """Save cells.json + dna/ to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from .saturation import save_cells
    from .dna_kernel import save_dna

    cells_path = out_dir / "cells.json"
    save_cells(cells, cells_path)

    dna_dir = out_dir / "dna"
    dna_dir.mkdir(exist_ok=True)
    for cell_id, dna in dnas.items():
        save_dna(dna, dna_dir / f"{cell_id}.dna.json")

    return cells_path


if __name__ == "__main__":
    import sys
    # Default test config (matches Mother)
    cfg_path = Path("/data/berna-r7/core/config/config.json")
    if len(sys.argv) > 1:
        cfg_path = Path(sys.argv[1])

    with open(cfg_path) as f:
        config = json.load(f)

    print(f"Config: layers={config['num_hidden_layers']}, interm={config['intermediate_size']}")

    cells = initialize_model_cells(config, model_uuid="mother-001")
    print(f"Created {len(cells)} cells")

    dnas = make_dna_for_cells(cells)
    print(f"Created {len(dnas)} DNA records")

    out = Path("/data/berna-r7/core/cells")
    p = save_birth(cells, dnas, out)
    print(f"Saved: {p}")

    # show first 3 cells
    for c in cells[:3]:
        print(f"  {c.cell_id}: neurons {c.neuron_start}-{c.neuron_end}")

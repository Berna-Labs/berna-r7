#!/usr/bin/env python3
"""
H1 Experiment: Saturation-Triggered Splitting.

Verifies R5 Hypothesis 1:
  If S(c,t) >= S* = 0.85 for T_cons >= 3 consecutive measurements,
  then splitting c weakly decreases global loss.

Method:
  1. Load Mother (or checkpoint).
  2. For each validation batch, capture per-cell activations.
  3. Compute K (6D) and S = ||K||_omega for each cell.
  4. Track S_history; trigger split when S >= S* for 3 steps.
  5. Measure loss before vs after split.

Output: results/h1_saturation.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

sys.path.insert(0, "/data/berna-r7/core/code")

from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from modeling_berna import BernaForCausalLM


class BernaConfig(PretrainedConfig):
    model_type = "berna"
    def __init__(self, **kw):
        super().__init__(**kw)


CONFIG_MAPPING.register("berna", BernaConfig, exist_ok=True)


class ActivationRecorder:
    """Captures intermediate MLP activations (silu(gate)*up) per layer."""

    def __init__(self, model, n_cells_per_layer=4):
        self.model = model
        self.n_cells = n_cells_per_layer
        self.gates = {}
        self.ups = {}
        self.handles = []
        self._register()

    def _register(self):
        layers = self.model.model.layers
        for idx, layer in enumerate(layers):
            mlp = layer.mlp
            self.handles.append(
                mlp.gate_proj.register_forward_hook(self._make(idx, self.gates))
            )
            self.handles.append(
                mlp.up_proj.register_forward_hook(self._make(idx, self.ups))
            )

    @staticmethod
    def _make(idx, store):
        def hook(module, inp, out):
            store[idx] = out.detach()
        return hook

    def get_intermediate(self, layer_idx):
        g = self.gates.get(layer_idx)
        u = self.ups.get(layer_idx)
        if g is None or u is None:
            return None
        return F.silu(g) * u  # [batch, seq, I]

    def clear(self):
        self.gates.clear()
        self.ups.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def extract_cell_activation(intermediate, cell, n_cells_per_layer, batch_idx=0):
    """Slice [batch, seq, I] -> [seq, cell_size] for the given cell."""
    if intermediate is None:
        return None
    x = intermediate[batch_idx]  # [seq, I]
    I = x.shape[-1]
    step = I // n_cells_per_layer
    start = cell.neuron_start
    end = cell.neuron_end
    return x[:, start:end].float().cpu().numpy()


def measure_all_cells(model, mgr, recorder, x, device, batch_idx=0):
    """Run forward on x, then measure every active cell."""
    with torch.no_grad():
        _ = model(x.to(device))

    activations = {}
    for cell in mgr.cells:
        if cell.state != "active":
            continue
        inter = recorder.get_intermediate(cell.layer_idx)
        a = extract_cell_activation(inter, cell, recorder.n_cells, batch_idx)
        if a is not None and a.size > 0:
            activations[cell.cell_id] = a
    return activations


def run_h1(ckpt_dir=None, n_batches=30, max_splits=5, device="cuda"):
    from cells.cell_manager import CellManager
    from cells.cell_birth import initialize_model_cells, make_dna_for_cells

    out_dir = Path("/data/berna-r7/core/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    cfg_path = Path("/data/berna-r7/core/config/config.json")
    with open(cfg_path) as f:
        cfg = BernaConfig(**json.load(f))
    model = BernaForCausalLM(cfg)

    if ckpt_dir:
        from safetensors.torch import load_file
        mfile = Path(ckpt_dir) / "model" / "model.safetensors"
        if mfile.exists():
            model.load_state_dict(load_file(str(mfile)))
            print(f"Loaded: {ckpt_dir}")
    model = model.to(device).eval()

    # Cells
    mgr = CellManager("mother-001", base_dir=Path("/data/berna-r7/core/cells"))
    if not mgr.load():
        cells = initialize_model_cells(cfg.to_dict(), model_uuid="mother-001")
        mgr.cells = cells
        mgr.dnas = make_dna_for_cells(cells)

    print(f"Cells: {len(mgr.cells)}")

    recorder = ActivationRecorder(model, n_cells_per_layer=4)

    # Data
    from dataset import BernaDataset
    from torch.utils.data import DataLoader
    ds = BernaDataset(seq_len=2048)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    S_history_global = []
    splits_log = []

    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        acts = measure_all_cells(model, mgr, recorder, x, device, batch_idx=0)
        recorder.clear()

        summary = mgr.step(activations_by_cell=acts, coactivations=None)
        step_S = [(c.cell_id, round(c.S, 4)) for c in mgr.cells if c.state == "active"]
        S_history_global.append({"batch": i+1, "cells": step_S})

        if summary["splits"]:
            for s in summary["splits"]:
                splits_log.append({"batch": i+1, "info": s})
                print(f"  SPLIT at batch {i+1}: {s['children']}")

        if len(splits_log) >= max_splits:
            print(f"Reached max_splits={max_splits}, stopping.")
            break

    result = {
        "n_batches": len(S_history_global),
        "n_cells_initial": 112,
        "n_cells_final": len(mgr.cells),
        "splits": splits_log,
        "S_history": S_history_global,
    }
    out = out_dir / "h1_saturation.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved: {out}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    dev = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    run_h1(ckpt_dir=args.ckpt, n_batches=args.batches,
           max_splits=args.splits, device=dev)

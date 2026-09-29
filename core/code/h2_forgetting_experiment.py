#!/usr/bin/env python3
"""
H2 Experiment: Bounded Forgetting.

Verifies R5 Hypothesis 2:
  For all tasks j and times t:
    F_j(t) = max_{s<=t} P_j(s) - P_j(t) <= eps_neg = 1e-6

Method:
  1. Load Mother.
  2. Measure baseline P_j(0) on held-out slices of {eng, math, code}.
  3. Train a Child on a new domain (e.g., a subset of code).
  4. Re-measure P_j(1) on the same held-out slices.
  5. Compute F_j = max(0, P_j(0) - P_j(1)).

Output: results/h2_forgetting.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/berna-r7/core/code")

from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from modeling_berna import BernaForCausalLM

EPS_NEG = 1e-6


class BernaConfig(PretrainedConfig):
    model_type = "berna"
    def __init__(self, **kw):
        super().__init__(**kw)


CONFIG_MAPPING.register("berna", BernaConfig, exist_ok=True)


# Held-out task slices (taken from the tail of each domain's tokenized data,
# never seen during training because training stops before TOTAL_STEPS).
HELD_OUT = {
    "eng_task":  {"path": "/data/berna-r7/core/data/tokenized/eng.npy",
                  "slice": "tail", "n_tokens": 2_000_000},
    "math_task": {"path": "/data/berna-r7/core/data/tokenized/math.npy",
                  "slice": "tail", "n_tokens": 2_000_000},
    "code_task": {"path": "/data/berna-r7/core/data/tokenized/code.npy",
                  "slice": "tail", "n_tokens": 2_000_000},
}


def load_slice(spec, seq_len=2048, max_batches=10):
    """Load a tail slice as flat int64 array (for quick P_j evaluation)."""
    arr = np.load(spec["path"], mmap_mode="r")
    n = spec["n_tokens"]
    tail = arr[-n:]
    return np.asarray(tail, dtype=np.int64)[: seq_len * max_batches + 1]


@torch.no_grad()
def measure_P(model, tokens, seq_len=2048, max_batches=10, device="cuda"):
    """
    Return P = exp(-loss) averaged over batches.
    This is a simple bounded metric: higher = better.
    """
    model.eval()
    n = min(len(tokens) // (seq_len + 1), max_batches)
    if n == 0:
        return float("nan")
    total = 0.0
    for i in range(n):
        s = i * seq_len
        x = torch.from_numpy(tokens[s:s+seq_len]).unsqueeze(0).to(device)
        y = torch.from_numpy(tokens[s+1:s+seq_len+1]).unsqueeze(0).to(device)
        out = model(input_ids=x, labels=y)
        total += float(torch.exp(-out.loss).item())
    return total / n


def measure_all(model, tasks, device="cuda"):
    """Measure P for every task. Returns {task_name: P}."""
    out = {}
    for name, spec in tasks.items():
        toks = load_slice(spec)
        out[name] = measure_P(model, toks, device=device)
    return out


def compute_F(P_before, P_after):
    """F_j = max(0, P_before - P_after)."""
    F = {}
    for k in P_before:
        delta = P_before[k] - P_after.get(k, 0.0)
        F[k] = max(0.0, delta)
    return F


def load_model_from_ckpt(ckpt_dir, device="cuda"):
    cfg_path = Path("/data/berna-r7/core/config/config.json")
    with open(cfg_path) as f:
        cfg = BernaConfig(**json.load(f))
    model = BernaForCausalLM(cfg)
    mfile = Path(ckpt_dir) / "model" / "model.safetensors"
    if mfile.exists():
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(mfile)))
    return model.to(device).eval()


def run_h2(mother_ckpt, child_ckpt=None, device="cuda"):
    """
    mother_ckpt: path to step_XXXXX of Mother
    child_ckpt:  path to step_XXXXX of Child (optional)
    """
    out_dir = Path("/data/berna-r7/core/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Mother: {mother_ckpt}")
    mother = load_model_from_ckpt(mother_ckpt, device=device)
    P_mother = measure_all(mother, HELD_OUT, device=device)
    print(f"Mother P: {P_mother}")

    result = {"P_mother": P_mother, "P_child": None, "F": None, "max_F": None}

    if child_ckpt:
        print(f"Loading Child: {child_ckpt}")
        child = load_model_from_ckpt(child_ckpt, device=device)
        P_child = measure_all(child, HELD_OUT, device=device)
        F = compute_F(P_mother, P_child)
        result["P_child"] = P_child
        result["F"] = F
        result["max_F"] = max(F.values())
        result["bounded_ok"] = result["max_F"] <= EPS_NEG
        print(f"Child P: {P_child}")
        print(f"F: {F}  max={result['max_F']:.2e}")
    else:
        print("No Child checkpoint yet; saved Mother baseline only.")

    out = out_dir / "h2_forgetting.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved: {out}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mother", required=True)
    ap.add_argument("--child", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dev = "cpu" if args.cpu else "cuda"
    run_h2(args.mother, args.child, device=dev)

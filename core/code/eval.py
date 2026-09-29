#!/usr/bin/env python3
"""Berna R7 evaluation - perplexity on eng/math/code."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/berna-r7/core/code")

from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from modeling_berna import BernaForCausalLM

DATA = Path("/data/berna-r7/core/data/tokenized")
CKPT_DIR = Path("/data/berna-r7/core/checkpoints")


class BernaConfig(PretrainedConfig):
    model_type = "berna"
    def __init__(self, **kw):
        super().__init__(**kw)


CONFIG_MAPPING.register("berna", BernaConfig, exist_ok=True)


def load_model(device="cuda"):
    cfg_path = Path("/data/berna-r7/core/config/config.json")
    with open(cfg_path) as f:
        cfg = BernaConfig(**json.load(f))
    model = BernaForCausalLM(cfg)

    if CKPT_DIR.exists():
        dirs = sorted([d for d in CKPT_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("step_")])
        if dirs:
            latest = dirs[-1]
            mfile = latest / "model" / "model.safetensors"
            if mfile.exists():
                from safetensors.torch import load_file
                model.load_state_dict(load_file(str(mfile)))
                print(f"Loaded: {latest.name}")
    return model.to(device).eval()


@torch.no_grad()
def perplexity(model, tokens, seq_len=4096, max_batches=20, device="cuda"):
    tokens = np.asarray(tokens, dtype=np.int64)
    n = min(len(tokens) // (seq_len + 1), max_batches)
    if n == 0:
        return float("nan")
    total_loss = 0.0
    for i in range(n):
        s = i * seq_len
        x = torch.from_numpy(tokens[s:s+seq_len]).unsqueeze(0).to(device)
        y = torch.from_numpy(tokens[s+1:s+seq_len+1]).unsqueeze(0).to(device)
        out = model(input_ids=x, labels=y)
        total_loss += out.loss.item() * seq_len
    return math.exp(total_loss / (n * seq_len))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = load_model(device=device)

    print()
    print(f"{'domain':10s}  {'perplexity':>12s}")
    print("-" * 30)
    for f in ["eng.npy", "math.npy", "code.npy"]:
        p = DATA / f
        if not p.exists():
            print(f"{f:10s}  (missing)")
            continue
        tokens = np.load(p, mmap_mode="r")
        ppl = perplexity(model, tokens, device=device)
        print(f"{f:10s}  {ppl:12.2f}")


if __name__ == "__main__":
    main()

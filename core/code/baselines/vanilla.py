#!/usr/bin/env python3
"""
Baseline: vanilla causal LM (no cells, no splitting).

Same architecture as Berna R7 Mother, but:
  - No cell partitioning
  - No DNA Kernel
  - No Plexus
  - No splitting

This is the control condition for H1 and H2 comparisons.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/berna-r7/core/code")

from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from modeling_berna import BernaForCausalLM


class BernaConfig(PretrainedConfig):
    model_type = "berna"
    def __init__(self, **kw):
        super().__init__(**kw)


CONFIG_MAPPING.register("berna", BernaConfig, exist_ok=True)


def load_mother_config():
    with open("/data/berna-r7/core/config/config.json") as f:
        return BernaConfig(**json.load(f))


def train_vanilla(steps=1000, device="cuda", lr=3e-4, log_every=50):
    """
    Train a vanilla model on eng+math+code for `steps` steps.
    Same hyperparameters as Berna R7, but no cell machinery.
    """
    from dataset import BernaDataset
    from torch.utils.data import DataLoader
    import bitsandbytes as bnb

    cfg = load_mother_config()
    model = BernaForCausalLM(cfg).to(device)
    print(f"Vanilla model: {sum(p.numel() for p in model.parameters()):,} params")

    opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    ds = BernaDataset(seq_len=2048)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    it = iter(loader)

    model.train()
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x = x.to(device)
        y = y.to(device)
        out = model(input_ids=x, labels=y)
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % log_every == 0:
            print(f"vanilla step {step+1} | loss {loss.item():.4f}")

    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dev = "cpu" if args.cpu else "cuda"
    train_vanilla(steps=args.steps, device=dev)

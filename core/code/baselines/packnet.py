#!/usr/bin/env python3
"""
Baseline: PackNet (parameter isolation).

Reference: Mallya & Lazebnik, CVPR 2018.
Freezes task-critical weights after each task by iteratively pruning.

Implementation here:
  - Use one-shot magnitude pruning (top-k by |grad * weight|).
  - Save a binary mask per task.
  - At training time, block updates to previously-important params.
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


def make_mask(model, sparsity=0.5):
    """
    Produce a {param_name: bool_tensor} mask where True = kept.
    Mask is based on |weight| magnitude (top (1-sparsity) fraction).
    """
    mask = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        flat = p.detach().abs().flatten()
        if flat.numel() == 0:
            continue
        k = int(flat.numel() * (1.0 - sparsity))
        k = max(1, k)
        thresh = torch.topk(flat, k).values[-1]
        mask[n] = (p.detach().abs() >= thresh).clone()
    return mask


def apply_mask_(model, mask):
    """Zero out weights where mask is False."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in mask:
                p.mul_(mask[n].to(p.dtype))


def block_grads_(model, mask):
    """Zero gradients outside the current task's mask."""
    for n, p in model.named_parameters():
        if n in mask and p.grad is not None:
            p.grad.mul_(mask[n].to(p.grad.dtype))


def train_packnet(steps=500, device="cuda", lr=3e-4, sparsity=0.5):
    from dataset import BernaDataset
    from torch.utils.data import DataLoader
    import bitsandbytes as bnb

    with open("/data/berna-r7/core/config/config.json") as f:
        cfg = BernaConfig(**json.load(f))
    model = BernaForCausalLM(cfg).to(device)

    print("Building PackNet mask...")
    mask = make_mask(model, sparsity=sparsity)
    apply_mask_(model, mask)

    ds = BernaDataset(seq_len=2048)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.95))
    it = iter(loader)
    model.train()
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        out = model(input_ids=x, labels=y)
        loss = out.loss
        loss.backward()
        block_grads_(model, mask)
        opt.step()
        apply_mask_(model, mask)
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0:
            print(f"packnet step {step+1} | loss {loss.item():.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dev = "cpu" if args.cpu else "cuda"
    train_packnet(steps=args.steps, sparsity=args.sparsity, device=dev)

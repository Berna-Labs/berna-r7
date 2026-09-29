#!/usr/bin/env python3
"""
Baseline: EWC (Elastic Weight Consolidation).

Reference: Kirkpatrick et al., PNAS 2017.
Adds a quadratic penalty to preserve important weights.

Loss = task_loss + (lambda/2) * sum_i F_i * (theta_i - theta_i*)^2

where F_i = Fisher information of parameter i.
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


def compute_fisher(model, data_loader, n_batches=10, device="cuda"):
    """Estimate Fisher diagonal from n_batches."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    n_done = 0
    for x, y in data_loader:
        if n_done >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(input_ids=x, labels=y)
        out.loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        n_done += 1
    for n in fisher:
        fisher[n] /= max(1, n_done)
    return fisher


def ewc_loss(model, fisher, theta_star, lam=1000.0):
    """Quadratic EWC penalty."""
    penalty = 0.0
    for n, p in model.named_parameters():
        if n in fisher:
            penalty = penalty + (fisher[n] * (p - theta_star[n]) ** 2).sum()
    return (lam / 2.0) * penalty


def train_ewc(steps=500, device="cuda", lr=3e-4, lam=1000.0):
    from dataset import BernaDataset
    from torch.utils.data import DataLoader
    import bitsandbytes as bnb

    with open("/data/berna-r7/core/config/config.json") as f:
        cfg = BernaConfig(**json.load(f))
    model = BernaForCausalLM(cfg).to(device)

    ds = BernaDataset(seq_len=2048)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # Snapshot theta_star and Fisher from a few batches
    print("Estimating Fisher...")
    fisher = compute_fisher(model, loader, n_batches=10, device=device)
    theta_star = {n: p.detach().clone() for n, p in model.named_parameters()}

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
        loss = out.loss + ewc_loss(model, fisher, theta_star, lam=lam)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0:
            print(f"ewc step {step+1} | loss {loss.item():.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dev = "cpu" if args.cpu else "cuda"
    train_ewc(steps=args.steps, lam=args.lam, device=dev)


def compute_fisher(model, data_loader, n_batches=10, device="cuda"):
    """Estimate Fisher diagonal from n_batches."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    n_done = 0
    for x, y in data_loader:
        if n_done >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(input_ids=x, labels=y)
        out.loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        n_done += 1
    for n in fisher:
        fisher[n] /= max(1, n_done)
    return fisher


def ewc_loss(model, fisher, theta_star, lam=1000.0):
    """Quadratic EWC penalty."""
    penalty = 0.0
    for n, p in model.named_parameters():
        if n in fisher:
            penalty = penalty + (fisher[n] * (p - theta_star[n]) ** 2).sum()
    return (lam / 2.0) * penalty


def train_ewc(steps=500, device="cuda", lr=3e-4, lam=1000.0):
    from dataset import BernaDataset
    from torch.utils.data import DataLoader
    import bitsandbytes as bnb

    with open("/data/berna-r7/core/config/config.json") as f:
        cfg = BernaConfig(**json.load(f))
    model = BernaForCausalLM(cfg).to(device)

    ds = BernaDataset(seq_len=2048)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    print("Estimating Fisher...")
    fisher = compute_fisher(model, loader, n_batches=10, device=device)
    theta_star = {n: p.detach().clone() for n, p in model.named_parameters()}

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
        loss = out.loss + ewc_loss(model, fisher, theta_star, lam=lam)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0:
            print(f"ewc step {step+1} | loss {loss.item():.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dev = "cpu" if args.cpu else "cuda"
    train_ewc(steps=args.steps, lam=args.lam, device=dev)

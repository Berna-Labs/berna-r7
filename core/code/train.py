#!/usr/bin/env python3
import os, sys, json, time, argparse, hashlib, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from torch.utils.data import DataLoader
import bitsandbytes as bnb
from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
sys.path.insert(0, "/data/berna-r7/core/code")
from modeling_berna import BernaForCausalLM
from dataset import BernaDataset

ROOT = Path("/data/berna-r7/core")
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "train.log"

SEQ_LEN = 4096
MICRO_BATCH = 1
GRAD_ACCUM = 64
LR = 3e-4
WARMUP = 500
TOTAL_STEPS = 10200
SAVE_EVERY = 50
LOG_EVERY = 50
KEEP_LAST = 3

class BernaConfig(PretrainedConfig):
    model_type = "berna"
    def __init__(self, **kw):
        super().__init__(**kw)

CONFIG_MAPPING.register("berna", BernaConfig, exist_ok=True)

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_config():
    with open(ROOT / "config" / "config.json") as f:
        d = json.load(f)
    return BernaConfig(**d)

def lr_lambda(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    p = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
    p = min(p, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * p))

def hashing(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def write_manifest(step, loss, tokens_seen):
    cfg_bytes = (ROOT / "config" / "config.json").read_bytes()
    m = {
        "uuid": "mother-001",
        "name": "Berna-R7-Mother",
        "level": 0,
        "parent_uuid": None,
        "connection_token": hashing(b"mother-001|root|Berna-R7-Mother")[:32],
        "children": [],
        "step": step,
        "tokens_seen": tokens_seen,
        "loss": loss,
        "config_hash": hashing(cfg_bytes),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "manifest.json").write_text(json.dumps(m, indent=2))

def save_checkpoint(model, optimizer, scheduler, step, tokens_seen, loss):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    final = CKPT_DIR / f"step_{step:08d}"
    tmp = CKPT_DIR / f".tmp_{step:08d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    (tmp / "model").mkdir()
    model.save_pretrained(tmp / "model", safe_serialization=True)
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    torch.save(scheduler.state_dict(), tmp / "scheduler.pt")
    meta = {"step": step, "tokens_seen": tokens_seen, "loss": loss,
            "saved_at": datetime.now(timezone.utc).isoformat()}
    (tmp / "metadata.json").write_text(json.dumps(meta, indent=2))
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return final

def cleanup_old_checkpoints(keep=KEEP_LAST):
    if not CKPT_DIR.exists():
        return
    dirs = sorted([d for d in CKPT_DIR.iterdir()
                   if d.is_dir() and d.name.startswith("step_")])
    for d in dirs[:-keep]:
        shutil.rmtree(d)

def find_latest():
    if not CKPT_DIR.exists():
        return None
    dirs = sorted([d for d in CKPT_DIR.iterdir()
                   if d.is_dir() and d.name.startswith("step_")])
    return dirs[-1] if dirs else None

def train(args):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("Berna R7 - Mother Training")
    log("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = load_config()
    model = BernaForCausalLM(cfg).to(device)
    n = sum(p.numel() for p in model.parameters())
    log(f"Model: {n:,} parameters")

    model.gradient_checkpointing_enable()
    log("Gradient checkpointing: ENABLED")

    def opt_ctor(params):
        return bnb.optim.AdamW8bit(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    optimizer = opt_ctor(model.parameters())
    log("Optimizer: AdamW8bit")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    tokens_seen = 0

    if args.resume:
        latest = find_latest()
        if latest:
            log(f"Resuming from: {latest}")
            model_file = latest / "model" / "model.safetensors"
            if model_file.exists():
                from safetensors.torch import load_file
                sd = load_file(str(model_file))
                model.load_state_dict(sd)
            opt_file = latest / "optimizer.pt"
            if opt_file.exists():
                optimizer.load_state_dict(torch.load(opt_file, map_location=device, weights_only=False))
            sch_file = latest / "scheduler.pt"
            if sch_file.exists():
                scheduler.load_state_dict(torch.load(sch_file, map_location=device, weights_only=False))
            meta_file = latest / "metadata.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                start_step = meta["step"]
                tokens_seen = meta["tokens_seen"]
                log(f"Resumed at step {start_step}, tokens {tokens_seen:,}")

    dataset = BernaDataset(seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=False,
                        num_workers=0, pin_memory=True)
    it = iter(loader)

    log(f"Total steps: {TOTAL_STEPS}")
    log(f"Effective batch: {MICRO_BATCH * GRAD_ACCUM} x {SEQ_LEN} = {MICRO_BATCH * GRAD_ACCUM * SEQ_LEN:,} tokens/step")
    log("=" * 60)

    model.train()
    t0 = time.time()
    last_log_tokens = tokens_seen
    last_log_time = t0

    for step in range(start_step, TOTAL_STEPS):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(input_ids=x, labels=y)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        tokens_seen += MICRO_BATCH * GRAD_ACCUM * SEQ_LEN

        if (step + 1) % LOG_EVERY == 0:
            now = time.time()
            dt = now - last_log_time
            dtok = tokens_seen - last_log_tokens
            tps = dtok / max(dt, 1)
            lr = optimizer.param_groups[0]["lr"]
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"step {step+1:>6} | loss {total_loss:.4f} | lr {lr:.2e} | tok/s {int(tps)} | VRAM {vram:.1f}GB | tokens {tokens_seen:,}")
            last_log_tokens = tokens_seen
            last_log_time = now

        if (step + 1) % SAVE_EVERY == 0:
            p = save_checkpoint(model, optimizer, scheduler, step + 1, tokens_seen, total_loss)
            log(f"Checkpoint saved: {p.name}")
            write_manifest(step + 1, total_loss, tokens_seen)
            cleanup_old_checkpoints()

    log("Training complete!")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    train(ap.parse_args())

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset

DATA = Path("/data/berna-r7/core/data/tokenized")
FILES = ["eng.npy", "math.npy", "code.npy"]

class BernaDataset(Dataset):
    def __init__(self, seq_len=4096):
        self.seq_len = seq_len
        parts = []
        for f in FILES:
            a = np.load(DATA / f, mmap_mode="r")
            parts.append(np.asarray(a))
            print(f"  {f}: {len(a):,} tokens")
        self.data = np.concatenate(parts)
        self.total = len(self.data)
        self.n_samples = (self.total - 1) // seq_len
        print(f"  Total: {self.total:,} | Samples: {self.n_samples:,}")
    def __len__(self): return self.n_samples
    def __getitem__(self, i):
        s = i * self.seq_len
        buf = self.data[s:s+self.seq_len+1]
        return torch.from_numpy(buf[:-1].astype(np.int64)), torch.from_numpy(buf[1:].astype(np.int64))

if __name__ == "__main__":
    ds = BernaDataset()
    x, y = ds[0]
    print(f"OK: {x.shape}, {y.shape}")

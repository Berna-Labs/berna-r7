#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from datasets import load_dataset
import sys, time

TOK_PATH = "/data/berna-r7/core/tokenizer/tokenizer.json"
OUT = Path("/data/berna-r7/core/data/tokenized")
ENG = Path("/data/berna-r4/corpus/multilingual/eng.txt")
MATH = Path("/data/berna-r4/corpus/math/openwebmath.txt")
CODE_DIR = Path("/data/berna-r4/corpus/parquet_code")

tok = Tokenizer.from_file(TOK_PATH)
print(f"Vocab: {tok.get_vocab_size()}")

def process_text_file(path, out_path, chunk_lines=10000):
    out = []
    t0 = time.time()
    n_lines = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\\n")
            if not line:
                continue
            ids = tok.encode(line).ids
            out.extend(ids)
            n_lines += 1
            if n_lines % 100000 == 0:
                el = time.time() - t0
                print(f"  {n_lines:,} lines | {len(out):,} tokens | {el:.0f}s", flush=True)
    arr = np.array(out, dtype=np.uint32)
    np.save(out_path, arr)
    print(f"Saved: {out_path} ({len(arr):,} tokens)")
    return len(arr)

def process_code_dir(code_dir, out_path):
    out = []
    t0 = time.time()
    files = sorted(code_dir.glob("*.parquet"))
    print(f"Found {len(files)} parquet files")
    for fi, f in enumerate(files):
        ds = load_dataset("parquet", data_files=str(f), split="train")
        for row in ds:
            for k in ("content", "code", "text"):
                if k in row and row[k]:
                    ids = tok.encode(row[k]).ids
                    out.extend(ids)
                    break
        if (fi + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {fi+1}/{len(files)} files | {len(out):,} tokens | {el:.0f}s", flush=True)
    arr = np.array(out, dtype=np.uint32)
    np.save(out_path, arr)
    print(f"Saved: {out_path} ({len(arr):,} tokens)")
    return len(arr)

def main():
    t0 = time.time()
    print("=== English ===")
    n_eng = process_text_file(ENG, OUT / "eng.npy")
    print("=== Math ===")
    n_math = process_text_file(MATH, OUT / "math.npy")
    print("=== Code ===")
    n_code = process_code_dir(CODE_DIR, OUT / "code.npy")
    total = n_eng + n_math + n_code
    print()
    print(f"=== DONE ===")
    print(f"eng : {n_eng:>15,}")
    print(f"math: {n_math:>15,}")
    print(f"code: {n_code:>15,}")
    print(f"TOT : {total:>15,}")
    print(f"Time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()

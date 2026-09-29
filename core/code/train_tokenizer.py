#!/usr/bin/env python3
import json, sys, itertools
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_dataset

VOCAB_SIZE = 32000
ENG = Path("/data/berna-r4/corpus/multilingual/eng.txt")
MATH = Path("/data/berna-r4/corpus/math/openwebmath.txt")
CODE_DIR = Path("/data/berna-r4/corpus/parquet_code")
OUT = Path("/data/berna-r7/core/tokenizer")
SAMPLE_BYTES = 500 * 1024 * 1024

def read_text(path, max_bytes):
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line
            n += len(line.encode("utf-8"))
            if n >= max_bytes:
                break

def read_code_parquets():
    files = sorted(CODE_DIR.glob("*.parquet"))
    for f in files:
        ds = load_dataset("parquet", data_files=str(f), split="train")
        for row in ds:
            for k in ("content", "code", "text"):
                if k in row and row[k]:
                    yield row[k]
                    break

def all_text():
    print("Loading eng...")
    yield from read_text(ENG, SAMPLE_BYTES)
    print("Loading math...")
    yield from read_text(MATH, SAMPLE_BYTES // 2)
    print("Loading code...")
    yield from read_code_parquets()

def main():
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        show_progress=True,
    )
    print("Training BPE...")
    tok.train_from_iterator(all_text(), trainer=trainer)
    OUT.mkdir(parents=True, exist_ok=True)
    tok.save(str(OUT / "tokenizer.json"))
    print('Saved tokenizer.json')
    print(f"Vocab size: {tok.get_vocab_size()}")

if __name__ == "__main__":
    main()

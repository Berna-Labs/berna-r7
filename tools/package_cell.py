#!/usr/bin/env python3
import argparse, json, sqlite3, tarfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/data/berna-r7")
DB = ROOT / "registry" / "cells.db"

def pack(src, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = out_dir / f"{name}-{ts}.tar.gz"
    with tarfile.open(out, "w:gz") as t:
        for f in src.rglob("*"):
            if f.is_file():
                t.add(f, arcname=f.relative_to(src.parent))
    print(f"Packed: {out.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mother", action="store_true")
    ap.add_argument("--out", default="/data/berna-r7/exports")
    a = ap.parse_args()
    out = Path(a.out)
    conn = sqlite3.connect(DB)
    if a.mother:
        r = conn.execute("SELECT uuid, path FROM mother LIMIT 1").fetchone()
        if not r: print("No mother"); return
        pack(Path(r[1]), out, f"berna-r7-mother-{r[0][:8]}")
        return
    if a.all:
        rows = conn.execute("SELECT uuid, domain, path FROM cells WHERE status=\u0027active\u0027").fetchall()
    elif a.uuid:
        rows = conn.execute("SELECT uuid, domain, path FROM cells WHERE uuid=?", (a.uuid,)).fetchall()
    else:
        print("Use --uuid, --all, or --mother"); return
    for uuid_, dom, path in rows:
        pack(Path(path), out, f"berna-r7-{dom}-{uuid_[:8]}")
    conn.close()

if __name__ == "__main__":
    main()

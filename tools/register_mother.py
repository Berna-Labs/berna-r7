#!/usr/bin/env python3
import hashlib, sqlite3, uuid as u
from datetime import datetime, timezone as tz
from pathlib import Path

ROOT = Path("/data/berna-r7")
DB = ROOT / "registry" / "cells.db"
CFG = ROOT / "core" / "config" / "config.json"

def main():
    h = hashlib.sha256(CFG.read_bytes()).hexdigest()
    conn = sqlite3.connect(DB)
    if conn.execute("SELECT 1 FROM mother LIMIT 1").fetchone():
        print("Mother already registered")
        conn.close()
        return
    conn.execute(
        "INSERT INTO mother VALUES (?,?,?,?,?,?)",
        (str(u.uuid4()), "r7-core-v1", str(ROOT/"core"),
         datetime.now(tz.utc).isoformat(), None, h))
    conn.commit()
    conn.close()
    print("Mother registered")

if __name__ == "__main__":
    main()

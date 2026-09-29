#!/usr/bin/env python3
import hashlib, json, sqlite3, sys, uuid as u
from datetime import datetime, timezone as tz
from pathlib import Path

ROOT = Path("/data/berna-r7")
CELLS = ROOT / "cells"
DB = ROOT / "registry" / "cells.db"

def h256(b): return hashlib.sha256(b).hexdigest()

def discover():
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT uuid FROM mother LIMIT 1").fetchone()
    if not row:
        print("[FATAL] No mother"); sys.exit(1)
    mother = row[0]
    print(f"Mother: {mother}")
    found = 0
    for mf in sorted(CELLS.rglob("manifest.json")):
        try:
            m = json.loads(mf.read_text())
        except Exception as e:
            print(f"[SKIP] {mf}: {e}"); continue
        d = mf.parent
        cfg = d / "config.json"
        if not cfg.exists():
            print(f"[SKIP] {d}: no config"); continue
        cid = m.get("uuid") or str(u.uuid4())
        ch = h256(cfg.read_bytes())
        mh = h256(mf.read_bytes())
        dom = m.get("domain", "unknown")
        ver = m.get("version", "v1")
        dk = h256(f"{dom}|{ver}|{ch}".encode())
        now = datetime.now(tz.utc).isoformat()
        exists = conn.execute("SELECT 1 FROM cells WHERE uuid=?", (cid,)).fetchone()
        if exists:
            conn.execute("UPDATE cells SET domain=?,domain_key=?,path=?,config_hash=?,manifest_hash=? WHERE uuid=?", (dom,dk,str(d),ch,mh,cid))
            print(f"[UPD] {cid[:8]} {dom}")
        else:
            conn.execute("INSERT INTO cells VALUES (?,?,?,?,?,?,?,?,?,?,?)", (cid,mother,m.get("name",d.name),dom,dk,str(d),"active",now,m.get("param_count"),ch,mh))
            print(f"[NEW] {cid[:8]} {dom}")
        bt = h256(f"{mother}|{cid}|{dk}".encode())[:32]
        conn.execute("DELETE FROM bindings WHERE cell_uuid=?", (cid,))
        conn.execute("INSERT INTO bindings (mother_uuid,cell_uuid,binding_token,created_at) VALUES (?,?,?,?)", (mother,cid,bt,now))
        print(f"      binding: {bt}")
        found += 1
    conn.commit()
    conn.close()
    print(f"Total: {found}")

if __name__ == "__main__":
    discover()

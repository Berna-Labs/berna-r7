#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path

T = Path("/data/berna-r7/tools")

def run(s):
    print(f"\n>> {s.name}")
    return subprocess.run([sys.executable, str(s)]).returncode

def main():
    print("=" * 50)
    print("  Berna R7 - Boot")
    print("=" * 50)
    run(T / "register_mother.py")
    run(T / "cell_discovery.py")
    print("\n" + "=" * 50)
    print("  R7 Ready")
    print("=" * 50)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
start.py — AMD Adapt Kiosk launcher
Just run:  python start.py  (from anywhere inside the repo)
It will auto-activate the venv and launch the server.
"""
import subprocess
import sys
import os
import webbrowser
import threading
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
VENV_PY = ROOT / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python3")
IN_VENV = os.environ.get("VIRTUAL_ENV") is not None or \
          Path(sys.executable).resolve() == VENV_PY.resolve()

if not IN_VENV:
    # Re-launch this script inside the venv automatically
    if not VENV_PY.exists():
        print("[ERROR] venv not found. Run:  bash setup_ubuntu.sh")
        sys.exit(1)
    print("[..] Activating venv and launching kiosk...")
    os.chdir(ROOT)
    os.execv(str(VENV_PY), [str(VENV_PY), str(__file__)])
    # execv replaces the current process — nothing below runs

# ── We're inside the venv from here ──────────────────────────────────────────
os.chdir(ROOT)

def open_browser():
    time.sleep(2.5)
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass

print("\n" + "="*60)
print("  AMD ADAPT KIOSK")
print("  http://localhost:8000")
print("="*60 + "\n")

threading.Thread(target=open_browser, daemon=True).start()

import uvicorn
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)

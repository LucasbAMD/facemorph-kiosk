#!/usr/bin/env python3
"""
start.py — Launch FaceMorph Kiosk
Run:  python start.py
Then open:  http://localhost:8000
"""
import subprocess
import sys
import os
import webbrowser
import threading
import time
from pathlib import Path

ROOT    = Path(__file__).parent
VENV_PY = ROOT / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

if not VENV_PY.exists():
    print("  [ERROR] venv not found. Run  python install.py  first.")
    sys.exit(1)

def open_browser():
    time.sleep(2.5)
    webbrowser.open("http://localhost:8000")

print()
print("  🎭  FaceMorph Kiosk")
print("  ─────────────────────────────────────")
print("  Starting server...")
print("  Browser will open at http://localhost:8000")
print("  Press Ctrl+C to stop")
print()

threading.Thread(target=open_browser, daemon=True).start()
os.chdir(ROOT)
subprocess.run([str(VENV_PY), "main.py"])

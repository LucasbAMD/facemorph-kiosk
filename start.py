#!/usr/bin/env python3
"""
start.py — Launch AMD Adapt Kiosk
Usage (recommended):
  cd ~/facemorph-kiosk && source venv/bin/activate && python start.py

Or via venv directly:
  ~/facemorph-kiosk/venv/bin/python start.py
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

# If we're already inside the venv, just run directly
IN_VENV = (Path(sys.executable).resolve() == VENV_PY.resolve()) or \
          (os.environ.get("VIRTUAL_ENV") is not None)

def open_browser():
    time.sleep(2.5)
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass

if IN_VENV:
    # Already in venv — run main directly
    threading.Thread(target=open_browser, daemon=True).start()
    os.chdir(ROOT)
    import uvicorn
    print("\n" + "="*60)
    print("  AMD ADAPT KIOSK")
    print("  http://localhost:8000")
    print("="*60 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
else:
    if not VENV_PY.exists():
        print("[ERROR] venv not found. Run:  bash setup_ubuntu.sh")
        sys.exit(1)
    threading.Thread(target=open_browser, daemon=True).start()
    os.chdir(ROOT)
    result = subprocess.run([str(VENV_PY), "-m", "uvicorn", "main:app",
                             "--host", "0.0.0.0", "--port", "8000",
                             "--workers", "1"])
    sys.exit(result.returncode)

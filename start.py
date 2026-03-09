#!/usr/bin/env python3
"""start.py — run via: bash start.sh"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

print("\n" + "="*60)
print("  AMD ADAPT KIOSK — http://localhost:8000")
print("="*60 + "\n")

import uvicorn
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)

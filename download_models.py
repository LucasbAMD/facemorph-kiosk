#!/usr/bin/env python3
"""
download_models.py — Download required models for AMD Adapt Kiosk
Run:  python download_models.py

Downloads inswapper_128.onnx (~526 MB) from HuggingFace.
Uses Python requests to handle redirects correctly.
"""
import sys
import os
from pathlib import Path

def download(url, dest, chunk_size=1024*1024):
    try:
        import requests
    except ImportError:
        print("[ERR] requests not installed. Run: pip install requests")
        sys.exit(1)

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[..] Downloading to {dest}")
    print(f"     URL: {url}\n")

    with requests.get(url, stream=True, timeout=60,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        mb  = done / 1024 / 1024
                        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                        print(f"\r  [{bar}] {pct:5.1f}%  {mb:.1f} MB", end="", flush=True)
    print()
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"\n[OK] Downloaded {size_mb:.1f} MB → {dest}")
    if size_mb < 100:
        print("[WARN] File seems too small — may be corrupted. Re-run this script.")
        sys.exit(1)
    print("[OK] File size looks correct!")
    return True

if __name__ == "__main__":
    # Try HuggingFace with ?download=true param
    url  = "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx?download=true"
    dest = "models/inswapper_128.onnx"

    if Path(dest).exists():
        size_mb = Path(dest).stat().st_size / 1024 / 1024
        if size_mb > 400:
            print(f"[OK] {dest} already exists ({size_mb:.0f} MB) — skipping download.")
            sys.exit(0)
        else:
            print(f"[WARN] Existing file looks corrupted ({size_mb:.1f} MB) — re-downloading...")
            os.remove(dest)

    try:
        download(url, dest)
        print("\n✅  Model ready. Run python start.py\n")
    except Exception as e:
        print(f"\n[ERR] Download failed: {e}")
        print("Try manually downloading from:")
        print("  https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx")
        print("and place the file at:  models/inswapper_128.onnx")
        sys.exit(1)

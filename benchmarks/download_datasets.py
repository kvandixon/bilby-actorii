#!/usr/bin/env python3
"""Download official LongMemEval (ICLR 2025) & LOCOMO (Snap Research) benchmark datasets.

Usage:
    python3 benchmarks/download_datasets.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "longmem"

DATASETS = {
    "longmemeval_oracle.json": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"
    ),
    "locomo10.json": (
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    ),
}


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in DATASETS.items():
        target = DATA_DIR / filename
        if target.exists() and target.stat().st_size > 1000:
            print(f"[OK] {filename} already present ({target.stat().st_size / (1024 * 1024):.1f} MB)")
            continue
        print(f"[DOWNLOADING] {filename} from {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Bilby-ActorII-Bench/4.1"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(target, "wb") as out:
            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                out.write(chunk)
        print(f"[DONE] Saved {target} ({target.stat().st_size / (1024 * 1024):.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

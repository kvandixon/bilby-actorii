#!/usr/bin/env python3
"""Top-level launcher for Bilby's ActorII v4.1 Telegram Gateway."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from actorii.telegram_gateway import main

if __name__ == "__main__":
    raise SystemExit(main())

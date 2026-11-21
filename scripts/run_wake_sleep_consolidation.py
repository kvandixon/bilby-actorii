#!/usr/bin/env python3
"""Idle Wake-Sleep Consolidation Runner for Bilby (0 MB GPU VRAM).

Invoked periodically by `bilby-consolidate.timer` (or manually via CLI) to run
`ConsolidatorBridge.runConsolidationCycle()` over recent Neo4j `:ActorEpisode`
traces using `stitch-core` + `egglog` and append structured telemetry to
`~/.actorii/logs/wake_sleep.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from actorii.classes.dictionary import Dictionary
from actorii.core import Symbol
from actorii.harness.telemetry import resolve_actorii_state_dir
from actorii.harness.wake_sleep import ConsolidatorBridge
from actorii.vm import ActorVM


def _to_plain(val):
    if isinstance(val, Dictionary):
        raw = getattr(val, "_entries", getattr(val, "_data", {}))
        return {
            (k.name if isinstance(k, Symbol) else str(k)): _to_plain(v)
            for k, v in raw.items()
        }
    if hasattr(val, "_items") and not isinstance(val, (str, bytes, dict)):
        return [_to_plain(x) for x in val._items]
    if isinstance(val, (list, tuple)):
        return [_to_plain(x) for x in val]
    if isinstance(val, Symbol):
        return val.name
    return val


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Bilby wake-sleep consolidation cycle (0 MB GPU VRAM).")
    parser.add_argument("--max-episodes", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--namespace", default="default")
    args = parser.parse_args()

    t0 = time.perf_counter()
    vm = ActorVM()
    bridge = ConsolidatorBridge(vm=vm, namespace=args.namespace)
    res = _to_plain(bridge.runConsolidationCycle(max_episodes=args.max_episodes, dry_run=args.dry_run))
    record = {
        "timestamp": round(time.time(), 3),
        "namespace": args.namespace,
        "dry_run": bool(args.dry_run),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "result": res,
    }

    log_dir = resolve_actorii_state_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "wake_sleep.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    print(json.dumps(record, default=str))
    return 0 if bool(res.get("ok", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())

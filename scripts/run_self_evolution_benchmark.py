#!/usr/bin/env python3
"""CLI for the Bilby Self-Evolution Loop Benchmark (`actorii.harness.self_evolution_bench`).

Examples:
    # deterministic CI run of every built-in spec (no LLM, ~2s)
    PYTHONPATH=src python scripts/run_self_evolution_benchmark.py --mode scripted --specs all

    # live Bilby against the local llama-server (temperature 0.0, seed 42)
    PYTHONPATH=src python scripts/run_self_evolution_benchmark.py --mode live \\
        --specs ring_buffer_telemetry,circuit_breaker_policy --output-json /tmp/selfevo_live.json

    # user-defined spec(s)
    PYTHONPATH=src python scripts/run_self_evolution_benchmark.py --spec-file my_spec.json --specs my_spec_id

Isolation: every run uses throw-away `ACTORII_STATE_DIR` / `ACTORII_MEMORY_NAMESPACE` values and
fresh in-memory VMs. It never touches `~/.actorii`.

Exit codes: 0 ok, 1 `--strict` and at least one spec failed, 2 usage / spec error / run error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from actorii.harness.self_evolution_bench import (  # noqa: E402
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    SelfEvolutionRunner,
    list_self_evolution_suites,
    render_markdown,
    resolve_specs,
    summarize_result,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ActorII Bilby self-evolution loop benchmark")
    p.add_argument("--mode", choices=("scripted", "live"), default="scripted")
    p.add_argument("--specs", default="all", help="'all' or comma separated spec ids (default: all)")
    p.add_argument("--spec-file", action="append", default=[], help="JSON spec file (repeatable); its specs join --specs")
    p.add_argument("--max-generations", type=int, default=None, help="override each spec's max_generations")
    p.add_argument("--output-json", default=None, help="write the JSON scoreboard here")
    p.add_argument("--output-md", default=None, help="write the Markdown scoreboard here")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"live OpenAI-compatible endpoint (default {DEFAULT_ENDPOINT})")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"live model name (default {DEFAULT_MODEL})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--persistence-backend", choices=("fileout", "engram"), default="fileout",
                   help="Phase-3 reload gate: `.act` file-out (default) or Neo4j `:ActorEngram` (isolated namespace)")
    p.add_argument("--work-dir", default=None, help="keep artifacts (file-outs, runtime state) in this directory")
    p.add_argument("--allow-host", action="store_true", help="do NOT force ACTORII_HOST_LOCKDOWN=1 during the run")
    p.add_argument("--llm-timeout", type=float, default=300.0)
    p.add_argument("--list", action="store_true", help="list built-in specs and exit")
    p.add_argument("--strict", action="store_true", help="exit 1 if any spec did not pass")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print(list_self_evolution_suites())
        return 0
    try:
        specs = resolve_specs(args.specs, args.spec_file)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2

    log = (lambda msg: None) if args.quiet else (lambda msg: print(msg, flush=True))
    try:
        runner = SelfEvolutionRunner(
            args.mode,
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
            seed=args.seed,
            max_generations=args.max_generations,
            work_dir=args.work_dir,
            host_lockdown=not args.allow_host,
            persistence_backend=args.persistence_backend,
            llm_timeout=args.llm_timeout,
            on_event=log,
        )
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    with runner:
        report = runner.run_suite(specs)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    md = render_markdown(report)
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")

    if not args.quiet:
        print()
        for r in report["specs"]:
            print(summarize_result(r))
        s = report["summary"]
        print(
            f"\nSUMMARY pass@1={s['pass_at_1_count']}/{s['specs']} pass@G={s['pass_at_g_count']}/{s['specs']} "
            f"persistence={s['persistence_verified']} rollbacks={s['rollbacks']} errors={s['errors']} "
            f"steps={s['steps_used']} gas={s['gas_used']} latency={s.get('total_latency_s')}s"
        )
    if report["summary"]["errors"]:
        return 2
    if args.strict and report["summary"]["pass_at_g_count"] != report["summary"]["specs"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

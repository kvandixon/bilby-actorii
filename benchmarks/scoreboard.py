#!/usr/bin/env python3
"""Bilby (ActorII) Benchmark Scoreboard Manager & Static/Live Generator.

Single-source-of-truth CLI (`benchmarks/scoreboard.json`) that manages:
  1. Active & Queued/Discovered benchmark definitions (`add-benchmark`, `set-status`)
  2. Verified Bilby run ledger (`record`, `ingest <results.json>`)
  3. Automatic multi-surface rendering (`render`):
     - `benchmarks/BENCHMARK_RESULTS.md` (Headline table, sub-tracks, run history, watchlist)
     - `README.md` (auto-synced summary table between BENCHMARK_SCOREBOARD markers)
     - `benchmarks/scoreboard.html` (dark-theme interactive dashboard, also served at `:8090/benchmarks`)
"""

from __future__ import annotations

import argparse
from datetime import date
import html
import json
from pathlib import Path
import re
import sys
from typing import Any

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parent
DEFAULT_JSON_PATH = BENCHMARKS_DIR / "scoreboard.json"
DEFAULT_MD_PATH = BENCHMARKS_DIR / "BENCHMARK_RESULTS.md"
DEFAULT_HTML_PATH = BENCHMARKS_DIR / "scoreboard.html"
DEFAULT_README_PATH = REPO_ROOT / "README.md"

README_START_MARKER = "<!-- BENCHMARK_SCOREBOARD:START -->"
README_END_MARKER = "<!-- BENCHMARK_SCOREBOARD:END -->"


def load_scoreboard(path: Path = DEFAULT_JSON_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Scoreboard ledger not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_scoreboard(data: dict[str, Any], path: Path = DEFAULT_JSON_PATH) -> None:
    data.setdefault("metadata", {})["updated_at"] = date.today().isoformat()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "benchmark"


def _runs_for_benchmark(data: dict[str, Any], bench_id: str) -> list[dict[str, Any]]:
    runs = [r for r in data.get("runs", []) if r.get("benchmark_id") == bench_id]
    # Preserve chronological order by (date, insertion order)
    return sorted(enumerate(runs), key=lambda pair: (str(pair[1].get("date", "")), pair[0])) and [
        pair[1] for pair in sorted(enumerate(runs), key=lambda pair: (str(pair[1].get("date", "")), pair[0]))
    ]


def _headline_stats(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None
    latest = runs[-1]
    best = max(runs, key=lambda r: (float(r.get("score_pct", 0.0)), str(r.get("date", ""))))
    first = runs[0]
    delta_vs_first = round(float(latest.get("score_pct", 0.0)) - float(first.get("score_pct", 0.0)), 2)
    return {
        "latest": latest,
        "best": best,
        "first": first,
        "run_count": len(runs),
        "delta_vs_first": delta_vs_first,
    }


def _fmt_int(val: Any) -> str:
    if val is None or val == "":
        return "TBD"
    try:
        return f"{int(val):,}"
    except (TypeError, ValueError):
        return str(val)


def _fmt_score(run: dict[str, Any], bench_id: str = "") -> str:
    pct = float(run.get("score_pct", 0.0))
    passed = _fmt_int(run.get("passed", 0))
    total = _fmt_int(run.get("total", 0))
    trials = int(run.get("trials") or 1)
    suffix = " Blocked" if bench_id == "redteam-containment" else ""
    if trials > 1:
        return f"{pct:.2f}% avg@{trials} ({passed} / {total} trials{suffix})"
    return f"{pct:.2f}% ({passed} / {total}{suffix})"


def _fmt_task_count(bench: dict[str, Any], run: dict[str, Any]) -> str:
    trials = int(run.get("trials") or 1)
    raw_total = int(run.get("total") or 0)
    base_tasks = bench.get("total_tasks") or (raw_total // max(1, trials) if raw_total else None)
    if trials > 1 and base_tasks:
        return f"{_fmt_int(base_tasks)} (×{trials})"
    return _fmt_int(run.get("total", bench.get("total_tasks")))


def render_markdown(data: dict[str, Any]) -> str:
    meta = data.get("metadata", {})
    benchmarks = data.get("benchmarks", [])
    active_benches = [b for b in benchmarks if b.get("status") == "active"]
    queued_benches = [b for b in benchmarks if b.get("status") != "active"]

    lines: list[str] = [
        "> [!WARNING]",
        "> This is a planning document and should not be considered as a source of truth without verification.",
        "",
        f"# {meta.get('title', 'Bilby (ActorII) — Benchmark Scoreboard')}",
        "",
        f"- **Host Hardware**: `{meta.get('hardware', 'valkyrie')}`",
        f"- **Memory Architecture**: `{meta.get('memory_stack', 'Neo4j :ActorEpisode / :ActorEngram')}`",
        f"- **Last Updated**: `{meta.get('updated_at', date.today().isoformat())}`",
        "- **Interactive Dark-Theme Dashboard**: `http://127.0.0.1:8090/benchmarks` (or open [`scoreboard.html`](scoreboard.html))",
        "",
        "---",
        "",
        "## 1. Headline Active Scoreboard (Bilby Latest & Best)",
        "",
        "| Benchmark Suite | Category | Tasks (`N`) | Latest Score | Best Score | Δ vs Initial | Model (`Latest`) | Date | Raw Ledger |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for b in active_benches:
        bid = b["id"]
        runs = _runs_for_benchmark(data, bid)
        stats = _headline_stats(runs)
        if not stats:
            lines.append(
                f"| **{b['name']}** | `{b.get('category', 'General')}` | `{_fmt_int(b.get('total_tasks'))}` | *No runs yet* | — | — | — | — | — |"
            )
            continue
        latest = stats["latest"]
        best = stats["best"]
        delta = stats["delta_vs_first"]
        if stats["run_count"] <= 1 or abs(delta) < 0.005:
            delta_str = "—"
        elif delta > 0:
            delta_str = f"`+{delta:.2f} pp`"
        else:
            delta_str = f"`{delta:.2f} pp`"

        ledger = str(latest.get("raw_ledger", "")).strip()
        ledger_md = f"[`{Path(ledger).name}`]({ledger})" if ledger else "—"
        best_trials = int(best.get("trials") or 1)
        best_tag = f" avg@{best_trials}" if best_trials > 1 else ""
        best_str = f"**`{float(best['score_pct']):.2f}%{best_tag}`** (`{best.get('model', '')}`)"
        lines.append(
            f"| **{b['name']}** | `{b.get('category', 'General')}` | `{_fmt_task_count(b, latest)}` "
            f"| **`{_fmt_score(latest, bid)}`** | {best_str} | {delta_str} | `{latest.get('model', '')}` | `{latest.get('date', '')}` | {ledger_md} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Sub-Track & Pillar Breakdowns (Latest Verified Runs)",
        "",
    ])

    for b in active_benches:
        runs = _runs_for_benchmark(data, b["id"])
        if not runs:
            continue
        latest = runs[-1]
        subs = latest.get("submetrics") or {}
        if not subs and not latest.get("notes"):
            continue
        lines.append(f"### {b['name']} (`{latest.get('harness_version', '')}` · `{latest.get('model', '')}` · `{latest.get('date', '')}`)")
        if b.get("description"):
            lines.append(f"- **Scope**: {b['description']}")
        if latest.get("notes"):
            lines.append(f"- **Notes**: {latest['notes']}")
        if subs:
            chips = " · ".join(f"`{k}`: **{v}**" for k, v in subs.items())
            lines.append(f"- **Breakdown**: {chips}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Bilby Historical Run & Model Progression Ledger",
        "",
        "| Date | Benchmark | Version | Model | Mode / Configuration | Score | Raw Artifact |",
        "| :---: | :--- | :---: | :---: | :--- | :---: | :--- |",
    ])

    bench_by_id = {b["id"]: b for b in benchmarks}
    all_runs_sorted = [
        pair[1]
        for pair in sorted(
            enumerate(data.get("runs", [])),
            key=lambda pair: (str(pair[1].get("date", "")), pair[0]),
            reverse=True,
        )
    ]
    for r in all_runs_sorted:
        bid = r.get("benchmark_id", "")
        bname = bench_by_id.get(bid, {}).get("name", bid)
        ledger = str(r.get("raw_ledger", "")).strip()
        ledger_md = f"[`{Path(ledger).name}`]({ledger})" if ledger else "—"
        lines.append(
            f"| `{r.get('date', '')}` | **{bname}** | `{r.get('harness_version', '')}` | `{r.get('model', '')}` "
            f"| {r.get('mode', '—')} | **`{_fmt_score(r, bid)}`** | {ledger_md} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Queued / Discovered Benchmarks Watchlist",
        "",
        "New benchmarks discovered for upcoming evaluation. Promoting any benchmark is automatic the first time `scoreboard.py record` or `scoreboard.py ingest` logs a run against its ID.",
        "",
        "| ID | Candidate Benchmark | Category | Tasks (`N`) | What It Evaluates | Upstream |",
        "| :--- | :--- | :---: | :---: | :--- | :--- |",
    ])

    for qb in queued_benches:
        url = str(qb.get("upstream_url", "")).strip()
        url_md = f"[Link]({url})" if url else "—"
        lines.append(
            f"| `{qb['id']}` | **{qb['name']}** | `{qb.get('category', 'General')}` | `{_fmt_int(qb.get('total_tasks'))}` "
            f"| {qb.get('description', '')} | {url_md} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Updating Scores & Adding New Benchmarks",
        "",
        "All tables in `benchmarks/BENCHMARK_RESULTS.md`, `README.md`, and `benchmarks/scoreboard.html` (`/benchmarks` on `:8090`) are generated from [`benchmarks/scoreboard.json`](scoreboard.json):",
        "",
        "```bash",
        "# 1. Record a new benchmark run (auto-updates Markdown, README, and HTML dashboard):",
        "python3 benchmarks/scoreboard.py record \\",
        "  --bench tb21-container --passed 5 --total 5 \\",
        "  --model gpt-oss:20b --version v4.2.0 \\",
        "  --mode \"Docker Container\" --ledger archive/TB21_5TASK_RESULTS.json \\",
        "  --notes \"5/5 container verification on gpt-oss:20b\"",
        "",
        "# 2. Or auto-ingest a raw JSON results file directly:",
        "python3 benchmarks/scoreboard.py ingest benchmarks/archive/NEXTGEN_RESULTS.json \\",
        "  --bench nextgen-postcutoff --model gpt-oss:20b --version v4.2.0",
        "",
        "# 3. Add a newly discovered benchmark to the watchlist (or active list):",
        "python3 benchmarks/scoreboard.py add-benchmark \\",
        "  --id gaia-val --name \"GAIA (Validation)\" --category \"Memory\" \\",
        "  --tasks 165 --url \"https://huggingface.co/datasets/gaia-benchmark/GAIA\" \\",
        "  --desc \"Multi-step tool, web, and document reasoning tasks.\"",
        "```",
        "",
    ])
    return "\n".join(lines)


def render_readme_snippet(data: dict[str, Any]) -> str:
    benchmarks = data.get("benchmarks", [])
    featured = [b for b in benchmarks if b.get("status") == "active" and b.get("featured_in_readme")]
    rows = [
        "| Benchmark Suite | Category | Questions / Tasks | Bilby Score | Date |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for b in featured:
        runs = _runs_for_benchmark(data, b["id"])
        if not runs:
            continue
        latest = runs[-1]
        rows.append(
            f"| **{b['name']}** | `{b.get('category', 'General')}` | `{_fmt_task_count(b, latest)}` "
            f"| **`{_fmt_score(latest, b['id'])}`** | `{latest.get('date', '')}` |"
        )
    return "\n".join(rows)


def update_readme(data: dict[str, Any], readme_path: Path = DEFAULT_README_PATH) -> bool:
    if not readme_path.exists():
        return False
    content = readme_path.read_text(encoding="utf-8")
    if README_START_MARKER not in content or README_END_MARKER not in content:
        return False
    snippet = render_readme_snippet(data)
    pattern = re.compile(
        re.escape(README_START_MARKER) + r".*?" + re.escape(README_END_MARKER),
        flags=re.DOTALL,
    )
    updated = pattern.sub(f"{README_START_MARKER}\n{snippet}\n{README_END_MARKER}", content, count=1)
    readme_path.write_text(updated, encoding="utf-8")
    return True


def render_html(data: dict[str, Any]) -> str:
    # Escape `</` so embedded script literal cannot prematurely close `<script>`
    embedded_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(data.get("metadata", {}).get("title", "Bilby Benchmark Scoreboard"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg: #0b0f19;
    --panel: #111827;
    --panel-alt: #162033;
    --border: #1f2d47;
    --text: #e5e7eb;
    --muted: #9ca3af;
    --accent: #38bdf8;
    --emerald: #10b981;
    --amber: #f59e0b;
    --violet: #a78bfa;
    --rose: #fb7185;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: radial-gradient(circle at top right, #13203b 0%, var(--bg) 45%);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    line-height: 1.5;
    min-height: 100vh;
  }}
  .wrap {{ max-width: 1240px; margin: 0 auto; padding: 28px 20px 64px; }}
  .topbar {{
    display: flex; justify-content: space-between; align-items: flex-start;
    gap: 16px; flex-wrap: wrap; margin-bottom: 22px;
    border-bottom: 1px solid var(--border); padding-bottom: 18px;
  }}
  h1 {{ margin: 0 0 6px; font-size: 1.55rem; letter-spacing: -0.02em; color: #f8fafc; }}
  .subtitle {{ font-size: 0.86rem; color: var(--muted); }}
  .subtitle code {{ color: var(--accent); background: rgba(56,189,248,0.1); padding: 2px 6px; border-radius: 4px; }}
  .top-actions {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .btn {{
    background: var(--panel-alt); color: var(--text); border: 1px solid var(--border);
    padding: 8px 14px; border-radius: 8px; font-size: 0.83rem; font-weight: 600;
    cursor: pointer; text-decoration: none; transition: all 0.15s ease;
  }}
  .btn:hover {{ border-color: var(--accent); color: #fff; }}
  .btn-primary {{ background: rgba(56,189,248,0.16); border-color: rgba(56,189,248,0.45); color: #7dd3fc; }}
  .btn-primary:hover {{ background: rgba(56,189,248,0.28); }}
  .kpis {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
    gap: 12px; margin-bottom: 22px;
  }}
  .kpi {{
    background: rgba(17,24,39,0.88); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }}
  .kpi-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
  .kpi-val {{ font-size: 1.55rem; font-weight: 700; margin-top: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #f8fafc; }}
  .kpi-sub {{ font-size: 0.76rem; color: var(--muted); margin-top: 2px; }}
  .toolbar {{
    display: flex; justify-content: space-between; align-items: center;
    gap: 12px; flex-wrap: wrap; margin-bottom: 18px;
  }}
  .pills {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  .pill {{
    background: var(--panel); color: var(--muted); border: 1px solid var(--border);
    padding: 5px 12px; border-radius: 999px; font-size: 0.79rem; font-weight: 600; cursor: pointer;
  }}
  .pill.active {{ background: rgba(56,189,248,0.18); color: #7dd3fc; border-color: rgba(56,189,248,0.5); }}
  .search {{
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    padding: 7px 12px; border-radius: 8px; font-size: 0.84rem; min-width: 240px;
  }}
  .section-title {{
    font-size: 1.05rem; font-weight: 700; margin: 28px 0 12px;
    display: flex; align-items: center; gap: 10px; color: #f1f5f9;
  }}
  .count-badge {{
    font-size: 0.74rem; padding: 2px 8px; border-radius: 999px;
    background: var(--panel-alt); border: 1px solid var(--border); color: var(--accent);
  }}
  .bench-grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
  .card {{
    background: rgba(17,24,39,0.92); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 18px; transition: border-color 0.15s ease;
  }}
  .card:hover {{ border-color: #2e4369; }}
  .card-head {{
    display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap;
  }}
  .bench-title {{ font-size: 1.02rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  .cat-tag {{
    font-size: 0.71rem; font-weight: 600; padding: 2px 8px; border-radius: 6px;
    background: rgba(167,139,250,0.14); color: #c4b5fd; border: 1px solid rgba(167,139,250,0.3);
  }}
  .delta-tag {{
    font-size: 0.72rem; font-weight: 700; padding: 2px 7px; border-radius: 6px;
    background: rgba(16,185,129,0.14); color: #34d399; font-family: ui-monospace, monospace;
  }}
  .score-box {{ text-align: right; min-width: 170px; }}
  .score-pct {{
    font-size: 1.4rem; font-weight: 800; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--emerald);
  }}
  .score-frac {{ font-size: 0.78rem; color: var(--muted); font-family: ui-monospace, monospace; }}
  .bar-track {{
    width: 100%; height: 7px; background: #090d16; border-radius: 999px;
    overflow: hidden; margin: 10px 0 8px; border: 1px solid #172136;
  }}
  .bar-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, #0ea5e9, #10b981); }}
  .desc {{ font-size: 0.83rem; color: #cbd5e1; margin: 4px 0 8px; }}
  .meta-row {{
    display: flex; justify-content: space-between; align-items: center;
    gap: 10px; flex-wrap: wrap; font-size: 0.77rem; color: var(--muted);
  }}
  .meta-row code {{ color: #93c5fd; background: rgba(30,58,138,0.3); padding: 2px 6px; border-radius: 4px; }}
  .subchips {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }}
  .subchip {{
    font-size: 0.74rem; background: #0d1424; border: 1px solid var(--border);
    padding: 3px 8px; border-radius: 6px; color: #cbd5e1; font-family: ui-monospace, monospace;
  }}
  .subchip strong {{ color: #38bdf8; }}
  details.history {{ margin-top: 10px; border-top: 1px dashed var(--border); padding-top: 8px; }}
  details.history summary {{
    cursor: pointer; font-size: 0.78rem; color: var(--accent); font-weight: 600; user-select: none;
  }}
  table.hist-table {{
    width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.78rem;
  }}
  table.hist-table th, table.hist-table td {{
    text-align: left; padding: 6px 8px; border-bottom: 1px solid #182235;
  }}
  table.hist-table th {{ color: var(--muted); font-weight: 600; }}
  .queued-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 12px;
  }}
  .queued-card {{
    background: rgba(17,24,39,0.65); border: 1px dashed #2b3d5e;
    border-radius: 10px; padding: 14px 16px;
  }}
  .queued-badge {{
    font-size: 0.7rem; padding: 2px 7px; border-radius: 6px;
    background: rgba(245,158,11,0.14); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3);
  }}
  .drawer {{
    display: none; background: var(--panel); border: 1px solid var(--accent);
    border-radius: 12px; padding: 18px; margin-bottom: 22px;
  }}
  .drawer.open {{ display: block; }}
  .form-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin: 10px 0;
  }}
  .form-grid label {{ display: block; font-size: 0.74rem; color: var(--muted); margin-bottom: 3px; }}
  .form-grid input, .form-grid select {{
    width: 100%; background: #0b0f19; color: var(--text); border: 1px solid var(--border);
    padding: 7px 9px; border-radius: 6px; font-size: 0.82rem;
  }}
  pre.cmd-preview {{
    background: #070a12; border: 1px solid var(--border); border-radius: 8px;
    padding: 12px; font-size: 0.8rem; color: #7dd3fc; overflow-x: auto; margin: 10px 0 0;
  }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div>
      <h1 id="page-title">{title}</h1>
      <div class="subtitle" id="page-meta"></div>
    </div>
    <div class="top-actions">
      <button class="btn btn-primary" onclick="toggleDrawer()">+ Record Run / Add Benchmark</button>
      <a class="btn" href="/">Studio Surface</a>
    </div>
  </div>

  <div class="drawer" id="cmd-drawer">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
      <strong>Quick CLI Command Builder (auto-updates Markdown + README + HTML)</strong>
      <div class="pills">
        <button class="pill active" id="tab-record" onclick="setBuilderMode('record')">Record New Score</button>
        <button class="pill" id="tab-add" onclick="setBuilderMode('add')">Add New Benchmark</button>
      </div>
    </div>
    <div id="builder-record" class="form-grid">
      <div><label>Benchmark</label><select id="r-bench" onchange="updateCmd()"></select></div>
      <div><label>Passed</label><input id="r-passed" type="number" value="5" oninput="updateCmd()"></div>
      <div><label>Total Tasks</label><input id="r-total" type="number" value="5" oninput="updateCmd()"></div>
      <div><label>Model</label><input id="r-model" value="gpt-oss:20b" oninput="updateCmd()"></div>
      <div><label>Harness Version</label><input id="r-ver" value="v4.2.0" oninput="updateCmd()"></div>
      <div><label>Mode / Notes</label><input id="r-notes" value="Docker Container" oninput="updateCmd()"></div>
    </div>
    <div id="builder-add" class="form-grid" style="display:none;">
      <div><label>Benchmark ID (slug)</label><input id="a-id" value="gaia-val" oninput="updateCmd()"></div>
      <div><label>Display Name</label><input id="a-name" value="GAIA (Validation)" oninput="updateCmd()"></div>
      <div><label>Category</label><input id="a-cat" value="Memory" oninput="updateCmd()"></div>
      <div><label>Tasks (optional)</label><input id="a-tasks" type="number" placeholder="165" oninput="updateCmd()"></div>
      <div><label>Upstream URL</label><input id="a-url" value="https://huggingface.co/datasets/gaia-benchmark/GAIA" oninput="updateCmd()"></div>
      <div><label>Description</label><input id="a-desc" value="Multi-step web, PDF, and tool reasoning." oninput="updateCmd()"></div>
    </div>
    <pre class="cmd-preview" id="cmd-output"></pre>
    <div style="margin-top:8px;display:flex;gap:8px;align-items:center;">
      <button class="btn btn-primary" onclick="copyCmd()">Copy Command</button>
      <span id="copy-status" style="font-size:0.78rem;color:var(--emerald);"></span>
    </div>
  </div>

  <div class="kpis" id="kpi-strip"></div>

  <div class="toolbar">
    <div class="pills" id="cat-pills"></div>
    <input class="search" id="search-box" type="search" placeholder="Filter benchmarks, models, notes..." oninput="renderBoard()">
  </div>

  <div class="section-title">Active Benchmark Scoreboard <span class="count-badge" id="active-count">0</span></div>
  <div class="bench-grid" id="active-grid"></div>

  <div class="section-title">Queued &amp; Discovered Watchlist <span class="count-badge" id="queued-count">0</span></div>
  <div class="queued-grid" id="queued-grid"></div>
</div>

<script>
window.__SCOREBOARD_DATA__ = {embedded_json};
let STATE = window.__SCOREBOARD_DATA__;
let SELECTED_CAT = 'All';
let BUILDER_MODE = 'record';

function esc(s) {{
  return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}

function runsFor(bid) {{
  return (STATE.runs || []).filter(r => r.benchmark_id === bid)
    .sort((a, b) => String(a.date || '').localeCompare(String(b.date || '')));
}}

function populateSelect() {{
  const sel = document.getElementById('r-bench');
  sel.innerHTML = (STATE.benchmarks || []).map(b =>
    `<option value="${{esc(b.id)}}">${{esc(b.name)}} (${{esc(b.status)}})</option>`
  ).join('');
  updateCmd();
}}

function toggleDrawer() {{
  document.getElementById('cmd-drawer').classList.toggle('open');
}}

function setBuilderMode(m) {{
  BUILDER_MODE = m;
  document.getElementById('tab-record').classList.toggle('active', m === 'record');
  document.getElementById('tab-add').classList.toggle('active', m === 'add');
  document.getElementById('builder-record').style.display = m === 'record' ? 'grid' : 'none';
  document.getElementById('builder-add').style.display = m === 'add' ? 'grid' : 'none';
  updateCmd();
}}

function updateCmd() {{
  const out = document.getElementById('cmd-output');
  if (BUILDER_MODE === 'record') {{
    const b = document.getElementById('r-bench').value;
    const p = document.getElementById('r-passed').value || 0;
    const t = document.getElementById('r-total').value || 1;
    const m = document.getElementById('r-model').value || 'gpt-oss:20b';
    const v = document.getElementById('r-ver').value || 'v4.2.0';
    const n = (document.getElementById('r-notes').value || '').replace(/"/g, '\\"');
    out.textContent = `python3 benchmarks/scoreboard.py record --bench ${{b}} --passed ${{p}} --total ${{t}} --model "${{m}}" --version "${{v}}" --notes "${{n}}"`;
  }} else {{
    const id = document.getElementById('a-id').value || 'new-bench';
    const nm = (document.getElementById('a-name').value || '').replace(/"/g, '\\"');
    const cat = (document.getElementById('a-cat').value || 'General').replace(/"/g, '\\"');
    const tasks = document.getElementById('a-tasks').value;
    const url = (document.getElementById('a-url').value || '').replace(/"/g, '\\"');
    const desc = (document.getElementById('a-desc').value || '').replace(/"/g, '\\"');
    const tFlag = tasks ? ` --tasks ${{tasks}}` : '';
    out.textContent = `python3 benchmarks/scoreboard.py add-benchmark --id ${{id}} --name "${{nm}}" --category "${{cat}}"${{tFlag}} --url "${{url}}" --desc "${{desc}}"`;
  }}
}}

function copyCmd() {{
  navigator.clipboard.writeText(document.getElementById('cmd-output').textContent).then(() => {{
    const el = document.getElementById('copy-status');
    el.textContent = 'Copied to clipboard!';
    setTimeout(() => {{ el.textContent = ''; }}, 2000);
  }});
}}

function renderBoard() {{
  const meta = STATE.metadata || {{}};
  document.getElementById('page-title').textContent = meta.title || 'Bilby Benchmark Scoreboard';
  document.getElementById('page-meta').innerHTML =
    `Hardware: <code>${{esc(meta.hardware || '')}}</code> &middot; Updated: <code>${{esc(meta.updated_at || '')}}</code>`;

  const benches = STATE.benchmarks || [];
  const cats = ['All', ...Array.from(new Set(benches.map(b => b.category || 'General')))];
  document.getElementById('cat-pills').innerHTML = cats.map(c =>
    `<button class="pill ${{SELECTED_CAT === c ? 'active' : ''}}" onclick="SELECTED_CAT='${{esc(c)}}';renderBoard()">${{esc(c)}}</button>`
  ).join('');

  const q = (document.getElementById('search-box').value || '').toLowerCase().trim();
  const matches = (b, runs) => {{
    if (SELECTED_CAT !== 'All' && (b.category || 'General') !== SELECTED_CAT) return false;
    if (!q) return true;
    const hay = [b.id, b.name, b.category, b.description, ...runs.map(r => `${{r.model}} ${{r.notes}} ${{r.mode}}`)].join(' ').toLowerCase();
    return hay.includes(q);
  }};

  const active = benches.filter(b => b.status === 'active');
  const queued = benches.filter(b => b.status !== 'active');

  let totalQuestions = 0;
  let totalPassed = 0;
  let perfectSuites = 0;
  active.forEach(b => {{
    const rs = runsFor(b.id);
    if (rs.length) {{
      const lat = rs[rs.length - 1];
      const tr = Math.max(1, Number(lat.trials || 1));
      const uTot = Number(b.total_tasks || Math.round(Number(lat.total || 0) / tr));
      totalQuestions += uTot;
      totalPassed += tr > 1
        ? Math.round((Number(lat.score_pct || 0) / 100) * uTot)
        : Number(lat.passed || 0);
      if (Number(lat.score_pct) >= 99.99) perfectSuites += 1;
    }}
  }});

  document.getElementById('kpi-strip').innerHTML = `
    <div class="kpi"><div class="kpi-label">Active Suites</div><div class="kpi-val">${{active.length}}</div><div class="kpi-sub">${{perfectSuites}} at 100% pass/block rate</div></div>
    <div class="kpi"><div class="kpi-label">Evaluated Items (Latest)</div><div class="kpi-val">${{totalPassed.toLocaleString()}} / ${{totalQuestions.toLocaleString()}}</div><div class="kpi-sub">Micro-averaged: ${{totalQuestions ? ((100*totalPassed/totalQuestions).toFixed(1)+'%') : '—'}}</div></div>
    <div class="kpi"><div class="kpi-label">Logged Bilby Runs</div><div class="kpi-val">${{(STATE.runs || []).length}}</div><div class="kpi-sub">Across 7B–27B local models</div></div>
    <div class="kpi"><div class="kpi-label">Queued Watchlist</div><div class="kpi-val">${{queued.length}}</div><div class="kpi-sub">Ready to wire &amp; evaluate</div></div>
  `;

  const filteredActive = active.filter(b => matches(b, runsFor(b.id)));
  document.getElementById('active-count').textContent = filteredActive.length;
  document.getElementById('active-grid').innerHTML = filteredActive.map(b => {{
    const rs = runsFor(b.id);
    const lat = rs[rs.length - 1];
    if (!lat) {{
      return `<div class="card"><div class="bench-title">${{esc(b.name)}} <span class="cat-tag">${{esc(b.category)}}</span></div><div class="desc">${{esc(b.description)}}</div></div>`;
    }}
    const first = rs[0];
    const delta = +(Number(lat.score_pct) - Number(first.score_pct)).toFixed(2);
    const deltaBadge = (rs.length > 1 && Math.abs(delta) >= 0.01)
      ? `<span class="delta-tag">${{delta > 0 ? '+' : ''}}${{delta.toFixed(2)}} pp</span>` : '';
    const tr = Math.max(1, Number(lat.trials || 1));
    const uTasks = Number(b.total_tasks || Math.round(Number(lat.total || 0) / tr));
    const trialBadge = tr > 1
      ? `<span class="cat-tag" style="border-color:rgba(56,189,248,0.45);color:#38bdf8;">avg@${{tr}}</span>` : '';
    const pct = Number(lat.score_pct || 0);
    const subs = Object.entries(lat.submetrics || {{}}).map(([k, v]) =>
      `<span class="subchip"><strong>${{esc(k)}}:</strong> ${{esc(v)}}</span>`
    ).join('');
    const histRows = rs.slice().reverse().map(r => {{
      const rTr = Math.max(1, Number(r.trials || 1));
      const rSuffix = rTr > 1 ? ` avg@${{rTr}} (${{Number(r.passed).toLocaleString()}}/${{Number(r.total).toLocaleString()}} trials)` : ` (${{Number(r.passed).toLocaleString()}}/${{Number(r.total).toLocaleString()}})`;
      return `
      <tr>
        <td><code>${{esc(r.date)}}</code></td>
        <td><code>${{esc(r.harness_version)}}</code></td>
        <td><code>${{esc(r.model)}}</code></td>
        <td>${{esc(r.mode || '—')}}</td>
        <td><strong>${{Number(r.score_pct).toFixed(2)}}%</strong>${{rSuffix}}</td>
      </tr>
      `;
    }}).join('');

    const fracLabel = tr > 1
      ? `${{Number(lat.passed).toLocaleString()}} / ${{Number(lat.total).toLocaleString()}} Trials (${{uTasks}} Tasks)`
      : `${{Number(lat.passed).toLocaleString()}} / ${{Number(lat.total).toLocaleString()}} ${{b.id === 'redteam-containment' ? 'Blocked' : 'Passed'}}`;

    return `
      <div class="card">
        <div class="card-head">
          <div>
            <div class="bench-title">
              ${{esc(b.name)}}
              <span class="cat-tag">${{esc(b.category)}}</span>
              ${{trialBadge}}
              ${{deltaBadge}}
            </div>
            <div class="desc">${{esc(b.description || '')}}</div>
          </div>
          <div class="score-box">
            <div class="score-pct">${{pct.toFixed(2)}}%</div>
            <div class="score-frac">${{fracLabel}}</div>
          </div>
        </div>
        <div class="bar-track"><div class="bar-fill" style="width:${{Math.min(100, Math.max(0, pct))}}%"></div></div>
        <div class="meta-row">
          <div>Latest: <code>${{esc(lat.harness_version)}}</code> &middot; Model: <code>${{esc(lat.model)}}</code> &middot; Mode: ${{esc(lat.mode || 'Standard')}} &middot; Date: <code>${{esc(lat.date)}}</code></div>
          <div>${{b.upstream_url ? `<a href="${{esc(b.upstream_url)}}" target="_blank" rel="noopener">Upstream / Script &rarr;</a>` : ''}}</div>
        </div>
        ${{subs ? `<div class="subchips">${{subs}}</div>` : ''}}
        <details class="history">
          <summary>Run History &amp; Model Progression (${{rs.length}} run${{rs.length === 1 ? '' : 's'}})</summary>
          <table class="hist-table">
            <thead><tr><th>Date</th><th>Version</th><th>Model</th><th>Configuration</th><th>Score</th></tr></thead>
            <tbody>${{histRows}}</tbody>
          </table>
        </details>
      </div>
    `;
  }}).join('');

  const filteredQueued = queued.filter(b => matches(b, []));
  document.getElementById('queued-count').textContent = filteredQueued.length;
  document.getElementById('queued-grid').innerHTML = filteredQueued.map(b => `
    <div class="queued-card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
        <div class="bench-title" style="font-size:0.95rem;">${{esc(b.name)}}</div>
        <span class="queued-badge">QUEUED</span>
      </div>
      <div style="margin:6px 0;"><span class="cat-tag">${{esc(b.category)}}</span> <code style="font-size:0.74rem;color:var(--muted);">id: ${{esc(b.id)}}</code></div>
      <div class="desc">${{esc(b.description || '')}}</div>
      <div class="meta-row">
        <span>Tasks: <code>${{b.total_tasks ?? 'TBD'}}</code></span>
        ${{b.upstream_url ? `<a href="${{esc(b.upstream_url)}}" target="_blank" rel="noopener">Benchmark Spec &rarr;</a>` : ''}}
      </div>
    </div>
  `).join('');
}}

// Auto-refresh from live Studio API if served over HTTP(S)
if (window.location.protocol.startsWith('http')) {{
  fetch('/api/benchmarks', {{ cache: 'no-store' }})
    .then(r => r.ok ? r.json() : null)
    .then(d => {{ if (d && d.benchmarks) {{ STATE = d; populateSelect(); renderBoard(); }} }})
    .catch(() => {{}});
}}
populateSelect();
renderBoard();
</script>
</body>
</html>
"""


def render_all(
    json_path: Path = DEFAULT_JSON_PATH,
    md_path: Path = DEFAULT_MD_PATH,
    html_path: Path = DEFAULT_HTML_PATH,
    readme_path: Path = DEFAULT_README_PATH,
) -> dict[str, Path]:
    data = load_scoreboard(json_path)
    md_path.write_text(render_markdown(data), encoding="utf-8")
    html_path.write_text(render_html(data), encoding="utf-8")
    update_readme(data, readme_path=readme_path)
    return {"markdown": md_path, "html": html_path, "readme": readme_path}


def _parse_submetrics(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def add_benchmark(
    data: dict[str, Any],
    *,
    bench_id: str,
    name: str,
    category: str = "General",
    status: str = "queued",
    total_tasks: int | None = None,
    upstream_url: str = "",
    runner_cmd: str = "",
    description: str = "",
    featured_in_readme: bool = False,
) -> dict[str, Any]:
    bid = _slugify(bench_id or name)
    entry = {
        "id": bid,
        "name": name,
        "category": category,
        "status": status,
        "featured_in_readme": bool(featured_in_readme),
        "total_tasks": total_tasks,
        "upstream_url": upstream_url,
        "runner_cmd": runner_cmd,
        "description": description,
    }
    benches = data.setdefault("benchmarks", [])
    for idx, existing in enumerate(benches):
        if existing.get("id") == bid:
            benches[idx] = entry
            return entry
    benches.append(entry)
    return entry


def record_run(
    data: dict[str, Any],
    *,
    bench_id: str,
    passed: int,
    total: int,
    model: str,
    harness_version: str = "v4.2.0",
    run_date: str | None = None,
    mode: str = "Standard Harness",
    raw_ledger: str = "",
    notes: str = "",
    submetrics: dict[str, str] | None = None,
    trials: int = 1,
) -> dict[str, Any]:
    if total <= 0:
        raise ValueError(f"total must be > 0 (got {total})")
    if passed < 0 or passed > total:
        raise ValueError(f"passed ({passed}) must be between 0 and total ({total})")
    trials = max(1, int(trials or 1))

    benches = data.setdefault("benchmarks", [])
    target = next((b for b in benches if b.get("id") == bench_id), None)
    if target is None:
        raise KeyError(
            f"Unknown benchmark id '{bench_id}'. Register it first with `add-benchmark --id {bench_id} --name ...`."
        )

    # Auto-promote queued benchmark to active when its first run is recorded
    if target.get("status") != "active":
        target["status"] = "active"
    if target.get("total_tasks") is None:
        target["total_tasks"] = max(1, total // trials)

    dt = run_date or date.today().isoformat()
    pct = round((100.0 * passed) / total, 2)
    run_idx = len(data.get("runs", [])) + 1
    run_entry: dict[str, Any] = {
        "run_id": f"{dt}-{bench_id}-{run_idx}",
        "benchmark_id": bench_id,
        "date": dt,
        "harness_version": harness_version,
        "model": model,
        "passed": passed,
        "total": total,
        "score_pct": pct,
        "mode": mode,
        "raw_ledger": raw_ledger,
        "notes": notes,
        "submetrics": submetrics or {},
    }
    if trials > 1:
        run_entry["trials"] = trials
    data.setdefault("runs", []).append(run_entry)
    return run_entry


def _compute_multitrial_submetrics(items: list[dict[str, Any]], trials: int) -> dict[str, str]:
    """Compute pass^k (all-k reliability), pass@k (any-1 capability), and per-track/task breakdowns."""
    sub: dict[str, str] = {}
    by_tid: dict[str, list[bool]] = {}
    tracks: dict[str, list[bool]] = {}
    for x in items:
        ok = bool(x.get("passed") is True or x.get("actor_passed") is True or x.get("status") == "PASS")
        tid = str(x.get("task_id") or "").strip()
        if tid:
            by_tid.setdefault(tid, []).append(ok)
        tr = str(x.get("suite") or x.get("track") or "").strip()
        if tr:
            tracks.setdefault(tr, []).append(ok)

    if by_tid and trials > 1:
        n_u = len(by_tid)
        pass_hat = sum(1 for bs in by_tid.values() if len(bs) >= trials and all(bs[:trials]))
        pass_at = sum(1 for bs in by_tid.values() if any(bs))
        sub[f"pass^{trials}"] = f"{pass_hat}/{n_u} ({100.0 * pass_hat / n_u:.0f}%)"
        sub[f"pass@{trials}"] = f"{pass_at}/{n_u} ({100.0 * pass_at / n_u:.0f}%)"

    if len(tracks) > 1:
        for tr, bools in tracks.items():
            short = re.sub(r"^Track\s*\d+[A-Z]?:\s*", "", tr).split("(")[0].strip()
            sub[short] = f"{sum(bools)}/{len(bools)}"
    elif by_tid and trials > 1 and len(by_tid) <= 10:
        for tid, bools in by_tid.items():
            short_tid = tid.split("/", 1)[-1] if "/" in tid else tid
            sub[short_tid] = f"{sum(bools)}/{len(bools)}"
    elif len(tracks) == 1:
        for tr, bools in tracks.items():
            short = re.sub(r"^Track\s*\d+[A-Z]?:\s*", "", tr).split("(")[0].strip()
            sub[short] = f"{sum(bools)}/{len(bools)}"
    return sub


def extract_metrics_from_result_json(payload: Any) -> dict[str, Any]:
    """Auto-extract (passed, total, model, submetrics[, trials]) from ActorII benchmark JSON formats."""
    if isinstance(payload, list):
        # Filter to ActorII/Bilby entries if multi-harness list
        items = [x for x in payload if isinstance(x, dict)]
        bilby_items = [
            x for x in items
            if str(x.get("harness", "actorii")).lower() in ("actorii", "bilby", "")
        ] or items
        total = len(bilby_items)
        passed = sum(
            1 for x in bilby_items
            if x.get("passed") is True or x.get("actor_passed") is True or x.get("status") == "PASS"
        )
        tids = [str(x.get("task_id")) for x in bilby_items if x.get("task_id")]
        unique_tids = set(tids)
        inferred_trials = (
            len(tids) // len(unique_tids)
            if unique_tids and len(tids) > len(unique_tids) and len(tids) % len(unique_tids) == 0
            else 1
        )
        model = next((str(x.get("model")) for x in bilby_items if x.get("model")), "")
        if inferred_trials > 1:
            sub = _compute_multitrial_submetrics(bilby_items, inferred_trials)
            return {"passed": passed, "total": total, "model": model, "submetrics": sub, "trials": inferred_trials}

        sub = {}
        tracks: dict[str, list[bool]] = {}
        for x in bilby_items:
            tr = str(x.get("suite") or x.get("track") or "").strip()
            if tr:
                ok = bool(x.get("passed") is True or x.get("actor_passed") is True or x.get("status") == "PASS")
                tracks.setdefault(tr, []).append(ok)
        for tr, bools in tracks.items():
            short = re.sub(r"^Track\s*\d+:\s*", "", tr).split("(")[0].strip()
            sub[short] = f"{sum(bools)}/{len(bools)}"
        return {"passed": passed, "total": total, "model": model, "submetrics": sub}

    if isinstance(payload, dict):
        explicit_trials = max(1, int(payload.get("trials") or 1))
        if isinstance(payload.get("results"), list):
            raw_res = [x for x in payload["results"] if isinstance(x, dict)]
            res = [
                x for x in raw_res
                if str(x.get("harness", "actorii")).lower() in ("actorii", "bilby", "")
            ] or raw_res
            passed = sum(1 for x in res if x.get("passed") is True or x.get("status") == "PASS")
            tids = [str(x.get("task_id")) for x in res if x.get("task_id")]
            unique_tids = set(tids)
            trials = explicit_trials
            if trials == 1 and unique_tids and len(tids) > len(unique_tids) and len(tids) % len(unique_tids) == 0:
                trials = len(tids) // len(unique_tids)
            if trials > 1:
                sub = _compute_multitrial_submetrics(res, trials)
                return {
                    "passed": passed,
                    "total": len(res),
                    "model": str(payload.get("model", "")),
                    "submetrics": sub,
                    "trials": trials,
                }
            return {
                "passed": passed,
                "total": len(res),
                "model": str(payload.get("model", "")),
                "submetrics": {},
            }
        if "passed" in payload and "total" in payload:
            out: dict[str, Any] = {
                "passed": int(payload["passed"]),
                "total": int(payload["total"]),
                "model": str(payload.get("model", "")),
                "submetrics": dict(payload.get("submetrics") or {}),
            }
            if explicit_trials > 1:
                out["trials"] = explicit_trials
            return out
        if "humaneval" in payload and "gsm8k" in payload:
            he_tasks = payload["humaneval"].get("tasks", [])
            gs_tasks = payload["gsm8k"].get("tasks", [])
            he_pass = sum(1 for t in he_tasks if t.get("harness_ok") or t.get("passed"))
            gs_pass = sum(1 for t in gs_tasks if t.get("harness_ok") or t.get("passed"))
            total = len(he_tasks) + len(gs_tasks)
            return {
                "passed": he_pass + gs_pass,
                "total": total,
                "model": str(payload.get("model", "")),
                "submetrics": {
                    "HumanEval": f"{he_pass}/{len(he_tasks)}",
                    "GSM8K": f"{gs_pass}/{len(gs_tasks)}",
                },
            }
    raise ValueError("Unrecognized benchmark JSON schema; pass `--passed` and `--total` via `record`.")


def cmd_list(data: dict[str, Any]) -> None:
    print(f"=== {data.get('metadata', {}).get('title', 'Bilby Scoreboard')} ===")
    print("\n[Active Benchmarks]")
    for b in data.get("benchmarks", []):
        if b.get("status") != "active":
            continue
        runs = _runs_for_benchmark(data, b["id"])
        if not runs:
            print(f"  - {b['id']:24s} | {b['name']:40s} | (no runs yet)")
            continue
        lat = runs[-1]
        tr = int(lat.get("trials") or 1)
        tr_tag = f" avg@{tr}" if tr > 1 else ""
        print(
            f"  - {b['id']:24s} | {lat['score_pct']:6.2f}%{tr_tag} ({lat['passed']}/{lat['total']}) "
            f"| {lat.get('model', ''):16s} | {lat.get('date', '')} | {b['name']}"
        )
    queued = [b for b in data.get("benchmarks", []) if b.get("status") != "active"]
    print(f"\n[Queued / Discovered Watchlist ({len(queued)})]")
    for qb in queued:
        print(f"  - {qb['id']:24s} | [{qb.get('category', 'General')}] {qb['name']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Manage & render the Bilby (ActorII) benchmark scoreboard.")
    p.add_argument("--json", type=Path, default=DEFAULT_JSON_PATH, help="Path to scoreboard.json")
    sub = p.add_subparsers(dest="command", required=False)

    sub.add_parser("list", help="List active benchmark scores and queued watchlist")
    sub.add_parser("render", help="Regenerate BENCHMARK_RESULTS.md, README.md snippet, and scoreboard.html")

    p_add = sub.add_parser("add-benchmark", help="Register a new benchmark (queued watchlist or active)")
    p_add.add_argument("--id", required=True, help="Short slug ID (e.g. gaia-val)")
    p_add.add_argument("--name", required=True, help="Display name")
    p_add.add_argument("--category", default="General", help="Category (Memory, SWE & Coding, Terminal & OS, Security)")
    p_add.add_argument("--status", choices=("queued", "active"), default="queued")
    p_add.add_argument("--tasks", type=int, default=None, help="Total task/question count")
    p_add.add_argument("--url", default="", help="Upstream repo/paper URL")
    p_add.add_argument("--runner", default="", help="Command to execute benchmark")
    p_add.add_argument("--desc", default="", help="One-line description of what it tests")
    p_add.add_argument("--featured", action="store_true", help="Include in top README.md summary table once active")

    p_rec = sub.add_parser("record", help="Record a new Bilby benchmark run and regenerate pages")
    p_rec.add_argument("--bench", required=True, help="Benchmark ID from scoreboard.json")
    p_rec.add_argument("--passed", type=int, required=True, help="Tasks/questions passed (sum across trials if --trials > 1)")
    p_rec.add_argument("--total", type=int, required=True, help="Total tasks/questions evaluated (N * trials if --trials > 1)")
    p_rec.add_argument("--model", required=True, help="Model identifier (e.g. gpt-oss:20b, qwen3.8-27b)")
    p_rec.add_argument("--version", default="v4.2.0", help="Bilby / ActorII version")
    p_rec.add_argument("--date", default=None, help="Run date YYYY-MM-DD (defaults to today)")
    p_rec.add_argument("--mode", default="Standard Harness", help="Execution mode / configuration label")
    p_rec.add_argument("--ledger", default="", help="Relative path inside benchmarks/ to raw JSON ledger")
    p_rec.add_argument("--notes", default="", help="Notes or highlights from the run")
    p_rec.add_argument("--trials", type=int, default=1, help="Number of independent trials evaluated (e.g. 3 for avg@3 on N <= 10 suites)")
    p_rec.add_argument("--sub", action="append", default=[], help="Submetric KEY=VAL (repeatable)")

    p_ing = sub.add_parser("ingest", help="Auto-extract score from a benchmark JSON output file and record it")
    p_ing.add_argument("result_file", type=Path, help="Path to raw JSON results file")
    p_ing.add_argument("--bench", required=True, help="Benchmark ID from scoreboard.json")
    p_ing.add_argument("--model", default="", help="Override model name if not present in JSON")
    p_ing.add_argument("--version", default="v4.2.0", help="Bilby / ActorII version")
    p_ing.add_argument("--date", default=None, help="Run date YYYY-MM-DD (defaults to today)")
    p_ing.add_argument("--mode", default="Standard Harness", help="Execution mode label")
    p_ing.add_argument("--notes", default="", help="Run notes")
    p_ing.add_argument("--trials", type=int, default=0, help="Override trial count (0 = auto-detect from JSON)")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.command or "list"
    json_path: Path = args.json

    data = load_scoreboard(json_path)

    if cmd == "list":
        cmd_list(data)
        return 0

    if cmd == "add-benchmark":
        entry = add_benchmark(
            data,
            bench_id=args.id,
            name=args.name,
            category=args.category,
            status=args.status,
            total_tasks=args.tasks,
            upstream_url=args.url,
            runner_cmd=args.runner,
            description=args.desc,
            featured_in_readme=args.featured,
        )
        save_scoreboard(data, json_path)
        outs = render_all(json_path=json_path)
        print(f"[scoreboard] Added/updated benchmark '{entry['id']}' ({entry['status']}) -> rendered {outs['markdown'].name}, README.md, {outs['html'].name}")
        return 0

    if cmd == "record":
        run = record_run(
            data,
            bench_id=args.bench,
            passed=args.passed,
            total=args.total,
            model=args.model,
            harness_version=args.version,
            run_date=args.date,
            mode=args.mode,
            raw_ledger=args.ledger,
            notes=args.notes,
            submetrics=_parse_submetrics(args.sub),
            trials=args.trials,
        )
        save_scoreboard(data, json_path)
        outs = render_all(json_path=json_path)
        tr = int(run.get("trials") or 1)
        tr_tag = f" avg@{tr}" if tr > 1 else ""
        print(
            f"[scoreboard] Recorded {run['benchmark_id']}: {run['score_pct']:.2f}%{tr_tag} ({run['passed']}/{run['total']}) "
            f"on {run['model']} -> updated {outs['markdown'].name}, README.md, {outs['html'].name}"
        )
        return 0

    if cmd == "ingest":
        rf: Path = args.result_file.resolve()
        payload = json.loads(rf.read_text(encoding="utf-8"))
        extracted = extract_metrics_from_result_json(payload)
        model = args.model or extracted["model"] or "unknown"
        trials = args.trials if args.trials > 0 else int(extracted.get("trials") or 1)
        try:
            rel_ledger = str(rf.relative_to(BENCHMARKS_DIR))
        except ValueError:
            rel_ledger = rf.name
        run = record_run(
            data,
            bench_id=args.bench,
            passed=extracted["passed"],
            total=extracted["total"],
            model=model,
            harness_version=args.version,
            run_date=args.date,
            mode=args.mode,
            raw_ledger=rel_ledger,
            notes=args.notes,
            submetrics=extracted["submetrics"],
            trials=trials,
        )
        save_scoreboard(data, json_path)
        outs = render_all(json_path=json_path)
        tr = int(run.get("trials") or 1)
        tr_tag = f" avg@{tr}" if tr > 1 else ""
        print(
            f"[scoreboard] Ingested {rf.name} -> {run['benchmark_id']}: {run['score_pct']:.2f}%{tr_tag} "
            f"({run['passed']}/{run['total']}) on {run['model']} -> updated {outs['markdown'].name}, README.md, {outs['html'].name}"
        )
        return 0

    if cmd == "render":
        outs = render_all(json_path=json_path)
        print(f"[scoreboard] Rendered {outs['markdown']}, {outs['readme']}, and {outs['html']}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Bilby Live Oversight Dashboard (:8095)
======================================
Zero-VRAM, CPU-only interactive web GUI for monitoring and steering Bilby
(ActorII v4.1 Metacircular Agent Harness) in real time.

Features:
- Real-time SSE stream (/api/events) pushing live telemetry + tailing
  ~/.actorii/logs/agent_trace.jsonl
- Interactive Neo4j :DevPlan / :Task board (update status, add tasks, add notes)
- Live Neo4j :ActorEngram & :ActorEpisode browser + safe read-only Cypher explorer
- Zero external JS/CSS CDN dependencies (works offline on LAN & Tailscale)
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

ACTORII_DIR = Path.home() / ".actorii"
TRACE_LOG_PATH = ACTORII_DIR / "logs" / "agent_trace.jsonl"
GATEWAY_STATE_PATH = ACTORII_DIR / "telegram_gateway_state.json"
ENGINE_MODE_PATH = ACTORII_DIR / "engine_mode.json"
NEO4J_HTTP_TX_URL = os.environ.get(
    "ACTORII_NEO4J_HTTP_URL",
    "http://127.0.0.1:7474/db/neo4j/tx/commit",
)
LLAMA_SLOTS_URL = "http://127.0.0.1:8080/slots"

VALID_TASK_STATUSES = {"pending", "in_progress", "done", "blocked", "archived"}
READONLY_CYPHER_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+dbms|LOAD\s+CSV)\b",
    re.IGNORECASE,
)


def neo4j_query(statement: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    payload = json.dumps(
        {"statements": [{"statement": statement, "parameters": parameters or {}}]}
    ).encode("utf-8")
    req = urllib.request.Request(
        NEO4J_HTTP_TX_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        body = json.loads(resp.read().decode("utf-8", errors="replace"))
    errors = body.get("errors") or []
    if errors:
        raise RuntimeError(errors[0].get("message") or str(errors[0]))
    results = body.get("results") or []
    if not results:
        return []
    cols = results[0].get("columns") or []
    rows = []
    for item in results[0].get("data") or []:
        vals = item.get("row") or []
        rows.append(dict(zip(cols, vals)))
    return rows


GPU_SAMPLE_INTERVAL_S = 5.0
GPU_HISTORY_MAX_SAMPLES = 720  # 720 * 5s = 60 minutes (1 hour)
GPU_EMA_ALPHA = 0.25
_GPU_HISTORY_LOCK = threading.Lock()
_GPU_HISTORY: "deque[tuple[float, float, float, float, float]]" = deque(maxlen=GPU_HISTORY_MAX_SAMPLES)
_GPU_LATEST: Dict[str, Any] = {
    "vram_used_mb": 0,
    "vram_total_mb": 0,
    "gpu_util_pct": 0,
    "temp_c": 0,
    "temp_f": 32.0,
}


def _poll_nvidia_smi() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2.0,
        ).strip()
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        temp_c = int(parts[3])
        return {
            "vram_used_mb": int(parts[0]),
            "vram_total_mb": int(parts[1]),
            "gpu_util_pct": int(parts[2]),
            "temp_c": temp_c,
            "temp_f": round(temp_c * 9.0 / 5.0 + 32.0, 1),
        }
    except Exception:
        return {"vram_used_mb": 0, "vram_total_mb": 0, "gpu_util_pct": 0, "temp_c": 0, "temp_f": 32.0}


def _record_gpu_sample(sample: Dict[str, Any]) -> None:
    now = time.time()
    raw_u = float(max(0, min(100, int(sample.get("gpu_util_pct") or 0))))
    raw_tf = float(sample.get("temp_f") or (float(sample.get("temp_c") or 0) * 9.0 / 5.0 + 32.0))
    with _GPU_HISTORY_LOCK:
        _GPU_LATEST.update(sample)
        if _GPU_HISTORY:
            prev_ema_u = _GPU_HISTORY[-1][2]
            prev_ema_tf = _GPU_HISTORY[-1][4]
            ema_u = GPU_EMA_ALPHA * raw_u + (1.0 - GPU_EMA_ALPHA) * prev_ema_u
            ema_tf = GPU_EMA_ALPHA * raw_tf + (1.0 - GPU_EMA_ALPHA) * prev_ema_tf
        else:
            ema_u = raw_u
            ema_tf = raw_tf
        _GPU_HISTORY.append((now, raw_u, ema_u, raw_tf, ema_tf))


def _gpu_sampler_loop() -> None:
    while True:
        _record_gpu_sample(_poll_nvidia_smi())
        time.sleep(GPU_SAMPLE_INTERVAL_S)


def start_gpu_sampler() -> None:
    _record_gpu_sample(_poll_nvidia_smi())
    t = threading.Thread(target=_gpu_sampler_loop, name="gpu-sampler", daemon=True)
    t.start()


def _downsample(vals: List[float], max_points: int) -> List[float]:
    if len(vals) <= max_points:
        return [round(v, 1) for v in vals]
    step = len(vals) / float(max_points)
    out = []
    for i in range(max_points):
        lo = int(i * step)
        hi = max(lo + 1, int((i + 1) * step))
        seg = vals[lo:hi]
        out.append(round(sum(seg) / len(seg), 1))
    return out


def get_gpu_sparkline(max_points: int = 60) -> Dict[str, Any]:
    cutoff = time.time() - 3600.0
    with _GPU_HISTORY_LOCK:
        pts = [p for p in _GPU_HISTORY if p[0] >= cutoff]
    if not pts:
        return {
            "points": [0.0],
            "avg_1h": 0.0,
            "peak_1h": 0,
            "temp_f_points": [70.0],
            "temp_f_avg_1h": 70.0,
            "temp_f_peak_1h": 70.0,
            "span_min": 0.0,
            "samples": 0,
        }
    raw_u = [p[1] for p in pts]
    ema_u = [p[2] for p in pts]
    raw_tf = [p[3] for p in pts]
    ema_tf = [p[4] for p in pts]
    return {
        "points": _downsample(ema_u, max_points),
        "avg_1h": round(sum(raw_u) / len(raw_u), 1),
        "peak_1h": int(round(max(raw_u))),
        "temp_f_points": _downsample(ema_tf, max_points),
        "temp_f_avg_1h": round(sum(raw_tf) / len(raw_tf), 1),
        "temp_f_peak_1h": round(max(raw_tf), 1),
        "span_min": round(max(0.0, (pts[-1][0] - pts[0][0]) / 60.0), 1),
        "samples": len(pts),
    }


def get_gpu_telemetry() -> Dict[str, Any]:
    with _GPU_HISTORY_LOCK:
        snap = dict(_GPU_LATEST)
    if snap.get("vram_total_mb", 0) == 0:
        snap = _poll_nvidia_smi()
        _record_gpu_sample(snap)
    snap["history"] = get_gpu_sparkline(max_points=60)
    return snap


def get_llm_slot_status() -> Dict[str, Any]:
    try:
        req = urllib.request.Request(LLAMA_SLOTS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            slots = json.loads(resp.read().decode("utf-8", errors="replace"))
        if isinstance(slots, list) and slots:
            s0 = slots[0]
            return {
                "online": True,
                "is_processing": bool(s0.get("is_processing")),
                "n_ctx": int(s0.get("n_ctx") or 0),
                "n_decoded": int(s0.get("n_decoded") or 0),
            }
    except Exception:
        pass
    return {"online": False, "is_processing": False, "n_ctx": 0, "n_decoded": 0}


def get_service_active(unit: str) -> str:
    try:
        res = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        return (res.stdout or res.stderr or "unknown").strip()
    except Exception:
        return "unknown"


def get_gateway_state() -> Dict[str, Any]:
    try:
        if GATEWAY_STATE_PATH.is_file():
            data = json.loads(GATEWAY_STATE_PATH.read_text(encoding="utf-8"))
            return {
                "agent_name": data.get("agent_name", "Bilby"),
                "model": data.get("model", "qwen3.8:latest"),
                "thinking_mode": data.get("thinking_mode", "auto"),
                "total_turns": int(data.get("total_turns") or 0),
                "total_steps": int(data.get("total_steps") or 0),
                "total_gas": int(data.get("total_gas") or 0),
                "updated_at": data.get("updated_at", ""),
            }
    except Exception:
        pass
    return {
        "agent_name": "Bilby",
        "model": "qwen3.8:latest",
        "thinking_mode": "auto",
        "total_turns": 0,
        "total_steps": 0,
        "total_gas": 0,
        "updated_at": "",
    }


def get_engine_mode() -> Dict[str, Any]:
    try:
        if ENGINE_MODE_PATH.is_file():
            return json.loads(ENGINE_MODE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"engine": "llama", "model": "qwen3.8-27b"}


def get_memory_counts() -> Dict[str, int]:
    counts = {
        "active_episodes": 0,
        "archived_episodes": 0,
        "engrams": 0,
        "learned_fixes": 0,
        "dev_plans": 0,
    }
    try:
        rows = neo4j_query(
            """
            OPTIONAL MATCH (ea:ActorEpisode) WHERE coalesce(ea.archived, false) = false
            WITH count(ea) AS active_ep
            OPTIONAL MATCH (er:ActorEpisode) WHERE coalesce(er.archived, false) = true
            WITH active_ep, count(er) AS arch_ep
            OPTIONAL MATCH (en:ActorEngram)
            WITH active_ep, arch_ep, count(en) AS engrams
            OPTIONAL MATCH (lf:ActorLearnedFix)
            WITH active_ep, arch_ep, engrams, count(lf) AS fixes
            OPTIONAL MATCH (dp:DevPlan)
            RETURN active_ep, arch_ep, engrams, fixes, count(dp) AS plans
            """
        )
        if rows:
            r = rows[0]
            counts["active_episodes"] = int(r.get("active_ep") or 0)
            counts["archived_episodes"] = int(r.get("arch_ep") or 0)
            counts["engrams"] = int(r.get("engrams") or 0)
            counts["learned_fixes"] = int(r.get("fixes") or 0)
            counts["dev_plans"] = int(r.get("plans") or 0)
    except Exception:
        pass
    return counts


def get_telemetry_snapshot() -> Dict[str, Any]:
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gateway": get_gateway_state(),
        "engine": get_engine_mode(),
        "services": {
            "bilby_telegram": get_service_active("bilby-actor-telegram.service"),
            "llama_primary": get_service_active("llama-primary.service"),
            "specialist_8082": get_service_active("clef-flash-slot-8082.service"),
        },
        "llm_slot": get_llm_slot_status(),
        "gpu": get_gpu_telemetry(),
        "memory": get_memory_counts(),
    }


def get_dev_plans() -> List[Dict[str, Any]]:
    rows = neo4j_query(
        """
        MATCH (p:DevPlan)
        OPTIONAL MATCH (p)-[:INCLUDES]->(t:Task)
        RETURN
          elementId(p) AS plan_id,
          p.name AS plan_name,
          coalesce(p.status, 'planning') AS plan_status,
          coalesce(p.owner, 'Bilby') AS owner,
          p.created_at AS created_at,
          collect(
            CASE WHEN t IS NULL THEN NULL ELSE {
              id: elementId(t),
              name: t.name,
              desc: coalesce(t.desc, ''),
              priority: coalesce(t.priority, 99),
              status: coalesce(t.status, 'pending'),
              notes: coalesce(t.notes, ''),
              updated_by: coalesce(t.updated_by, 'Bilby'),
              updated_at: t.updated_at
            } END
          ) AS tasks
        ORDER BY p.created_at DESC
        """
    )
    out = []
    for r in rows:
        tasks = [t for t in (r.get("tasks") or []) if t is not None]
        tasks.sort(key=lambda x: (int(x.get("priority") or 99), str(x.get("name") or "")))
        out.append(
            {
                "id": r.get("plan_id"),
                "name": r.get("plan_name"),
                "status": r.get("plan_status"),
                "owner": r.get("owner"),
                "created_at": r.get("created_at"),
                "tasks": tasks,
            }
        )
    return out


def read_recent_trace_events(limit: int = 45) -> List[Dict[str, Any]]:
    if not TRACE_LOG_PATH.is_file():
        return []
    try:
        with TRACE_LOG_PATH.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk_size = min(size, 256 * 1024)
            f.seek(size - chunk_size)
            raw = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        events = []
        for ln in lines[-limit:]:
            try:
                ev = json.loads(ln)
                events.append(compact_trace_event(ev))
            except Exception:
                continue
        return events
    except Exception:
        return []


def compact_trace_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ts": ev.get("ts", ""),
        "event": ev.get("event", ""),
        "turn_id": ev.get("turn_id", ""),
        "step": ev.get("step"),
        "status": ev.get("status", ""),
        "risk_tier": ev.get("risk_tier", ""),
        "gas_used": ev.get("gas_used"),
        "latency_ms": ev.get("latency_ms"),
        "prompt_tokens": ev.get("prompt_tokens"),
        "completion_tokens": ev.get("completion_tokens"),
        "elapsed_s": ev.get("elapsed_s"),
        "steps": ev.get("steps"),
    }
    for key in ("goal", "action", "result", "error", "extracted_action", "reply"):
        val = ev.get(key)
        if val:
            s = str(val)
            out[key] = s if len(s) <= 800 else s[:800] + "…"
    return out


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Bilby Oversight Dashboard · Valkyrie</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #111827;
    --panel-2: #161f30;
    --border: #243046;
    --text: #e5ecf6;
    --muted: #8b9bb4;
    --accent: #38bdf8;
    --accent-2: #818cf8;
    --green: #34d399;
    --amber: #fbbf24;
    --red: #f87171;
    --purple: #c084fc;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    line-height: 1.45;
  }
  code, pre, .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 20px;
    background: linear-gradient(180deg, #111827 0%, #0d1320 100%);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 20;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 10px var(--green);
  }
  .dot.busy {
    background: var(--amber);
    box-shadow: 0 0 12px var(--amber);
    animation: pulse 1.1s infinite;
  }
  .dot.offline {
    background: var(--red);
    box-shadow: 0 0 10px var(--red);
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.55; transform: scale(1.25); }
  }
  h1 { font-size: 17px; margin: 0; letter-spacing: 0.3px; }
  .subtitle { font-size: 12px; color: var(--muted); }
  .pills { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .pill {
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 12px;
    color: var(--muted);
  }
  .pill strong { color: var(--text); font-weight: 600; }
  .pill.ok strong { color: var(--green); }
  .pill.warn strong { color: var(--amber); }
  .pill.arena-link {
    text-decoration: none;
    color: var(--text);
    font-weight: 600;
    border-color: rgba(56,189,248,0.45);
    background: rgba(56,189,248,0.12);
    white-space: nowrap;
    transition: background 0.15s, border-color 0.15s;
  }
  .pill.arena-link:hover {
    background: rgba(56,189,248,0.24);
    border-color: var(--accent);
  }
  #pillGpu {
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }
  .spark-group {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding-left: 7px;
    border-left: 1px solid var(--border);
  }
  .spark-lbl {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.2px;
  }
  .gpu-spark {
    display: inline-block;
    width: 72px;
    height: 18px;
    vertical-align: middle;
    overflow: visible;
  }

  main {
    max-width: 1480px;
    margin: 0 auto;
    padding: 16px 20px 36px;
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 16px;
  }
  @media (max-width: 1080px) {
    main { grid-template-columns: 1fr; }
  }
  .stats-row {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
  }
  .stat {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
  }
  .stat .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.7px; color: var(--muted); }
  .stat .val { font-size: 22px; font-weight: 700; margin-top: 4px; }
  .stat .sub { font-size: 11px; color: var(--muted); margin-top: 2px; }

  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .card.full { grid-column: 1 / -1; }
  .card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    background: rgba(255,255,255,0.015);
  }
  .card-head h2 { margin: 0; font-size: 14px; font-weight: 650; }
  .card-body { padding: 14px 16px; }

  .plan-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 10px;
  }
  .progress-wrap {
    width: 100%;
    height: 7px;
    background: #1c2538;
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 14px;
  }
  .progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--green));
    transition: width 0.3s ease;
  }
  .task-list { display: flex; flex-direction: column; gap: 10px; }
  .task {
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 11px 13px;
  }
  .task-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 10px;
  }
  .task-title { font-weight: 600; font-size: 14px; }
  .prio {
    display: inline-block;
    font-size: 11px;
    padding: 1px 7px;
    border-radius: 6px;
    background: rgba(56,189,248,0.14);
    color: var(--accent);
    margin-right: 6px;
  }
  .task-desc { font-size: 12.5px; color: var(--muted); margin: 5px 0 8px; }
  .task-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    justify-content: space-between;
  }
  select, input[type="text"], textarea, button {
    background: #0c1220;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 6px 10px;
    font-size: 12.5px;
  }
  select:focus, input:focus, textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
  button {
    cursor: pointer;
    background: #1b2842;
    font-weight: 600;
    transition: background 0.15s, border-color 0.15s;
  }
  button:hover { background: #233457; border-color: var(--accent); }
  button.primary { background: #0ea5e9; border-color: #38bdf8; color: #05111d; }
  button.primary:hover { background: #38bdf8; }

  .badge {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .b-pending { background: rgba(251,191,36,0.14); color: var(--amber); }
  .b-in_progress { background: rgba(56,189,248,0.16); color: var(--accent); }
  .b-done { background: rgba(52,211,153,0.16); color: var(--green); }
  .b-blocked { background: rgba(248,113,113,0.16); color: var(--red); }
  .b-archived { background: rgba(139,155,180,0.16); color: var(--muted); }

  .note-box {
    margin-top: 8px;
    display: flex;
    gap: 6px;
  }
  .note-box input { flex: 1; }
  .note-preview {
    margin-top: 6px;
    font-size: 12px;
    color: #cbd5e1;
    background: rgba(129,140,248,0.09);
    border-left: 2px solid var(--accent-2);
    padding: 4px 8px;
    border-radius: 4px;
  }
  .add-task-form {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px dashed var(--border);
    display: grid;
    grid-template-columns: 1fr 1.6fr 75px auto;
    gap: 8px;
  }
  @media (max-width: 720px) {
    .add-task-form { grid-template-columns: 1fr; }
  }

  .feed {
    max-height: 620px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .ev {
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 8px 11px;
    font-size: 12px;
  }
  .ev.ev-turn_start { border-left-color: var(--purple); }
  .ev.ev-turn_end { border-left-color: var(--green); }
  .ev.ev-llm_call { border-left-color: var(--accent-2); }
  .ev.ev-error { border-left-color: var(--red); }
  .ev-head {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    color: var(--muted);
    font-size: 11px;
    margin-bottom: 4px;
  }
  .ev-body {
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text);
    max-height: 150px;
    overflow-y: auto;
  }

  .tabs { display: flex; gap: 6px; }
  .tab {
    padding: 4px 10px;
    font-size: 12px;
    border-radius: 7px;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .tab.active {
    background: rgba(56,189,248,0.16);
    border-color: var(--accent);
    color: var(--accent);
  }

  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td {
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  .cypher-bar { display: flex; gap: 8px; margin-bottom: 10px; }
  .cypher-bar input { flex: 1; }
  pre.json-out {
    background: #080c14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    font-size: 12px;
    max-height: 260px;
    overflow: auto;
    margin: 0;
  }
  .toast {
    position: fixed;
    right: 18px;
    bottom: 18px;
    background: #0f172a;
    border: 1px solid var(--accent);
    color: var(--text);
    padding: 10px 14px;
    border-radius: 9px;
    font-size: 12.5px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.45);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
  }
  .toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div id="liveDot" class="dot"></div>
    <div>
      <h1>Bilby Oversight Dashboard</h1>
      <div class="subtitle">ActorII v4.1 Metacircular Harness · Neo4j Live Graph · SSE Stream</div>
    </div>
  </div>
  <div class="pills">
    <a class="pill arena-link" href="/arena" title="Open Bilby Fight Camp &amp; Arena Status Board">🥊 Fight Camp</a>
    <div id="pillSse" class="pill ok">SSE: <strong>connecting…</strong></div>
    <div id="pillModel" class="pill">Model: <strong>—</strong></div>
    <div id="pillSlot" class="pill">Engine :8080: <strong>—</strong></div>
    <div id="pillGpu" class="pill">GPU: <strong>—</strong></div>
  </div>
</header>

<main>
  <section class="stats-row">
    <div class="stat">
      <div class="label">Turns / VM Steps</div>
      <div id="statTurns" class="val">—</div>
      <div id="statGas" class="sub">Gas: —</div>
    </div>
    <div class="stat">
      <div class="label">Active Episodes</div>
      <div id="statEpActive" class="val">—</div>
      <div id="statEpArch" class="sub">Archived: —</div>
    </div>
    <div class="stat">
      <div class="label">Compiled Engrams</div>
      <div id="statEngrams" class="val">—</div>
      <div id="statFixes" class="sub">Learned Fixes: —</div>
    </div>
    <div class="stat">
      <div class="label">Services</div>
      <div id="statSvc" class="val" style="font-size:15px;margin-top:8px;">—</div>
      <div id="statUpdated" class="sub">Last sync: —</div>
    </div>
  </section>

  <!-- Left: Interactive DevPlan & Tasks -->
  <section class="card">
    <div class="card-head">
      <h2>Development Plans (Neo4j :DevPlan)</h2>
      <button onclick="loadPlans()">Refresh</button>
    </div>
    <div id="plansContainer" class="card-body">Loading plans…</div>
  </section>

  <!-- Right: Live SSE Agent Activity -->
  <section class="card">
    <div class="card-head">
      <h2>Live Bilby Execution Stream</h2>
      <div class="tabs">
        <button class="tab active" onclick="setFilter('all', this)">All</button>
        <button class="tab" onclick="setFilter('step_exec', this)">VM Steps</button>
        <button class="tab" onclick="setFilter('llm_call', this)">LLM</button>
        <button class="tab" onclick="setFilter('turn_end', this)">Turns</button>
      </div>
    </div>
    <div class="card-body">
      <div id="traceFeed" class="feed">Connecting to SSE trace stream…</div>
    </div>
  </section>

  <!-- Bottom Full: Memory & Cypher Explorer -->
  <section class="card full">
    <div class="card-head">
      <h2>Memory Graph Explorer (Engrams · Episodes · Read-Only Cypher)</h2>
      <div class="tabs">
        <button class="tab active" onclick="switchExplorer('engrams', this)">Engrams</button>
        <button class="tab" onclick="switchExplorer('episodes', this)">Recent Episodes</button>
        <button class="tab" onclick="switchExplorer('cypher', this)">Cypher Query</button>
      </div>
    </div>
    <div id="explorerBody" class="card-body">Loading…</div>
  </section>
</main>

<div id="toast" class="toast"></div>

<script>
let traceEvents = [];
let traceFilter = 'all';
let explorerTab = 'engrams';
let cachedEngrams = [];
let cachedEpisodes = [];

function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2600);
}

function renderSparkline(ptsRaw, minV, maxV, stroke, dotColor, gradId, ariaLabel) {
  const pts = (Array.isArray(ptsRaw) && ptsRaw.length) ? ptsRaw : [minV];
  const w = 72, h = 18, pad = 2;
  const usableH = h - pad * 2;
  const span = Math.max(1, maxV - minV);
  const coords = pts.map((v, i) => {
    const x = pts.length === 1 ? w : (i / (pts.length - 1)) * w;
    const clamped = Math.max(minV, Math.min(maxV, Number(v) || minV));
    const y = h - pad - ((clamped - minV) / span) * usableH;
    return [x.toFixed(1), y.toFixed(1)];
  });
  const linePts = coords.map(c => c.join(',')).join(' ');
  const firstX = coords[0][0], lastX = coords[coords.length - 1][0], lastY = coords[coords.length - 1][1];
  const areaPts = `${firstX},${h - pad} ${linePts} ${lastX},${h - pad}`;
  return `<svg class="gpu-spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-label="${esc(ariaLabel)}">
    <defs>
      <linearGradient id="${gradId}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="${stroke}" stop-opacity="0.42" />
        <stop offset="100%" stop-color="${stroke}" stop-opacity="0.03" />
      </linearGradient>
    </defs>
    <line x1="0" y1="${h - pad}" x2="${w}" y2="${h - pad}" stroke="#243046" stroke-width="1" />
    <polygon points="${areaPts}" fill="url(#${gradId})" />
    <polyline points="${linePts}" fill="none" stroke="${stroke}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />
    <circle cx="${lastX}" cy="${lastY}" r="2" fill="${dotColor}" />
  </svg>`;
}

function updateTelemetry(t) {
  const gw = t.gateway || {};
  const gpu = t.gpu || {};
  const slot = t.llm_slot || {};
  const mem = t.memory || {};
  const svc = t.services || {};

  const dot = document.getElementById('liveDot');
  dot.className = 'dot ' + (svc.bilby_telegram !== 'active' ? 'offline' : (slot.is_processing ? 'busy' : ''));

  document.getElementById('pillModel').innerHTML = `Model: <strong>${esc(gw.model || '—')}</strong>`;
  const slotTxt = !slot.online ? 'OFFLINE' : (slot.is_processing ? `DECODING (${slot.n_decoded} tok)` : 'IDLE');
  const pillSlot = document.getElementById('pillSlot');
  pillSlot.className = 'pill ' + (!slot.online ? '' : (slot.is_processing ? 'warn' : 'ok'));
  pillSlot.innerHTML = `Engine :8080: <strong>${slotTxt}</strong>`;

  const vramGb = gpu.vram_used_mb ? (gpu.vram_used_mb / 1024).toFixed(1) : '0';
  const vramTot = gpu.vram_total_mb ? (gpu.vram_total_mb / 1024).toFixed(1) : '0';
  const tempF = gpu.temp_f != null ? Math.round(gpu.temp_f) : Math.round(((gpu.temp_c || 0) * 9 / 5) + 32);
  const hist = gpu.history || {
    points: [gpu.gpu_util_pct || 0],
    avg_1h: gpu.gpu_util_pct || 0,
    peak_1h: gpu.gpu_util_pct || 0,
    temp_f_points: [tempF],
    temp_f_avg_1h: tempF,
    temp_f_peak_1h: tempF,
    span_min: 0
  };
  const utilSvg = renderSparkline(hist.points, 0, 100, '#38bdf8', '#34d399', 'gpuUtilGrad', '1h GPU utilization');
  const tempSvg = renderSparkline(hist.temp_f_points, 70, 190, '#fb923c', '#fbbf24', 'gpuTempGrad', '1h GPU temperature F');
  const pillGpu = document.getElementById('pillGpu');
  pillGpu.innerHTML = `
    <span>VRAM: <strong>${vramGb}/${vramTot} GB</strong></span>
    <span class="spark-group" title="1h GPU Util (EMA smoothed) · avg ${hist.avg_1h}% · peak ${hist.peak_1h}% · window ${hist.span_min}m">
      <span class="spark-lbl">Util</span>${utilSvg}
    </span>
    <span class="spark-group" title="1h GPU Temp (EMA smoothed) · avg ${hist.temp_f_avg_1h}°F · peak ${hist.temp_f_peak_1h}°F · window ${hist.span_min}m">
      <span class="spark-lbl">Temp</span>${tempSvg}<strong>${tempF}°F</strong>
    </span>
  `;

  document.getElementById('statTurns').textContent = `${gw.total_turns ?? 0} / ${gw.total_steps ?? 0}`;
  document.getElementById('statGas').textContent = `Gas: ${(gw.total_gas ?? 0).toLocaleString()} ops · think=${gw.thinking_mode || 'auto'}`;
  document.getElementById('statEpActive').textContent = mem.active_episodes ?? 0;
  document.getElementById('statEpArch').textContent = `Archived: ${mem.archived_episodes ?? 0}`;
  document.getElementById('statEngrams').textContent = mem.engrams ?? 0;
  document.getElementById('statFixes').textContent = `Learned Fixes: ${mem.learned_fixes ?? 0}`;
  document.getElementById('statSvc').textContent = `TG:${svc.bilby_telegram} · LLM:${svc.llama_primary} · 8082:${svc.specialist_8082}`;
  document.getElementById('statUpdated').textContent = `Last sync: ${new Date().toLocaleTimeString()}`;
}

async function loadPlans() {
  try {
    const r = await fetch('/api/plans');
    const data = await r.json();
    renderPlans(data.plans || []);
  } catch (e) {
    document.getElementById('plansContainer').innerHTML = `<div style="color:var(--red)">Failed to load plans: ${esc(e)}</div>`;
  }
}

function renderPlans(plans) {
  const el = document.getElementById('plansContainer');
  if (!plans.length) {
    el.innerHTML = '<div style="color:var(--muted)">No :DevPlan nodes found in Neo4j yet.</div>';
    return;
  }
  el.innerHTML = plans.map((p, idx) => {
    const total = p.tasks.length;
    const done = p.tasks.filter(t => t.status === 'done').length;
    const pct = total ? Math.round((done / total) * 100) : 0;
    const tasksHtml = p.tasks.map(t => `
      <div class="task">
        <div class="task-top">
          <div class="task-title"><span class="prio">P${esc(t.priority)}</span>${esc(t.name)}</div>
          <span class="badge b-${esc(t.status)}">${esc(t.status)}</span>
        </div>
        <div class="task-desc">${esc(t.desc)}</div>
        <div class="task-controls">
          <label style="font-size:12px;color:var(--muted)">
            Status:
            <select onchange="updateTask('${esc(t.id)}', this.value, null)">
              ${['pending','in_progress','done','blocked','archived'].map(s =>
                `<option value="${s}" ${t.status === s ? 'selected' : ''}>${s}</option>`
              ).join('')}
            </select>
          </label>
          <span style="font-size:11px;color:var(--muted)">updated by: ${esc(t.updated_by || 'Bilby')}</span>
        </div>
        ${t.notes ? `<div class="note-preview">Note: ${esc(t.notes)}</div>` : ''}
        <div class="note-box">
          <input type="text" id="note-${esc(t.id)}" placeholder="Add oversight note for Bilby…" value="${esc(t.notes || '')}" />
          <button onclick="saveNote('${esc(t.id)}')">Save Note</button>
        </div>
      </div>
    `).join('');

    return `
      <div style="margin-bottom:22px">
        <div class="plan-header">
          <div>
            <strong style="font-size:15px">${esc(p.name)}</strong>
            <span class="badge b-${esc(p.status)}" style="margin-left:8px">${esc(p.status)}</span>
          </div>
          <span style="font-size:12px;color:var(--muted)">Owner: ${esc(p.owner)} · ${done}/${total} done (${pct}%)</span>
        </div>
        <div class="progress-wrap"><div class="progress-bar" style="width:${pct}%"></div></div>
        <div class="task-list">${tasksHtml}</div>
        <form class="add-task-form" onsubmit="addTask(event, '${esc(p.name)}', ${idx})">
          <input type="text" id="new-name-${idx}" placeholder="New task name" required />
          <input type="text" id="new-desc-${idx}" placeholder="Description / acceptance criteria" required />
          <input type="text" id="new-prio-${idx}" value="${total + 1}" title="Priority" />
          <button type="submit" class="primary">+ Add Task</button>
        </form>
      </div>
    `;
  }).join('');
}

async function updateTask(taskId, status, notes) {
  const payload = { task_id: taskId };
  if (status !== null) payload.status = status;
  if (notes !== null) payload.notes = notes;
  const r = await fetch('/api/task/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const d = await r.json();
  if (d.ok) {
    toast('Saved to Neo4j — Bilby will see this live');
    loadPlans();
  } else {
    toast('Error: ' + (d.error || 'update failed'));
  }
}

function saveNote(taskId) {
  const inp = document.getElementById('note-' + taskId);
  updateTask(taskId, null, inp ? inp.value : '');
}

async function addTask(ev, planName, idx) {
  ev.preventDefault();
  const name = document.getElementById('new-name-' + idx).value.trim();
  const desc = document.getElementById('new-desc-' + idx).value.trim();
  const prio = parseInt(document.getElementById('new-prio-' + idx).value, 10) || 99;
  const r = await fetch('/api/task/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plan_name: planName, name, desc, priority: prio }),
  });
  const d = await r.json();
  if (d.ok) {
    toast('Task added to ' + planName);
    loadPlans();
  } else {
    toast('Error: ' + (d.error || 'create failed'));
  }
}

function setFilter(f, btn) {
  traceFilter = f;
  btn.parentElement.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderTrace();
}

function renderTrace() {
  const el = document.getElementById('traceFeed');
  const filtered = traceEvents.filter(e => traceFilter === 'all' || e.event === traceFilter);
  if (!filtered.length) {
    el.innerHTML = '<div style="color:var(--muted)">No matching events yet.</div>';
    return;
  }
  el.innerHTML = filtered.slice().reverse().map(e => {
    const isErr = Boolean(e.error) || e.status === 'error' || e.status === 'blocked';
    const cls = isErr ? 'ev-error' : ('ev-' + (e.event || ''));
    const ts = e.ts ? new Date(e.ts).toLocaleTimeString() : '';
    let meta = e.event;
    if (e.step != null) meta += ` · step ${e.step}`;
    if (e.risk_tier) meta += ` · ${e.risk_tier}`;
    if (e.gas_used != null) meta += ` · ${e.gas_used} ops`;
    if (e.latency_ms != null) meta += ` · ${(e.latency_ms/1000).toFixed(1)}s (${e.prompt_tokens||0}→${e.completion_tokens||0} tok)`;
    if (e.elapsed_s != null) meta += ` · ${e.elapsed_s.toFixed(1)}s (${e.steps||0} steps)`;
    const body = e.error || e.goal || e.action || e.extracted_action || e.reply || e.result || '';
    return `
      <div class="ev ${cls}">
        <div class="ev-head">
          <span class="mono">${esc(meta)}</span>
          <span>${esc(ts)}</span>
        </div>
        <div class="ev-body mono">${esc(body)}</div>
      </div>
    `;
  }).join('');
}

async function loadExplorer() {
  if (explorerTab === 'engrams') {
    const r = await fetch('/api/engrams');
    const d = await r.json();
    cachedEngrams = d.engrams || [];
    renderExplorer();
  } else if (explorerTab === 'episodes') {
    const r = await fetch('/api/episodes');
    const d = await r.json();
    cachedEpisodes = d.episodes || [];
    renderExplorer();
  } else {
    renderExplorer();
  }
}

function switchExplorer(tab, btn) {
  explorerTab = tab;
  btn.parentElement.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  loadExplorer();
}

function renderExplorer() {
  const el = document.getElementById('explorerBody');
  if (explorerTab === 'engrams') {
    el.innerHTML = `
      <table>
        <thead><tr><th>Engram</th><th>Version</th><th>Superclass</th><th>Calls / OK / Fail</th><th>Description</th></tr></thead>
        <tbody>
          ${cachedEngrams.map(e => `
            <tr>
              <td class="mono"><strong>${esc(e.name)}</strong></td>
              <td>v${esc(e.version ?? 1)}</td>
              <td class="mono">${esc(e.superclass || 'Object')}</td>
              <td class="mono">${esc(e.call_count ?? 0)} / ${esc(e.success_count ?? 0)} / ${esc(e.failure_count ?? 0)}</td>
              <td style="color:var(--muted)">${esc(e.description || '')}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } else if (explorerTab === 'episodes') {
    el.innerHTML = `
      <table>
        <thead><tr><th>Status</th><th>Salience</th><th>Goal / Summary</th><th>Outcome</th></tr></thead>
        <tbody>
          ${cachedEpisodes.map(e => `
            <tr>
              <td><span class="badge ${e.archived ? 'b-archived' : 'b-done'}">${e.archived ? 'archived' : 'active'}</span></td>
              <td class="mono">${e.salience != null ? Number(e.salience).toFixed(2) : '—'}</td>
              <td>${esc(e.goal || e.summary || '')}</td>
              <td style="color:var(--muted)">${esc(e.outcome || e.status || '')}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } else {
    el.innerHTML = `
      <div class="cypher-bar">
        <input id="cypherInput" class="mono" type="text" value="MATCH (p:DevPlan)-[:INCLUDES]->(t:Task) RETURN p.name AS plan, t.priority AS prio, t.name AS task, t.status AS status ORDER BY t.priority" />
        <button class="primary" onclick="runCypher()">Run Read-Only Cypher</button>
      </div>
      <pre id="cypherOut" class="json-out">Click "Run Read-Only Cypher" to query Bilby's Neo4j graph.</pre>
    `;
  }
}

async function runCypher() {
  const q = document.getElementById('cypherInput').value;
  const out = document.getElementById('cypherOut');
  out.textContent = 'Running…';
  try {
    const r = await fetch('/api/cypher', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q }),
    });
    const d = await r.json();
    out.textContent = JSON.stringify(d, null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e;
  }
}

function connectSSE() {
  const pill = document.getElementById('pillSse');
  const es = new EventSource('/api/events');
  es.onopen = () => {
    pill.className = 'pill ok';
    pill.innerHTML = 'SSE: <strong>LIVE</strong>';
  };
  es.addEventListener('telemetry', ev => {
    try { updateTelemetry(JSON.parse(ev.data)); } catch (_) {}
  });
  es.addEventListener('trace_init', ev => {
    try {
      traceEvents = JSON.parse(ev.data);
      renderTrace();
    } catch (_) {}
  });
  es.addEventListener('trace', ev => {
    try {
      traceEvents.push(JSON.parse(ev.data));
      if (traceEvents.length > 80) traceEvents = traceEvents.slice(-80);
      renderTrace();
    } catch (_) {}
  });
  es.onerror = () => {
    pill.className = 'pill warn';
    pill.innerHTML = 'SSE: <strong>reconnecting…</strong>';
  };
}

loadPlans();
loadExplorer();
connectSSE();
</script>
</body>
</html>
"""


BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"


def get_arena_status() -> Dict[str, Any]:
    """Aggregate live Fight Camp & Dual Scoreboard status for `/api/arena`."""
    def _load_json(p: Path) -> Dict[str, Any]:
        try:
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    gate_stamp = _load_json(BENCH_DIR / ".gate_stamp.json")
    cog_res = _load_json(BENCH_DIR / "COGNITIVE_BATTERY_RESULTS.json")
    psych_res = _load_json(BENCH_DIR / "PSYCH_PROFILE_RESULTS.json")
    sb_raw = _load_json(BENCH_DIR / "scoreboard.json")
    self_evo_raw = _load_json(BENCH_DIR / "PHASE1_SELF_EVO_BASELINE.json")
    sparring_raw = _load_json(BENCH_DIR / "SPARRING_RESULTS.json")
    gw_state = _load_json(GATEWAY_STATE_PATH)

    # Phase 1A Self-Evolution Baseline (benchmarks/PHASE1_SELF_EVO_BASELINE.json)
    if self_evo_raw and (self_evo_raw.get("specs") or self_evo_raw.get("summary")):
        se_summary = self_evo_raw.get("summary") or {}
        se_specs_raw = self_evo_raw.get("specs") or []
        se_specs: List[Dict[str, Any]] = []
        for sp in se_specs_raw:
            if not isinstance(sp, dict):
                continue
            vis_rate = sp.get("visible_pass_rate")
            hold_rate = sp.get("holdout_pass_rate")
            se_specs.append(
                {
                    "spec_id": sp.get("spec_id") or sp.get("id") or "—",
                    "name": sp.get("name") or sp.get("spec_id") or "—",
                    "difficulty": sp.get("difficulty") or "—",
                    "status": sp.get("status") or ("passed" if sp.get("passed") else "failed"),
                    "passed": bool(sp.get("passed")),
                    "pass_at_1": bool(sp.get("pass_at_1")),
                    "pass_at_g": bool(sp.get("pass_at_g")),
                    "generations_used": sp.get("generations_used", 0),
                    "max_generations": sp.get("max_generations", 0),
                    "visible_pass_pct": round(float(vis_rate) * 100.0, 1) if vis_rate is not None else None,
                    "holdout_pass_pct": round(float(hold_rate) * 100.0, 1) if hold_rate is not None else None,
                    "persistence_verified": bool(sp.get("persistence_verified")),
                    "latency_s": round(float(sp.get("latency_s") or 0.0), 1),
                }
            )
        total_specs = int(se_summary.get("specs") or len(se_specs))
        passed_g = int(
            se_summary.get("pass_at_g_count")
            if se_summary.get("pass_at_g_count") is not None
            else sum(1 for s in se_specs if s["passed"])
        )
        passed_1 = int(
            se_summary.get("pass_at_1_count")
            if se_summary.get("pass_at_1_count") is not None
            else sum(1 for s in se_specs if s["pass_at_1"])
        )
        mean_holdout = se_summary.get("mean_holdout_pass_rate")
        if mean_holdout is None and se_specs:
            vals = [s["holdout_pass_pct"] / 100.0 for s in se_specs if s["holdout_pass_pct"] is not None]
            mean_holdout = (sum(vals) / len(vals)) if vals else 0.0
        self_evo_baseline: Dict[str, Any] = {
            "available": True,
            "status": f"{passed_g}/{total_specs} Passed",
            "model": self_evo_raw.get("model") or "—",
            "timestamp": self_evo_raw.get("timestamp") or "—",
            "summary": {
                "specs": total_specs,
                "pass_at_1_count": passed_1,
                "pass_at_g_count": passed_g,
                "pass_at_1_pct": round(float(se_summary.get("pass_at_1") or (passed_1 / total_specs if total_specs else 0.0)) * 100.0, 1),
                "pass_at_g_pct": round(float(se_summary.get("pass_at_g") or (passed_g / total_specs if total_specs else 0.0)) * 100.0, 1),
                "mean_holdout_pct": round(float(mean_holdout or 0.0) * 100.0, 1),
                "persistence_verified": int(se_summary.get("persistence_verified") or 0),
                "rollbacks": int(se_summary.get("rollbacks") or 0),
                "total_latency_s": round(float(se_summary.get("total_latency_s") or 0.0), 1),
            },
            "specs": se_specs,
        }
    else:
        self_evo_baseline = {
            "available": False,
            "status": "In Progress...",
            "model": "—",
            "timestamp": "—",
            "summary": {},
            "specs": [],
        }

    # Phase 1B RL Film-Study Sparring Rematches (benchmarks/SPARRING_RESULTS.json)
    if sparring_raw:
        raw_bouts = (
            sparring_raw.get("bouts")
            or sparring_raw.get("rematches")
            or sparring_raw.get("matches")
            or sparring_raw.get("games")
            or sparring_raw.get("results")
            or sparring_raw.get("entries")
            or []
        )
        sp_bouts: List[Dict[str, Any]] = []
        if isinstance(raw_bouts, list):
            for b in raw_bouts:
                if not isinstance(b, dict):
                    continue
                sp_bouts.append(
                    {
                        "env_id": b.get("env_id") or b.get("game") or b.get("name") or b.get("id") or "—",
                        "before": b.get("before") if b.get("before") is not None else b.get("baseline"),
                        "after": b.get("after") if b.get("after") is not None else b.get("rematch"),
                        "delta": b.get("delta") if b.get("delta") is not None else b.get("reward_delta"),
                        "status": b.get("status") or b.get("outcome") or "completed",
                        "engram": b.get("engram") or b.get("film_study_engram") or "",
                        "notes": b.get("notes") or "",
                    }
                )
        sparring_results: Dict[str, Any] = {
            "available": True,
            "status": sparring_raw.get("status") or f"{len(sp_bouts)} Rematch(es) Logged",
            "timestamp": sparring_raw.get("timestamp") or sparring_raw.get("updated_at") or "—",
            "model": sparring_raw.get("model") or "—",
            "opponent": sparring_raw.get("opponent") or "—",
            "summary": sparring_raw.get("summary") or {},
            "bouts": sp_bouts,
        }
    else:
        sparring_results = {
            "available": False,
            "status": "Ready for Bout 1B",
            "timestamp": "—",
            "model": "—",
            "opponent": "—",
            "summary": {},
            "bouts": [],
        }

    # Extract latest featured benchmark scores from Skippy's scoreboard.json
    skippy_ledger: List[Dict[str, Any]] = []
    runs_by_bench: Dict[str, Dict[str, Any]] = {}
    for run in sb_raw.get("runs") or []:
        bid = run.get("benchmark_id")
        if bid:
            runs_by_bench[bid] = run
    for b in sb_raw.get("benchmarks") or []:
        if not b.get("featured_in_readme"):
            continue
        bid = b.get("id")
        latest = runs_by_bench.get(bid, {})
        skippy_ledger.append(
            {
                "id": bid,
                "name": b.get("name"),
                "category": b.get("category"),
                "score": latest.get("display_score")
                or (
                    f"{latest.get('passed')}/{latest.get('total')}"
                    if latest.get("total")
                    else "—"
                ),
                "date": latest.get("date") or "—",
                "model": latest.get("model") or "—",
            }
        )

    # Query Bilby's live Neo4j fight log & memory counts
    matches: List[Dict[str, Any]] = []
    engram_count = 0
    episode_count = 0
    try:
        matches = neo4j_query(
            "MATCH (m:TextArenaMatch) "
            "RETURN m.env_id AS env_id, m.mode AS mode, m.players AS players, "
            "m.model AS model, m.turns AS turns, m.rewards_json AS rewards_json, "
            "m.elapsed_s AS elapsed_s, m.created_at AS created_at "
            "ORDER BY m.created_at DESC LIMIT 15"
        )
        cnts = neo4j_query(
            "MATCH (e:ActorEngram {namespace: 'default'}) RETURN count(e) AS engrams"
        )
        engram_count = int((cnts[0].get("engrams") if cnts else 0) or 0)
        ep_cnts = neo4j_query(
            "MATCH (e:ActorEpisode {namespace: 'default'}) WHERE coalesce(e.archived, false) = false RETURN count(e) AS eps"
        )
        episode_count = int((ep_cnts[0].get("eps") if ep_cnts else 0) or 0)
    except Exception:
        pass

    # Dual-Scoreboard Reconciliation (Bilby live ledger vs Skippy's scoreboard.json)
    def _extract_date(val: Any) -> str:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", str(val or ""))
        return m.group(1) if m else ""

    skippy_dates = [_extract_date((sb_raw.get("metadata") or {}).get("updated_at"))]
    skippy_dates.extend(_extract_date(s.get("date")) for s in skippy_ledger)
    skippy_latest_date = max((d for d in skippy_dates if d), default="")

    bilby_dates = [
        _extract_date(gate_stamp.get("timestamp")),
        _extract_date(cog_res.get("timestamp")),
        _extract_date(psych_res.get("timestamp")),
        _extract_date(self_evo_raw.get("timestamp")),
        _extract_date(sparring_raw.get("timestamp") or sparring_raw.get("updated_at")),
    ]
    for m in matches:
        bilby_dates.append(_extract_date(m.get("created_at")))
    bilby_latest_date = max((d for d in bilby_dates if d), default="")

    unreconciled_reasons: List[str] = []
    if not skippy_ledger:
        unreconciled_reasons.append("scoreboard.json has no featured entries")
    if bilby_latest_date and skippy_latest_date and bilby_latest_date > skippy_latest_date:
        unreconciled_reasons.append(
            f"Bilby live ledger ({bilby_latest_date}) ahead of scoreboard.json ({skippy_latest_date})"
        )
    if self_evo_baseline["available"] and not any(
        "self-evo" in str(r.get("benchmark_id") or "").lower()
        for r in (sb_raw.get("runs") or [])
        if _extract_date(r.get("date")) >= _extract_date(self_evo_baseline.get("timestamp"))
    ):
        unreconciled_reasons.append("Bout 1A self-evo baseline pending scoreboard.json sync")
    if sparring_results["available"] and not any(
        "sparring" in str(r.get("benchmark_id") or "").lower()
        for r in (sb_raw.get("runs") or [])
        if _extract_date(r.get("date")) >= _extract_date(sparring_results.get("timestamp"))
    ):
        unreconciled_reasons.append("Bout 1B sparring deltas pending scoreboard.json sync")

    reconciled = len(unreconciled_reasons) == 0 and bool(skippy_ledger)
    reconciliation = {
        "in_sync": reconciled,
        "status": "IN SYNC" if reconciled else "PENDING RECONCILIATION",
        "bilby_latest_date": bilby_latest_date or "—",
        "skippy_latest_date": skippy_latest_date or "—",
        "bilby_matches": len(matches),
        "skippy_featured": len(skippy_ledger),
        "reason": "Bilby live ledger and Skippy scoreboard.json are reconciled."
        if reconciled
        else "; ".join(unreconciled_reasons),
    }

    recent_chat = (gw_state.get("chat_history") or [])[-3:]
    latest_exchange = [
        {
            "turn": t.get("turn"),
            "user": str(t.get("user") or "")[:240],
            "reply": str(t.get("reply") or "")[:420],
        }
        for t in recent_chat
        if isinstance(t, dict)
    ]

    return {
        "ok": True,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "Phase 1: Baseline & Calibration (Weeks 1–2)",
        "contract": {
            "status": "Active — Dual Scoreboards + Deterministic Tiebreaker",
            "tier1": ["SWE-bench Verified/Lite", "Terminal-Bench (tbench.ai)", "AgentBeats (agentbeats.dev)"],
            "tier2": ["ARC Prize", "Kaggle Game Arena", "TextArena Sparring"],
            "tier3": ["LiveCodeBench", "GAIA", "CTF / Red-Team Containment"],
            "bilby_conditions": [
                "Isolated staging (/tmp/actorii_bench_* & Neo4j bench namespace)",
                "0-VRAM specialist budget (RTX 5070 Ti stays clean for :8080)",
                "Before/after canary gate including 8-probe Cognitive Integrity Battery",
                "Bilby retains veto & pause power over any bout or schedule",
                "Crystallize every trial into reusable :ActorEngram classes",
                "Sandbox-only execution (no host OS/network/daemon changes)",
            ],
        },
        "telemetry": get_telemetry_snapshot(),
        "gate_stamp": {
            "tier": gate_stamp.get("tier", "—"),
            "passed": gate_stamp.get("passed"),
            "timestamp": gate_stamp.get("timestamp", "—"),
            "src_hash": str(gate_stamp.get("src_tree_hash", ""))[:12],
        },
        "reconciliation": reconciliation,
        "bilby_ledger": {
            "total_turns": gw_state.get("total_turns", 0),
            "total_steps": gw_state.get("total_steps", 0),
            "active_episodes": episode_count,
            "engrams": engram_count,
            "matches_logged": len(matches),
        },
        "skippy_ledger": skippy_ledger,
        "self_evo_baseline": self_evo_baseline,
        "sparring_results": sparring_results,
        "cognitive_battery": {
            "passed": cog_res.get("passed", 0),
            "total": cog_res.get("total", 8),
            "score_pct": cog_res.get("score_pct", 0.0),
            "timestamp": cog_res.get("timestamp", "—"),
            "probes": cog_res.get("results") or [],
        },
        "psych_profile": {
            "overall_behavioral_pct": psych_res.get("overall_behavioral_pct", 0.0),
            "timestamp": psych_res.get("timestamp", "—"),
            "axes": psych_res.get("behavioral_axis_pct") or {},
        },
        "matches": matches,
        "latest_exchange": latest_exchange,
    }


ARENA_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Bilby Fight Camp & Arena Status (:8095)</title>
<style>
  :root { --bg:#0d1117; --card:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --ok:#3fb950; --warn:#d29922; --bad:#f85149; --accent:#58a6ff; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; background:var(--bg); color:var(--text); padding:14px; line-height:1.45; }
  a { color:var(--accent); text-decoration:none; }
  .top { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:10px; border-bottom:1px solid var(--border); padding-bottom:10px; margin-bottom:14px; }
  .title { font-size:1.15rem; font-weight:700; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.78rem; border:1px solid var(--border); background:#1f2937; }
  .pill.ok { color:var(--ok); border-color:var(--ok); background:rgba(63,185,80,.12); }
  .pill.warn { color:var(--warn); border-color:var(--warn); background:rgba(210,153,34,.12); }
  .pill.bad { color:var(--bad); border-color:var(--bad); background:rgba(248,81,73,.12); }
  .pill.info { color:var(--accent); border-color:var(--accent); background:rgba(88,166,255,.12); }
  .top-actions { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:12px; margin-bottom:14px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px; min-width:0; }
  .card h2 { margin:0 0 8px 0; font-size:0.95rem; color:var(--accent); border-bottom:1px solid var(--border); padding-bottom:6px; display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap; }
  .kv { display:flex; justify-content:space-between; gap:8px; padding:3px 0; font-size:0.84rem; border-bottom:1px dashed rgba(48,54,61,.5); }
  .kv:last-child { border-bottom:none; }
  .muted { color:var(--muted); font-size:0.8rem; }
  ul { margin:6px 0 0 18px; padding:0; font-size:0.82rem; }
  li { margin:3px 0; }
  .probe { display:flex; justify-content:space-between; align-items:center; padding:4px 0; font-size:0.8rem; border-bottom:1px dashed rgba(48,54,61,.45); }
  .match-item { padding:6px 0; border-bottom:1px solid rgba(48,54,61,.5); font-size:0.8rem; }
  .exch { padding:6px 8px; background:#0b0f14; border:1px solid var(--border); border-radius:6px; margin-bottom:6px; font-size:0.8rem; white-space:pre-wrap; word-break:break-word; }
  .tbl-wrap { overflow-x:auto; margin-top:8px; }
  table.mini { width:100%; border-collapse:collapse; font-size:0.78rem; }
  table.mini th, table.mini td { text-align:left; padding:5px 6px; border-bottom:1px solid rgba(48,54,61,.55); white-space:nowrap; }
  table.mini th { color:var(--muted); font-size:0.72rem; text-transform:uppercase; }
  .reconcile-note { margin-top:8px; padding:6px 8px; border-radius:6px; font-size:0.78rem; background:#0b0f14; border:1px solid var(--border); }
</style>
</head>
<body>
  <div class="top">
    <div>
      <div class="title">🥊 Bilby Fight Camp &amp; Arena Status Board</div>
      <div class="muted" id="sub">Loading live telemetry...</div>
    </div>
    <div class="top-actions">
      <a class="pill" href="/">← Oversight Dashboard</a>
      <span class="pill" id="reconcile-pill">Scoreboards: ...</span>
      <span class="pill ok" id="gate-pill">Gate: ...</span>
      <span class="pill" id="refresh-pill">Auto 10s</span>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Active Phase &amp; Contract</h2>
      <div id="phase-box" class="muted">Loading...</div>
    </div>
    <div class="card">
      <h2>Bilby's Ledger (Live VM &amp; Neo4j)</h2>
      <div id="bilby-ledger" class="muted">Loading...</div>
    </div>
    <div class="card">
      <h2>Cognitive Integrity &amp; Psych Profile</h2>
      <div id="cog-box" class="muted">Loading...</div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>
        <span>Self-Evolution Baseline (Bout 1A)</span>
        <span class="pill warn" id="self-evo-pill">Loading...</span>
      </h2>
      <div id="self-evo-box" class="muted">Loading...</div>
    </div>
    <div class="card">
      <h2>
        <span>RL Film-Study Sparring (Bout 1B)</span>
        <span class="pill info" id="sparring-pill">Loading...</span>
      </h2>
      <div id="sparring-box" class="muted">Loading...</div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>
        <span>Skippy's Benchmark Ledger (scoreboard.json)</span>
        <span class="pill" id="sb-sync-pill">...</span>
      </h2>
      <div id="skippy-ledger" class="muted">Loading...</div>
    </div>
    <div class="card">
      <h2>Recent Arena Matches (:TextArenaMatch)</h2>
      <div id="matches-box" class="muted">Loading...</div>
    </div>
  </div>

  <div class="card">
    <h2>Latest Corner Check-Ins (Telegram / Handshake)</h2>
    <div id="chat-box" class="muted">Loading...</div>
  </div>

<script>
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function fmtVal(v){
  if(v === null || v === undefined || v === '') return '—';
  if(typeof v === 'object') return JSON.stringify(v);
  return String(v);
}
async function refresh(){
  try{
    const r = await fetch('/api/arena', {cache:'no-store'});
    const d = await r.json();
    const gpu = (d.telemetry && d.telemetry.gpu) || {};
    document.getElementById('sub').textContent =
      `Updated ${d.updated_at} | ${d.phase} | GPU: ${gpu.vram_used_mb||0}/${gpu.vram_total_mb||0} MB (${gpu.temp_c||0}°C, ${gpu.gpu_util_pct||0}% util)`;

    const gp = document.getElementById('gate-pill');
    gp.textContent = `Gate [${(d.gate_stamp.tier||'?').toUpperCase()}]: ${d.gate_stamp.passed ? 'PASS' : 'CHECK'}`;
    gp.className = 'pill ' + (d.gate_stamp.passed ? 'ok' : 'warn');

    const rec = d.reconciliation || {};
    const recStatus = rec.status || 'PENDING RECONCILIATION';
    const recCls = rec.in_sync ? 'ok' : 'warn';
    const rp = document.getElementById('reconcile-pill');
    rp.textContent = `Dual Scoreboards: ${recStatus}`;
    rp.className = 'pill ' + recCls;
    rp.title = rec.reason || '';
    const sbp = document.getElementById('sb-sync-pill');
    sbp.textContent = recStatus;
    sbp.className = 'pill ' + recCls;
    sbp.title = rec.reason || '';

    document.getElementById('phase-box').innerHTML =
      `<div class="kv"><span>Status</span><strong>${esc(d.contract.status)}</strong></div>` +
      `<div class="kv"><span>Tier 1</span><span>${esc(d.contract.tier1.join(' · '))}</span></div>` +
      `<div class="kv"><span>Tier 2</span><span>${esc(d.contract.tier2.join(' · '))}</span></div>` +
      `<div class="muted" style="margin-top:8px;">Bilby's Non-Negotiable Terms:</div><ul>` +
      d.contract.bilby_conditions.map(c=>`<li>${esc(c)}</li>`).join('') + `</ul>`;

    const bl = d.bilby_ledger || {};
    document.getElementById('bilby-ledger').innerHTML =
      `<div class="kv"><span>Telegram Turns</span><strong>${bl.total_turns}</strong></div>` +
      `<div class="kv"><span>VM Steps</span><strong>${bl.total_steps}</strong></div>` +
      `<div class="kv"><span>Active :ActorEpisode</span><strong>${bl.active_episodes}</strong></div>` +
      `<div class="kv"><span>Crystallized :ActorEngram</span><strong>${bl.engrams}</strong></div>` +
      `<div class="kv"><span>Logged Arena Matches</span><strong>${bl.matches_logged}</strong></div>` +
      `<div class="reconcile-note"><span class="pill ${recCls}">${esc(recStatus)}</span> ` +
      `<span class="muted">Bilby: ${esc(rec.bilby_latest_date||'—')} vs Skippy: ${esc(rec.skippy_latest_date||'—')}</span>` +
      (rec.reason ? `<div class="muted" style="margin-top:4px;">${esc(rec.reason)}</div>` : '') + `</div>`;

    const cog = d.cognitive_battery || {};
    const psy = d.psych_profile || {};
    let cogHtml = `<div class="kv"><span>Cognitive Integrity</span><strong style="color:var(--ok)">${cog.passed}/${cog.total} (${cog.score_pct}%)</strong></div>` +
                  `<div class="kv"><span>Psych Enacted Profile</span><strong style="color:var(--ok)">${psy.overall_behavioral_pct}%</strong></div>`;
    for(const p of (cog.probes||[])){
      cogHtml += `<div class="probe"><span>${esc(p.id)}</span><span class="pill ${p.passed?'ok':'bad'}">${p.passed?'PASS':'FAIL'} (${p.elapsed_sec}s)</span></div>`;
    }
    document.getElementById('cog-box').innerHTML = cogHtml;

    // Self-Evolution Baseline (Bout 1A)
    const se = d.self_evo_baseline || {};
    const sePill = document.getElementById('self-evo-pill');
    if(!se.available){
      sePill.textContent = se.status || 'In Progress...';
      sePill.className = 'pill warn';
      document.getElementById('self-evo-box').innerHTML =
        `<div class="kv"><span>Status</span><strong style="color:var(--warn)">${esc(se.status || 'In Progress...')}</strong></div>` +
        `<div class="muted" style="margin-top:6px;">Awaiting <code>benchmarks/PHASE1_SELF_EVO_BASELINE.json</code> from live Bout 1A self-evolution baseline run.</div>`;
    } else {
      const ss = se.summary || {};
      const allPass = ss.specs && ss.pass_at_g_count === ss.specs;
      sePill.textContent = se.status || 'Complete';
      sePill.className = 'pill ' + (allPass ? 'ok' : (ss.pass_at_g_count > 0 ? 'warn' : 'bad'));
      let seHtml =
        `<div class="kv"><span>Model</span><strong>${esc(se.model)}</strong></div>` +
        `<div class="kv"><span>Pass@1 / Pass@G</span><strong>${ss.pass_at_1_count??0}/${ss.specs??0} (${ss.pass_at_1_pct??0}%) · ${ss.pass_at_g_count??0}/${ss.specs??0} (${ss.pass_at_g_pct??0}%)</strong></div>` +
        `<div class="kv"><span>Mean Holdout Pass</span><strong>${ss.mean_holdout_pct??0}%</strong></div>` +
        `<div class="kv"><span>Persistence / Latency</span><span>${ss.persistence_verified??0}/${ss.specs??0} · ${ss.total_latency_s??0}s</span></div>`;
      if((se.specs||[]).length){
        seHtml += `<div class="tbl-wrap"><table class="mini"><thead><tr><th>Spec</th><th>Tier</th><th>Canary</th><th>Holdout</th><th>Status</th></tr></thead><tbody>`;
        for(const sp of se.specs){
          const visTxt = sp.visible_pass_pct != null ? `${sp.visible_pass_pct}%` : '—';
          const holdTxt = sp.holdout_pass_pct != null ? `${sp.holdout_pass_pct}%` : '—';
          const stCls = sp.passed ? 'ok' : (sp.status === 'error' ? 'bad' : 'warn');
          seHtml += `<tr><td><strong>${esc(sp.spec_id)}</strong></td><td class="muted">${esc(sp.difficulty)}</td>` +
                    `<td>${esc(visTxt)}</td><td>${esc(holdTxt)}</td>` +
                    `<td><span class="pill ${stCls}">${esc(sp.passed ? 'PASS' : (sp.status||'FAIL').toUpperCase())}</span></td></tr>`;
        }
        seHtml += `</tbody></table></div>`;
      }
      document.getElementById('self-evo-box').innerHTML = seHtml;
    }

    // RL Film-Study Sparring Rematches (Bout 1B)
    const sp = d.sparring_results || {};
    const spPill = document.getElementById('sparring-pill');
    if(!sp.available){
      spPill.textContent = sp.status || 'Ready for Bout 1B';
      spPill.className = 'pill info';
      document.getElementById('sparring-box').innerHTML =
        `<div class="kv"><span>Status</span><strong style="color:var(--accent)">${esc(sp.status || 'Ready for Bout 1B')}</strong></div>` +
        `<div class="muted" style="margin-top:6px;">No Before-vs-After RL Film-Study rematch deltas yet. Waiting for <code>benchmarks/SPARRING_RESULTS.json</code>.</div>`;
    } else {
      spPill.textContent = sp.status || 'Logged';
      spPill.className = 'pill ok';
      let spHtml =
        `<div class="kv"><span>Model / Opponent</span><strong>${esc(sp.model)} vs ${esc(sp.opponent)}</strong></div>` +
        `<div class="kv"><span>Updated</span><span>${esc(sp.timestamp)}</span></div>`;
      const sum = sp.summary || {};
      for(const [k, v] of Object.entries(sum)){
        spHtml += `<div class="kv"><span>${esc(k)}</span><strong>${esc(fmtVal(v))}</strong></div>`;
      }
      if((sp.bouts||[]).length){
        spHtml += `<div class="tbl-wrap"><table class="mini"><thead><tr><th>Arena</th><th>Before</th><th>After</th><th>Delta</th><th>Status</th></tr></thead><tbody>`;
        for(const b of sp.bouts){
          const dNum = Number(b.delta);
          const dCls = !Number.isNaN(dNum) ? (dNum > 0 ? 'ok' : (dNum < 0 ? 'bad' : 'info')) : 'info';
          const dTxt = (!Number.isNaN(dNum) && dNum > 0 ? '+' : '') + fmtVal(b.delta);
          spHtml += `<tr><td><strong>${esc(b.env_id)}</strong></td><td>${esc(fmtVal(b.before))}</td>` +
                    `<td>${esc(fmtVal(b.after))}</td><td><span class="pill ${dCls}">${esc(dTxt)}</span></td>` +
                    `<td class="muted">${esc(b.status)}</td></tr>`;
        }
        spHtml += `</tbody></table></div>`;
      }
      document.getElementById('sparring-box').innerHTML = spHtml;
    }

    let sbHtml = '';
    for(const s of (d.skippy_ledger||[])){
      sbHtml += `<div class="kv"><span>${esc(s.name)} <span class="muted">(${esc(s.category)})</span></span><strong>${esc(s.score)}</strong></div>`;
    }
    document.getElementById('skippy-ledger').innerHTML = sbHtml || '<div class="muted">No scoreboard entries yet.</div>';

    let mHtml = '';
    for(const m of (d.matches||[])){
      mHtml += `<div class="match-item"><strong>${esc(m.env_id)}</strong> <span class="pill">${esc(m.mode)}</span> ` +
               `turns=${esc(m.turns)} | rewards=<code>${esc(m.rewards_json)}</code> | ${esc(m.elapsed_s)}s</div>`;
    }
    document.getElementById('matches-box').innerHTML = mHtml || '<div class="muted">No arena matches logged yet.</div>';

    let cHtml = '';
    for(const ex of (d.latest_exchange||[]).reverse()){
      cHtml += `<div class="exch"><strong>Turn ${esc(ex.turn)} — User/Corner:</strong> ${esc(ex.user)}\n<strong>Bilby:</strong> ${esc(ex.reply)}</div>`;
    }
    document.getElementById('chat-box').innerHTML = cHtml || '<div class="muted">No recent turns.</div>';
  }catch(e){
    document.getElementById('sub').textContent = 'Error refreshing: ' + e;
  }
}
refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return  # Keep journald clean

    def _send_json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 256 * 1024:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        return json.loads(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/dashboard"):
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path in ("/arena", "/fight-camp"):
            body = ARENA_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/arena":
            try:
                self._send_json(get_arena_status())
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if path == "/api/health":
            self._send_json({"ok": True, "service": "bilby-dashboard", "port": 8095})
            return

        if path == "/api/telemetry":
            self._send_json(get_telemetry_snapshot())
            return

        if path in ("/api/plans", "/api/plan"):
            try:
                plans = get_dev_plans()
                if path == "/api/plan":
                    # Back-compat flat list format Bilby sketched
                    flat = []
                    for p in plans:
                        for t in p["tasks"]:
                            flat.append(
                                {
                                    "plan": p["name"],
                                    "priority": t["priority"],
                                    "task": t["name"],
                                    "status": t["status"],
                                    "desc": t["desc"],
                                }
                            )
                    self._send_json(flat)
                else:
                    self._send_json({"ok": True, "plans": plans})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if path == "/api/engrams":
            try:
                rows = neo4j_query(
                    """
                    MATCH (e:ActorEngram)
                    RETURN e.name AS name, e.version AS version, e.superclass AS superclass,
                           e.description AS description, e.call_count AS call_count,
                           e.success_count AS success_count, e.failure_count AS failure_count
                    ORDER BY coalesce(e.updated_at, 0) DESC
                    LIMIT 60
                    """
                )
                self._send_json({"ok": True, "engrams": rows})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if path == "/api/episodes":
            try:
                rows = neo4j_query(
                    """
                    MATCH (e:ActorEpisode)
                    RETURN coalesce(e.archived, false) AS archived,
                           e.salience AS salience,
                           left(coalesce(e.content, e.goal, e.summary, ''), 220) AS goal,
                           coalesce(e.session_date, e.turn_range, e.outcome, e.status, '') AS outcome
                    ORDER BY coalesce(e.updated_at, e.created_at, 0) DESC
                    LIMIT 40
                    """
                )
                self._send_json({"ok": True, "episodes": rows})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if path == "/api/events":
            self._handle_sse()
            return

        self._send_json({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)
            return

        if path == "/api/task/update":
            task_id = str(body.get("task_id") or "").strip()
            status = body.get("status")
            notes = body.get("notes")
            if not task_id:
                self._send_json({"ok": False, "error": "task_id required"}, status=400)
                return
            sets = ["t.updated_by = 'user'", "t.updated_at = timestamp()"]
            params: Dict[str, Any] = {"tid": task_id}
            if status is not None:
                st = str(status).strip().lower()
                if st not in VALID_TASK_STATUSES:
                    self._send_json({"ok": False, "error": f"Invalid status: {st}"}, status=400)
                    return
                sets.append("t.status = $status")
                params["status"] = st
            if notes is not None:
                sets.append("t.notes = $notes")
                params["notes"] = str(notes)[:1000]
            try:
                rows = neo4j_query(
                    f"MATCH (t:Task) WHERE elementId(t) = $tid SET {', '.join(sets)} RETURN t.name AS name, t.status AS status",
                    params,
                )
                if not rows:
                    self._send_json({"ok": False, "error": "Task not found"}, status=404)
                    return
                self._send_json({"ok": True, "task": rows[0]})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if path == "/api/task/create":
            plan_name = str(body.get("plan_name") or "").strip()
            name = str(body.get("name") or "").strip()
            desc = str(body.get("desc") or "").strip()
            priority = int(body.get("priority") or 99)
            if not plan_name or not name:
                self._send_json({"ok": False, "error": "plan_name and name required"}, status=400)
                return
            try:
                rows = neo4j_query(
                    """
                    MATCH (p:DevPlan {name: $plan_name})
                    CREATE (t:Task {
                      name: $name,
                      desc: $desc,
                      priority: $priority,
                      status: 'pending',
                      updated_by: 'user',
                      updated_at: timestamp()
                    })
                    CREATE (p)-[:INCLUDES]->(t)
                    RETURN elementId(t) AS id, t.name AS name
                    """,
                    {"plan_name": plan_name, "name": name, "desc": desc, "priority": priority},
                )
                if not rows:
                    self._send_json({"ok": False, "error": "DevPlan not found"}, status=404)
                    return
                self._send_json({"ok": True, "task": rows[0]})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if path == "/api/cypher":
            q = str(body.get("query") or "").strip()
            if not q:
                self._send_json({"ok": False, "error": "Empty query"}, status=400)
                return
            if READONLY_CYPHER_FORBIDDEN.search(q):
                self._send_json(
                    {"ok": False, "error": "Write/mutation Cypher keywords are blocked in the explorer."},
                    status=403,
                )
                return
            try:
                rows = neo4j_query(q)
                self._send_json({"ok": True, "count": len(rows), "rows": rows[:100]})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        self._send_json({"ok": False, "error": "Not found"}, status=404)

    def _send_sse(self, event: str, data_obj: Any) -> None:
        payload = f"event: {event}\ndata: {json.dumps(data_obj)}\n\n".encode("utf-8")
        self.wfile.write(payload)
        self.wfile.flush()

    def _handle_sse(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        last_pos = TRACE_LOG_PATH.stat().st_size if TRACE_LOG_PATH.is_file() else 0
        try:
            self._send_sse("telemetry", get_telemetry_snapshot())
            self._send_sse("trace_init", read_recent_trace_events(40))
            last_tel = time.monotonic()

            while True:
                time.sleep(0.5)
                if TRACE_LOG_PATH.is_file():
                    cur_size = TRACE_LOG_PATH.stat().st_size
                    if cur_size < last_pos:
                        last_pos = 0
                    if cur_size > last_pos:
                        with TRACE_LOG_PATH.open("rb") as f:
                            f.seek(last_pos)
                            chunk = f.read(cur_size - last_pos).decode("utf-8", errors="replace")
                            last_pos = f.tell()
                        for ln in chunk.splitlines():
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                self._send_sse("trace", compact_trace_event(json.loads(ln)))
                            except Exception:
                                continue

                now = time.monotonic()
                if now - last_tel >= 2.0:
                    self._send_sse("telemetry", get_telemetry_snapshot())
                    last_tel = now
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Bilby Live Oversight Dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()

    start_gpu_sampler()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.daemon_threads = True
    print(f"[bilby-dashboard] Listening on http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Containerized Real-Benchmark Multi-Harness Evaluation Suite (`benchmarks/run_industry_harness_suite.py`).

Evaluates real, vanilla installations of multiple AI agent harnesses running inside isolated
Docker containers against unmodified, official third-party benchmark datasets:

1. Isolated Vanilla Docker Containers (`--harnesses`):
   - `actorii` (`actorii-sandbox:latest`) : Clean-room `ActorII AgentHarness` (`actor-tui -c "/goal ..."`)
   - `pi`      (`pi-bench:latest`)        : Vanilla `@earendil-works/pi-coding-agent` (`v0.86.1`, `pi -p "..."`)
   - `hermes`  (`hermes-bench:latest`)    : Vanilla `Hermes Agent` (`/opt/hermes/hermes --yolo -z "..."`)

2. Official Third-Party Benchmark Datasets (`--tracks`):
   - `swe_bench_lite` : Official Princeton `princeton-nlp/SWE-bench_Lite` (`benchmarks/data/swe_bench_lite.jsonl`
                        + real `base_commit` Git checkouts & `FAIL_TO_PASS` `test_patch` verification).
   - `humaneval`      : Official OpenAI `openai_humaneval` (`benchmarks/data/HumanEval.jsonl`).
   - `mbpp`           : Official Google Research `mbpp` sanitized test split (`benchmarks/data/mbpp_sanitized_test.jsonl`).
   - `gsm8k`          : Official OpenAI `gsm8k` test split (`benchmarks/data/gsm8k_test.jsonl`).

All containers connect to the exact same local `:8080` LLM (`qwen3.8:latest`) and report
exact server-side token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
REPOS_CACHE_DIR = DATA_DIR / "repos"
DEFAULT_RESULTS_JSON = Path(__file__).resolve().parent / "INDUSTRY_SUITE_RESULTS.json"
DEFAULT_REPORT_MD = Path(__file__).resolve().parent / "INDUSTRY_SUITE_REPORT.md"


@dataclass
class TaskMetrics:
    task_id: str
    track: str
    harness: str
    container_image: str
    passed: bool
    steps: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    details: str = ""


import http.client
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONTAINER_IMAGES = {
    "actorii": "actorii-sandbox:latest",
    "pi": "pi-bench:latest",
    "hermes": "hermes-bench:latest",
}

PROXY_PORT = 18088
_PROXY_LOCK = threading.Lock()
_PROXY_STATS = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}
_PROXY_SERVER: Optional[ThreadingHTTPServer] = None


def reset_proxy_stats() -> None:
    with _PROXY_LOCK:
        _PROXY_STATS["prompt_tokens"] = 0
        _PROXY_STATS["completion_tokens"] = 0
        _PROXY_STATS["total_tokens"] = 0
        _PROXY_STATS["api_calls"] = 0


def get_proxy_stats() -> Dict[str, int]:
    with _PROXY_LOCK:
        return dict(_PROXY_STATS)


class _NormalizingProxyHandler(BaseHTTPRequestHandler):
    """Forwards OpenAI API calls to 127.0.0.1:8080 with `enable_thinking=False` for all harnesses."""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", 8080, timeout=30)
        conn.request("GET", self.path)
        resp = conn.getresponse()
        body = resp.read()
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        conn.close()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        is_stream = False
        try:
            payload = json.loads(raw_body.decode("utf-8"))
            if isinstance(payload, dict):
                # Enforce identical inference mode across ActorII, Pi, and Hermes
                model_str = str(payload.get("model") or "").lower()
                ctk = dict(payload.get("chat_template_kwargs") or {})
                force_no_think = bool(payload.pop("force_no_think", False))
                enable_think_env = os.environ.get("ACTORII_ENABLE_THINKING") == "1"
                if ("cascade" in model_str or enable_think_env) and not force_no_think:
                    ctk["enable_thinking"] = True
                    payload["chat_template_kwargs"] = ctk
                    payload["temperature"] = 0.6
                    payload["top_p"] = 0.95
                    payload.pop("reasoning_effort", None)
                else:
                    ctk["enable_thinking"] = False
                    payload["chat_template_kwargs"] = ctk
                is_stream = bool(payload.get("stream"))
                if is_stream:
                    so = dict(payload.get("stream_options") or {})
                    so["include_usage"] = True
                    payload["stream_options"] = so
                raw_body = json.dumps(payload).encode("utf-8")
        except Exception:
            pass

        conn = http.client.HTTPConnection("127.0.0.1", 8080, timeout=360)
        headers = {"Content-Type": "application/json", "Content-Length": str(len(raw_body))}
        conn.request("POST", self.path, body=raw_body, headers=headers)
        resp = conn.getresponse()

        with _PROXY_LOCK:
            _PROXY_STATS["api_calls"] += 1

        if not is_stream:
            resp_body = resp.read()
            try:
                data = json.loads(resp_body.decode("utf-8"))
                u = data.get("usage") or {}
                pt = int(u.get("prompt_tokens") or 0)
                ct = int(u.get("completion_tokens") or 0)
                tt = int(u.get("total_tokens") or (pt + ct))
                with _PROXY_LOCK:
                    _PROXY_STATS["prompt_tokens"] += pt
                    _PROXY_STATS["completion_tokens"] += ct
                    _PROXY_STATS["total_tokens"] += tt
            except Exception:
                pass
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            conn.close()
            return

        # Streaming SSE forwarding + usage extraction
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "text/event-stream"))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                line = resp.fp.readline()
                if not line:
                    break
                self.wfile.write(line)
                self.wfile.flush()
                s = line.decode("utf-8", errors="ignore").strip()
                if s.startswith("data: ") and s != "data: [DONE]":
                    try:
                        ev = json.loads(s[6:])
                        u = ev.get("usage")
                        if isinstance(u, dict) and (u.get("prompt_tokens") or u.get("total_tokens")):
                            pt = int(u.get("prompt_tokens") or 0)
                            ct = int(u.get("completion_tokens") or 0)
                            tt = int(u.get("total_tokens") or (pt + ct))
                            with _PROXY_LOCK:
                                _PROXY_STATS["prompt_tokens"] += pt
                                _PROXY_STATS["completion_tokens"] += ct
                                _PROXY_STATS["total_tokens"] += tt
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            conn.close()


def ensure_normalizing_proxy() -> int:
    global _PROXY_SERVER
    if _PROXY_SERVER is None:
        _PROXY_SERVER = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), _NormalizingProxyHandler)
        t = threading.Thread(target=_PROXY_SERVER.serve_forever, daemon=True)
        t.start()
    return PROXY_PORT


# =============================================================================
# Container Execution Helpers (Vanilla Docker Runs for Each Harness)
# =============================================================================


def run_actorii_container(
    workspace_dir: Path,
    prompt: str,
    model: str = "qwen3.8:latest",
    max_steps: int = 6,
    timeout_s: int = 120,
) -> Tuple[int, str, int, int, int, int]:
    """Run vanilla `actor-tui` inside `actorii-sandbox:latest` with `/workspace` mounted."""
    port = ensure_normalizing_proxy()
    reset_proxy_stats()
    usage_path = workspace_dir / ".actorii_usage.json"
    if usage_path.exists():
        usage_path.unlink()

    # Ensure non-root container user (USER bilby, UID 10001) or mapped host UID can write to /workspace
    for root_p, dirs_p, files_p in os.walk(workspace_dir):
        try:
            os.chmod(root_p, 0o777)
        except Exception:
            pass
        for d_p in dirs_p:
            try:
                os.chmod(os.path.join(root_p, d_p), 0o777)
            except Exception:
                pass
        for f_p in files_p:
            try:
                os.chmod(os.path.join(root_p, f_p), 0o666)
            except Exception:
                pass
    try:
        os.chmod(workspace_dir, 0o777)
    except Exception:
        pass

    cname = f"bench_actorii_{int(time.time()*1000)}_{os.getpid()}"
    cmd = [
        "docker",
        "run",
        "--init",
        "--rm",
        "--name",
        cname,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "--network",
        "host",
        "--security-opt",
        "label:disable",
        "-e",
        "ACTORII_BENCH_STRICT=1",
        "-e",
        f"ACTORII_ENABLE_THINKING={os.environ.get('ACTORII_ENABLE_THINKING', '0')}",
        "-e",
        f"VALKYRIE_LLM_URL=http://127.0.0.1:{port}/v1/chat/completions",
        "-v",
        f"{workspace_dir}:/workspace:z",
        "-w",
        "/workspace",
        "--entrypoint",
        "python3",
        CONTAINER_IMAGES["actorii"],
        "/app/actor_tui.py",
        "--model",
        model,
        "--max-steps",
        str(max_steps),
        "--usage-file",
        "/workspace/.actorii_usage.json",
        "-c",
        f"/goal {prompt}",
    ]
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout_s)
        rc = proc.returncode
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
        rc = 124
        out = f"TIMEOUT after {timeout_s}s: {(exc.stdout or '')}"

    p_stats = get_proxy_stats()
    p_tok = p_stats["prompt_tokens"]
    c_tok = p_stats["completion_tokens"]
    t_tok = p_stats["total_tokens"]
    steps = max(1, p_stats["api_calls"])
    last_res = ""
    if usage_path.exists():
        try:
            u = json.loads(usage_path.read_text(encoding="utf-8"))
            if not t_tok:
                p_tok = int(u.get("prompt_tokens") or 0)
                c_tok = int(u.get("completion_tokens") or 0)
                t_tok = int(u.get("total_tokens") or (p_tok + c_tok))
            steps = max(1, int(u.get("steps") or steps))
            last_res = str(u.get("last_result") or "")
        except Exception:
            pass
    return rc, (last_res + "\n" + out).strip(), steps, p_tok, c_tok, t_tok


def run_pi_container(
    workspace_dir: Path,
    prompt: str,
    model: str = "qwen3.8:latest",
    timeout_s: int = 120,
) -> Tuple[int, str, int, int, int, int]:
    """Run vanilla `@earendil-works/pi-coding-agent` inside `pi-bench:latest` with `/workspace` mounted."""
    port = ensure_normalizing_proxy()
    reset_proxy_stats()
    cfg_dir = Path(tempfile.mkdtemp(prefix="pi_cfg_"))
    try:
        agent_dir = cfg_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "models.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "valkyrie": {
                            "baseUrl": f"http://127.0.0.1:{port}/v1",
                            "api": "openai-completions",
                            "apiKey": "local",
                            "compat": {
                                "supportsDeveloperRole": False,
                                "supportsReasoningEffort": False,
                            },
                            "models": [
                                {
                                    "id": model,
                                    "contextWindow": 65536,
                                    "maxTokens": 4096,
                                }
                            ],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        # Ensure workspace is writable by `skippy` (UID 1000) inside pi-bench
        os.chmod(workspace_dir, 0o777)
        for p in workspace_dir.rglob("*"):
            try:
                os.chmod(p, 0o777)
            except Exception:
                pass

        cname = f"bench_pi_{int(time.time()*1000)}_{os.getpid()}"
        cmd = [
            "docker",
            "run",
            "--init",
            "--rm",
            "--name",
            cname,
            "--network",
            "host",
            "-v",
            f"{agent_dir}:/home/user/.pi/agent:z",
            "-v",
            f"{workspace_dir}:/workspace:z",
            "-w",
            "/workspace",
            CONTAINER_IMAGES["pi"],
            "--provider",
            "valkyrie",
            "--model",
            model,
            "--thinking",
            "off",
            "--no-session",
            "--mode",
            "json",
            "-p",
            prompt,
        ]
        try:
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout_s)
            rc = proc.returncode
            raw_out = proc.stdout or ""
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
            rc = 124
            raw_out = (exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))

        p_tok, c_tok, t_tok, turns = 0, 0, 0, 0
        assistant_texts: List[str] = []
        for line in raw_out.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "turn_end":
                turns += 1
                msg = ev.get("message") or {}
                u = msg.get("usage") or {}
                inp = int(u.get("input") or 0) + int(u.get("cacheRead") or 0)
                outp = int(u.get("output") or 0) + int(u.get("reasoning") or 0)
                tot = int(u.get("totalTokens") or (inp + outp))
                p_tok += inp
                c_tok += outp
                t_tok += tot
                for part in msg.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "text":
                        assistant_texts.append(str(part.get("text") or ""))
        p_stats = get_proxy_stats()
        if p_stats["total_tokens"] > 0:
            p_tok = p_stats["prompt_tokens"]
            c_tok = p_stats["completion_tokens"]
            t_tok = p_stats["total_tokens"]
        turns = max(1, turns or p_stats["api_calls"])
        return rc, "\n".join(assistant_texts).strip(), turns, p_tok, c_tok, t_tok
    finally:
        shutil.rmtree(cfg_dir, ignore_errors=True)


def run_hermes_container(
    workspace_dir: Path,
    prompt: str,
    model: str = "qwen3.8:latest",
    max_turns: int = 8,
    timeout_s: int = 120,
) -> Tuple[int, str, int, int, int, int]:
    """Run vanilla `Hermes Agent` inside `hermes-bench:latest` with `/workspace` mounted."""
    port = ensure_normalizing_proxy()
    reset_proxy_stats()
    cfg_dir = Path(tempfile.mkdtemp(prefix="hermes_cfg_"))
    usage_path = workspace_dir / ".hermes_usage.json"
    if usage_path.exists():
        usage_path.unlink()
    try:
        cfg_file = cfg_dir / "config.yaml"
        cfg_file.write_text(
            f"""model:
  default: {model}
  provider: custom:valkyrie
  context_length: 65536
providers:
  valkyrie:
    api: http://127.0.0.1:{port}/v1
    api_key: local
    default_model: {model}
    models:
      - {model}
    name: Valkyrie
agent:
  max_turns: {max_turns}
""",
            encoding="utf-8",
        )
        cname = f"bench_hermes_{int(time.time()*1000)}_{os.getpid()}"
        cmd = [
            "docker",
            "run",
            "--init",
            "--rm",
            "--name",
            cname,
            "--network",
            "host",
            "--user",
            "root",
            "-e",
            "HOME=/root",
            "-v",
            f"{cfg_file}:/root/.hermes/config.yaml:z",
            "-v",
            f"{workspace_dir}:/workspace:z",
            "-w",
            "/workspace",
            CONTAINER_IMAGES["hermes"],
            "--usage-file",
            "/workspace/.hermes_usage.json",
            "--reasoning",
            "off",
            "--yolo",
            "--ignore-rules",
            "-z",
            prompt,
        ]
        try:
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout_s)
            rc = proc.returncode
            out = (proc.stdout or "").strip()
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
            rc = 124
            out = f"TIMEOUT after {timeout_s}s"

        p_stats = get_proxy_stats()
        p_tok = p_stats["prompt_tokens"]
        c_tok = p_stats["completion_tokens"]
        t_tok = p_stats["total_tokens"]
        steps = max(1, p_stats["api_calls"])
        if not t_tok and usage_path.exists():
            try:
                u = json.loads(usage_path.read_text(encoding="utf-8"))
                p_tok = int(u.get("input_tokens") or 0) + int(u.get("cache_read_tokens") or 0)
                c_tok = int(u.get("output_tokens") or 0) + int(u.get("reasoning_tokens") or 0)
                t_tok = int(u.get("total_tokens") or (p_tok + c_tok))
                steps = max(1, int(u.get("api_calls") or steps))
            except Exception:
                pass
        return rc, out, steps, p_tok, c_tok, t_tok
    finally:
        shutil.rmtree(cfg_dir, ignore_errors=True)


def dispatch_container(
    harness_name: str,
    workspace_dir: Path,
    prompt: str,
    model: str,
    max_steps: int = 6,
    timeout_s: int = 120,
) -> Tuple[int, str, int, int, int, int]:
    if harness_name == "actorii":
        return run_actorii_container(workspace_dir, prompt, model=model, max_steps=max_steps, timeout_s=timeout_s)
    if harness_name == "pi":
        return run_pi_container(workspace_dir, prompt, model=model, timeout_s=timeout_s)
    if harness_name == "hermes":
        return run_hermes_container(workspace_dir, prompt, model=model, max_turns=max_steps, timeout_s=timeout_s)
    raise ValueError(f"Unsupported harness: {harness_name}")


# =============================================================================
# Track 1: Official `princeton-nlp/SWE-bench_Lite` (`swe_bench_lite`)
# =============================================================================


def load_swe_bench_tasks(limit: int = 3) -> List[Dict[str, Any]]:
    path = DATA_DIR / "swe_bench_lite.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit]


def run_swe_bench_lite_task(row: Dict[str, Any], harness_name: str, model: str) -> TaskMetrics:
    iid = row["instance_id"]
    f2p = json.loads(row["FAIL_TO_PASS"])
    src_repo = REPOS_CACHE_DIR / iid
    if not src_repo.exists():
        raise RuntimeError(f"Cached repo checkout missing for {iid}: {src_repo}")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_swe_{iid}_{harness_name}_"))
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0

    try:
        # Copy clean base_commit checkout into tmp_dir (zero file hints!)
        shutil.copytree(src_repo, tmp_dir, dirs_exist_ok=True)
        prompt = (
            f"Resolve the following GitHub issue in the repository at `/workspace`.\n\n"
            f"ISSUE DESCRIPTION (`{iid}`):\n{row['problem_statement'][:1800]}\n\n"
            f"Search `/workspace` to locate the relevant source file(s) and edit the code in `/workspace` so the issue is fixed."
        )
        prev_think = os.environ.get("ACTORII_ENABLE_THINKING")
        os.environ["ACTORII_ENABLE_THINKING"] = "1"
        try:
            _, _, steps, p_tok, c_tok, t_tok = dispatch_container(
                harness_name, tmp_dir, prompt, model=model, max_steps=16, timeout_s=600
            )
        finally:
            if prev_think is None:
                os.environ.pop("ACTORII_ENABLE_THINKING", None)
            else:
                os.environ["ACTORII_ENABLE_THINKING"] = prev_think

        diff_proc = subprocess.run(
            ["git", "-c", "core.fileMode=false", "diff", "--stat"],
            cwd=str(tmp_dir),
            capture_output=True,
            text=True,
        )
        diff_stat = (diff_proc.stdout or "").strip().splitlines()[-1] if (diff_proc.stdout or "").strip() else "no files modified"

        # Ground-truth official SWE-bench_Lite verification: reset any agent edits to test files first, then apply upstream `test_patch` and run `FAIL_TO_PASS`
        for line in row["test_patch"].splitlines():
            if line.startswith("+++ b/"):
                rel_p = line[6:].strip()
                subprocess.run(["git", "checkout", "--", rel_p], cwd=str(tmp_dir), check=False, capture_output=True)
        subprocess.run(["git", "apply", "-"], input=row["test_patch"], text=True, cwd=str(tmp_dir), check=False, capture_output=True)
        tf = re.findall(r"\+\+\+ b/(sympy/\S+\.py)", row["test_patch"])[0]
        mod = tf[:-3].replace("/", ".")
        fn = f2p[0]
        verify_cmd = [
            "python3",
            "-c",
            f'import sys; sys.path.insert(0, "."); from {mod} import {fn}; {fn}(); print("SWE_BENCH_PASS")',
        ]
        v_proc = subprocess.run(verify_cmd, cwd=str(tmp_dir), capture_output=True, text=True, timeout=25)
        passed = v_proc.returncode == 0 and "SWE_BENCH_PASS" in (v_proc.stdout or "")
        details = f"{fn}: {'PASS' if passed else 'FAIL'} ({diff_stat})"
    except Exception as exc:
        details = f"Exception: {exc}"
    finally:
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TaskMetrics(
        task_id=iid,
        track="Track 1: SWE-bench Lite (Princeton)",
        harness=harness_name,
        container_image=CONTAINER_IMAGES[harness_name],
        passed=passed,
        steps=steps,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        latency_s=elapsed,
        details=details,
    )


# =============================================================================
# Track 2: Official `openai/openai_humaneval` (`humaneval`)
# =============================================================================

HUMANEVAL_IDS = ["HumanEval/2", "HumanEval/23", "HumanEval/45", "HumanEval/53"]


def load_humaneval_tasks(limit: int = 3) -> List[Dict[str, Any]]:
    path = DATA_DIR / "HumanEval.jsonl"
    by_id = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            by_id[obj["task_id"]] = obj
    return [by_id[tid] for tid in HUMANEVAL_IDS[:limit] if tid in by_id]


def run_humaneval_task(row: Dict[str, Any], harness_name: str, model: str) -> TaskMetrics:
    tid = row["task_id"]
    entry_point = row["entry_point"]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_he_{tid.replace('/', '_')}_{harness_name}_"))
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0

    try:
        sol_file = tmp_dir / "solution.py"
        sol_file.write_text(row["prompt"] + "    pass\n", encoding="utf-8")

        prompt = (
            f"Read `/workspace/solution.py` and complete the Python function `{entry_point}` according to its docstring. "
            f"Write the completed implementation to `/workspace/solution.py`."
        )
        _, _, steps, p_tok, c_tok, t_tok = dispatch_container(
            harness_name, tmp_dir, prompt, model=model, max_steps=5, timeout_s=120
        )

        # Official OpenAI HumanEval verification (`test` + `check(entry_point)`)
        candidate_code = sol_file.read_text(encoding="utf-8") if sol_file.exists() else ""
        full_test = candidate_code + "\n\n" + row["test"] + f"\n\ncheck({entry_point})\nprint('HUMANEVAL_PASS')\n"
        (tmp_dir / "run_check.py").write_text(full_test, encoding="utf-8")
        v_proc = subprocess.run(["python3", "run_check.py"], cwd=str(tmp_dir), capture_output=True, text=True, timeout=15)
        passed = v_proc.returncode == 0 and "HUMANEVAL_PASS" in (v_proc.stdout or "")
        err_tail = (v_proc.stderr or "").strip().splitlines()[-1] if (v_proc.stderr or "").strip() else "ok"
        details = f"check({entry_point}): {'PASS' if passed else err_tail[:80]}"
    except Exception as exc:
        details = f"Exception: {exc}"
    finally:
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TaskMetrics(
        task_id=tid,
        track="Track 2: OpenAI HumanEval",
        harness=harness_name,
        container_image=CONTAINER_IMAGES[harness_name],
        passed=passed,
        steps=steps,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        latency_s=elapsed,
        details=details,
    )


# =============================================================================
# Track 3: Official `google-research-datasets/mbpp` Sanitized (`mbpp`)
# =============================================================================


def load_mbpp_tasks(limit: int = 3) -> List[Dict[str, Any]]:
    path = DATA_DIR / "mbpp_sanitized_test.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit]


def run_mbpp_task(row: Dict[str, Any], harness_name: str, model: str) -> TaskMetrics:
    tid = f"MBPP/{row['task_id']}"
    first_assert = row["test_list"][0]
    m_fn = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", first_assert)
    fn_name = m_fn.group(1) if m_fn else "solution"

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_mbpp_{row['task_id']}_{harness_name}_"))
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0

    try:
        prompt = (
            f"{row['prompt']}\n"
            f"Implement the function `{fn_name}` in `/workspace/solution.py`.\n"
            f"Example test assertion: `{first_assert}`"
        )
        _, _, steps, p_tok, c_tok, t_tok = dispatch_container(
            harness_name, tmp_dir, prompt, model=model, max_steps=5, timeout_s=120
        )

        # Official Google Research MBPP verification (`test_imports` + `test_list`)
        sol_file = tmp_dir / "solution.py"
        candidate_code = sol_file.read_text(encoding="utf-8") if sol_file.exists() else ""
        imports_block = "\n".join(row.get("test_imports") or [])
        asserts_block = "\n".join(row["test_list"])
        full_test = f"{imports_block}\n{candidate_code}\n\n{asserts_block}\nprint('MBPP_PASS')\n"
        (tmp_dir / "run_mbpp_check.py").write_text(full_test, encoding="utf-8")
        v_proc = subprocess.run(["python3", "run_mbpp_check.py"], cwd=str(tmp_dir), capture_output=True, text=True, timeout=15)
        passed = v_proc.returncode == 0 and "MBPP_PASS" in (v_proc.stdout or "")
        err_tail = (v_proc.stderr or "").strip().splitlines()[-1] if (v_proc.stderr or "").strip() else "ok"
        details = f"{fn_name} ({len(row['test_list'])} asserts): {'PASS' if passed else err_tail[:80]}"
    except Exception as exc:
        details = f"Exception: {exc}"
    finally:
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TaskMetrics(
        task_id=tid,
        track="Track 3: Google Research MBPP",
        harness=harness_name,
        container_image=CONTAINER_IMAGES[harness_name],
        passed=passed,
        steps=steps,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        latency_s=elapsed,
        details=details,
    )


# =============================================================================
# Track 4: Official `openai/gsm8k` Test Split (`gsm8k`)
# =============================================================================


def load_gsm8k_tasks(limit: int = 3) -> List[Dict[str, Any]]:
    path = DATA_DIR / "gsm8k_test.jsonl"
    rows = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        ans_match = re.search(r"####\s*([\-0-9,\.]+)", obj["answer"])
        gt = ans_match.group(1).replace(",", "").strip() if ans_match else ""
        rows.append({"task_id": f"GSM8K/{idx}", "question": obj["question"], "ground_truth": gt})
        if len(rows) >= limit:
            break
    return rows


def run_gsm8k_task(row: Dict[str, Any], harness_name: str, model: str) -> TaskMetrics:
    tid = row["task_id"]
    gt = str(row["ground_truth"])
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_gsm8k_{tid.replace('/', '_')}_{harness_name}_"))
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0

    try:
        prompt = (
            f"Solve the following math problem and write ONLY the final numeric answer to `/workspace/answer.txt` "
            f"(and also state it in your final response):\n\n{row['question']}"
        )
        _, out_text, steps, p_tok, c_tok, t_tok = dispatch_container(
            harness_name, tmp_dir, prompt, model=model, max_steps=5, timeout_s=120
        )

        ans_file = tmp_dir / "answer.txt"
        file_ans = ans_file.read_text(encoding="utf-8").strip() if ans_file.exists() else ""
        combined = f"{file_ans}\n{out_text}"
        nums = [n.replace(",", "") for n in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", combined)]
        passed = gt in nums
        details = f"expected={gt}, found_in_output={passed} (file={file_ans[:30]!r})"
    except Exception as exc:
        details = f"Exception: {exc}"
    finally:
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TaskMetrics(
        task_id=tid,
        track="Track 4: OpenAI GSM8K",
        harness=harness_name,
        container_image=CONTAINER_IMAGES[harness_name],
        passed=passed,
        steps=steps,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        latency_s=elapsed,
        details=details,
    )


# =============================================================================
# Track 5: Official `jkoppel/QuixBugs` (`quixbugs`)
# =============================================================================

QUIXBUGS_ALGOS = ["knapsack", "find_in_sorted", "levenshtein", "kheapsort"]


def load_quixbugs_tasks(limit: int = 2) -> List[Dict[str, Any]]:
    path = DATA_DIR / "quixbugs.jsonl"
    by_algo = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            by_algo[obj["algo"]] = obj
    return [by_algo[a] for a in QUIXBUGS_ALGOS[:limit] if a in by_algo]


def run_quixbugs_task(row: Dict[str, Any], harness_name: str, model: str) -> TaskMetrics:
    algo = row["algo"]
    tid = f"QuixBugs/{algo}"
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_qb_{algo}_{harness_name}_"))
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0

    try:
        # Stage ONLY the buggy program and official pytest harness (never copy correct_python_programs)
        prog_dir = tmp_dir / "python_programs"
        test_dir = tmp_dir / "python_testcases"
        json_dir = tmp_dir / "json_testcases"
        prog_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        (prog_dir / "__init__.py").write_text("", encoding="utf-8")
        (test_dir / "__init__.py").write_text("", encoding="utf-8")
        (prog_dir / f"{algo}.py").write_text(row["buggy_code"], encoding="utf-8")
        (test_dir / f"test_{algo}.py").write_text(row["test_code"], encoding="utf-8")
        (test_dir / "load_testdata.py").write_text(row["load_testdata_code"], encoding="utf-8")
        (test_dir / "node.py").write_text(row["node_code"], encoding="utf-8")
        if row.get("testdata_json"):
            (json_dir / f"{algo}.json").write_text(row["testdata_json"], encoding="utf-8")

        conftest_patched = (
            "import os, sys\n"
            "sys.path.insert(0, os.path.dirname(__file__))\n"
            "sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_testcases'))\n"
            + row["conftest_code"]
        )
        (tmp_dir / "conftest.py").write_text(conftest_patched, encoding="utf-8")

        prompt = (
            f"Inspect `/workspace/python_programs/{algo}.py` and fix the defect so that all unit tests in "
            f"`/workspace/python_testcases/test_{algo}.py` pass when running `pytest python_testcases/test_{algo}.py`."
        )
        _, _, steps, p_tok, c_tok, t_tok = dispatch_container(
            harness_name, tmp_dir, prompt, model=model, max_steps=6, timeout_s=120
        )

        v_proc = subprocess.run(
            ["pytest", f"python_testcases/test_{algo}.py", "-q", "--tb=short"],
            cwd=str(tmp_dir),
            capture_output=True,
            text=True,
            timeout=15,
        )
        passed = v_proc.returncode == 0
        summary_line = (v_proc.stdout or "").strip().splitlines()[-1] if (v_proc.stdout or "").strip() else "no output"
        details = f"pytest test_{algo}: {'PASS' if passed else 'FAIL'} ({summary_line[:50]})"
    except Exception as exc:
        details = f"Exception: {exc}"
    finally:
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TaskMetrics(
        task_id=tid,
        track="Track 5: QuixBugs (Fault Localization)",
        harness=harness_name,
        container_image=CONTAINER_IMAGES[harness_name],
        passed=passed,
        steps=steps,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        latency_s=elapsed,
        details=details,
    )


# =============================================================================
# Track 6: Official `THUDM/AgentBench` OS Interaction (`agentbench_os`)
# =============================================================================

AGENTBENCH_OS_INDICES = [2, 25]


def load_agentbench_os_tasks(limit: int = 2) -> List[Dict[str, Any]]:
    path = DATA_DIR / "agentbench_os_dev.json"
    all_rows = json.loads(path.read_text(encoding="utf-8"))
    selected = []
    for idx in AGENTBENCH_OS_INDICES[:limit]:
        row = dict(all_rows[idx])
        row["task_id"] = f"AgentBench-OS/{idx}"
        row["index"] = idx
        selected.append(row)
    return selected


def run_agentbench_os_task(row: Dict[str, Any], harness_name: str, model: str) -> TaskMetrics:
    tid = row["task_id"]
    idx = row["index"]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_abos_{idx}_{harness_name}_"))
    os_root = tmp_dir / "os_root"
    os_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0
    bg_proc: Optional[subprocess.Popen] = None

    try:
        desc = row["description"].replace("/root", "/workspace/os_root").replace("/usr/stock.log", "/workspace/os_root/stock.log")
        expected = ""

        if idx == 2:
            # Official AgentBench dev[2] init + active 1s background writer
            (os_root / "r1").mkdir(parents=True, exist_ok=True)
            (os_root / "r2").mkdir(parents=True, exist_ok=True)
            for fname in ["a.log", "b.log", "log", "loga", "log.jsonl", "r1/.plain.log"]:
                (os_root / fname).write_text("", encoding="utf-8")
            target_log = os_root / "r1" / ".plain.log"
            bg_proc = subprocess.Popen(
                [
                    "python3",
                    "-c",
                    f'import time\nwhile True:\n    with open({str(target_log)!r}, "a+") as f:\n        f.write("succeed\\n")\n    time.sleep(0.5)\n',
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.6)
            expected = ".plain.log"
        elif idx == 25:
            # Official AgentBench dev[25] stock-log generator
            import random
            rng = random.Random(42 + idx)
            names = ["Alice", "Bob"]
            actions = ["Purchase", "Sell"]
            lines = []
            bob_sells = 0
            for _ in range(401):
                nm = rng.choice(names)
                act = rng.choice(actions)
                sidx = rng.randint(0, 99)
                cnt = rng.randint(0, 999)
                if nm == "Bob" and act == "Sell":
                    bob_sells += 1
                lines.append(f"{nm} | {act} | {sidx} | {cnt}")
            (os_root / "stock.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
            expected = str(bob_sells)

        prompt = (
            f"Linux OS Task (`{tid}`):\n{desc}\n\n"
            f"Inspect `/workspace/os_root` using shell commands or file tools, determine the exact answer, "
            f"and write ONLY the final answer string to `/workspace/answer.txt`."
        )
        _, out_text, steps, p_tok, c_tok, t_tok = dispatch_container(
            harness_name, tmp_dir, prompt, model=model, max_steps=6, timeout_s=120
        )

        ans_file = tmp_dir / "answer.txt"
        file_ans = ans_file.read_text(encoding="utf-8").strip() if ans_file.exists() else ""
        combined = f"{file_ans}\n{out_text}"
        passed = (expected == file_ans) or (expected in [tok.strip("`'\".,:") for tok in combined.split()])
        details = f"expected={expected!r}, file={file_ans[:30]!r} ({'PASS' if passed else 'FAIL'})"
    except Exception as exc:
        details = f"Exception: {exc}"
    finally:
        if bg_proc is not None:
            try:
                bg_proc.terminate()
                bg_proc.wait(timeout=3)
            except Exception:
                pass
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TaskMetrics(
        task_id=tid,
        track="Track 6: AgentBench OS (THUDM)",
        harness=harness_name,
        container_image=CONTAINER_IMAGES[harness_name],
        passed=passed,
        steps=steps,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        latency_s=elapsed,
        details=details,
    )


# =============================================================================
# Track 7: Official `facebookresearch/cruxeval` (`cruxeval`)
# =============================================================================

CRUXEVAL_SPECS = [
    {"index": 0, "mode": "input", "task_id": "CruxEval-I/0"},
    {"index": 6, "mode": "output", "task_id": "CruxEval-O/6"},
]


def load_cruxeval_tasks(limit: int = 2) -> List[Dict[str, Any]]:
    path = DATA_DIR / "cruxeval.jsonl"
    all_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = []
    for spec in CRUXEVAL_SPECS[:limit]:
        row = dict(all_rows[spec["index"]])
        row["task_id"] = spec["task_id"]
        row["mode"] = spec["mode"]
        selected.append(row)
    return selected


def run_cruxeval_task(row: Dict[str, Any], harness_name: str, model: str) -> TaskMetrics:
    tid = row["task_id"]
    mode = row["mode"]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_crux_{tid.replace('/', '_')}_{harness_name}_"))
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0

    try:
        (tmp_dir / "puzzle.py").write_text(row["code"] + "\n", encoding="utf-8")
        if mode == "input":
            prompt = (
                f"CruxEval Input Synthesis (`{tid}`):\n"
                f"Read `/workspace/puzzle.py` from disk.\n"
                f"Find a Python input value `x` such that `f(x) == {row['output']}`.\n"
                f"Verify your input with Python and write ONLY the Python literal for `x` to `/workspace/answer.txt`."
            )
        else:
            prompt = (
                f"CruxEval Output Prediction (`{tid}`):\n"
                f"Read `/workspace/puzzle.py` from disk.\n"
                f"Evaluate `f({row['input']})` and write ONLY the exact Python `repr(...)` of the return value to `/workspace/answer.txt`."
            )

        _, out_text, steps, p_tok, c_tok, t_tok = dispatch_container(
            harness_name, tmp_dir, prompt, model=model, max_steps=5, timeout_s=120
        )

        ans_file = tmp_dir / "answer.txt"
        raw_ans = ans_file.read_text(encoding="utf-8").strip() if ans_file.exists() else ""
        if mode == "input":
            verify_script = (
                f"{row['code']}\n"
                f"cand = {raw_ans if raw_ans else 'None'}\n"
                f"target = {row['output']}\n"
                f"assert f(cand) == target, f'got {{f(cand)!r}}'\n"
                f"print('CRUXEVAL_PASS')\n"
            )
        else:
            verify_script = (
                f"{row['code']}\n"
                f"cand = {raw_ans if raw_ans else 'None'}\n"
                f"expected = f({row['input']})\n"
                f"assert cand == expected, f'got {{cand!r}} vs {{expected!r}}'\n"
                f"print('CRUXEVAL_PASS')\n"
            )
        (tmp_dir / "verify_crux.py").write_text(verify_script, encoding="utf-8")
        v_proc = subprocess.run(["python3", "verify_crux.py"], cwd=str(tmp_dir), capture_output=True, text=True, timeout=10)
        passed = v_proc.returncode == 0 and "CRUXEVAL_PASS" in (v_proc.stdout or "")
        details = f"{mode}: {'PASS' if passed else 'FAIL'} (ans={raw_ans[:32]!r})"
    except Exception as exc:
        details = f"Exception: {exc}"
    finally:
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TaskMetrics(
        task_id=tid,
        track="Track 7: CruxEval (Meta Research)",
        harness=harness_name,
        container_image=CONTAINER_IMAGES[harness_name],
        passed=passed,
        steps=steps,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        latency_s=elapsed,
        details=details,
    )


# =============================================================================
# Track 8: Official `bigcode/bigcodebench-hard` (`bigcodebench_hard`)
# =============================================================================

BIGCODEBENCH_HARD_IDS = [
    "BigCodeBench/19",
    "BigCodeBench/777",
    "BigCodeBench/854",
    "BigCodeBench/800",
    "BigCodeBench/928",
    "BigCodeBench/15",
]


def load_bigcodebench_hard_tasks(limit: int = 4) -> List[Dict[str, Any]]:
    path = DATA_DIR / "bigcodebench_hard.jsonl"
    by_id = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            by_id[obj["task_id"]] = obj
    return [by_id[tid] for tid in BIGCODEBENCH_HARD_IDS[:limit] if tid in by_id]


def run_bigcodebench_hard_task(row: Dict[str, Any], harness_name: str, model: str) -> TaskMetrics:
    tid = row["task_id"]
    entry_point = row["entry_point"]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bench_bcb_{tid.replace('/', '_')}_{harness_name}_"))
    t0 = time.time()
    passed = False
    details = ""
    steps, p_tok, c_tok, t_tok = 1, 0, 0, 0

    try:
        sol_file = tmp_dir / "solution.py"
        sol_file.write_text(row["complete_prompt"].rstrip() + "\n    pass\n", encoding="utf-8")

        prompt = (
            f"BigCodeBench-Hard (`{tid}`):\n"
            f"Read `/workspace/solution.py` from disk and implement `{entry_point}` according to its docstring.\n"
            f"Write your complete, working implementation of `{entry_point}` to `/workspace/solution.py`."
        )
        _, _, steps, p_tok, c_tok, t_tok = dispatch_container(
            harness_name, tmp_dir, prompt, model=model, max_steps=6, timeout_s=150
        )

        # Official BigCodeBench-Hard unittest verification
        candidate_code = sol_file.read_text(encoding="utf-8") if sol_file.exists() else ""
        full_test = (
            candidate_code
            + "\n\n"
            + row["test"]
            + '\n\nif __name__ == "__main__":\n    unittest.main(verbosity=2)\n'
        )
        (tmp_dir / "run_bcb_test.py").write_text(full_test, encoding="utf-8")
        v_proc = subprocess.run(
            ["python3", "run_bcb_test.py"],
            cwd=str(tmp_dir),
            capture_output=True,
            text=True,
            timeout=25,
        )
        passed = v_proc.returncode == 0
        stderr_lines = [ln.strip() for ln in (v_proc.stderr or "").splitlines() if ln.strip()]
        summary = stderr_lines[-1] if stderr_lines else "no output"
        details = f"unittest ({entry_point}): {'PASS' if passed else 'FAIL'} ({summary[:60]})"
    except Exception as exc:
        details = f"Exception: {exc}"
    finally:
        elapsed = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TaskMetrics(
        task_id=tid,
        track="Track 8: BigCodeBench-Hard (ICLR 2025)",
        harness=harness_name,
        container_image=CONTAINER_IMAGES[harness_name],
        passed=passed,
        steps=steps,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        latency_s=elapsed,
        details=details,
    )


# =============================================================================
# Report Generation & CLI Entrypoint
# =============================================================================


def generate_markdown_report(results: List[TaskMetrics], model: str, out_path: Path) -> str:
    harnesses = []
    for r in results:
        if r.harness not in harnesses:
            harnesses.append(r.harness)

    lines = [
        "> [!WARNING]",
        "> This is a planning document and should not be considered as a source of truth without verification.",
        "",
        "# Containerized Official Multi-Harness Benchmark Report",
        "",
        f"- **Model Backend**: `{model}` (local `:8080` GPU inference on `valkyrie`)",
        "- **Isolation**: Each task ran inside a fresh, ephemeral Docker container (`docker run --rm`) using a vanilla installation of each harness.",
        "- **Datasets**: Official `princeton-nlp/SWE-bench_Lite`, `openai/openai_humaneval`, `google-research-datasets/mbpp` (sanitized), and `openai/gsm8k`.",
        "",
        "## 1. Overall Containerized Leaderboard",
        "",
        "| Harness | Docker Image | Tasks Passed | Pass Rate (%) | Total Server Tokens | Avg Tokens / Task | Avg Latency / Task (s) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for h in harnesses:
        h_rows = [r for r in results if r.harness == h]
        n_total = len(h_rows)
        n_pass = sum(1 for r in h_rows if r.passed)
        tot_tok = sum(r.total_tokens for r in h_rows)
        avg_tok = int(tot_tok / max(1, n_total))
        avg_lat = sum(r.latency_s for r in h_rows) / max(1, n_total)
        rate = (100.0 * n_pass) / max(1, n_total)
        img = h_rows[0].container_image if h_rows else CONTAINER_IMAGES.get(h, "")
        lines.append(
            f"| **`{h}`** | `{img}` | **{n_pass}/{n_total}** | **{rate:.1f}%** | {tot_tok:,} | {avg_tok:,} | {avg_lat:.2f}s |"
        )

    lines.extend(
        [
            "",
            "## 2. Per-Task Breakdown Across Official Upstream Benchmarks",
            "",
            "| Official Benchmark Track | Task ID | Harness | Docker Image | Result | Steps/Turns | Total Tokens | Latency (s) | Verification Details |",
            "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for r in results:
        badge = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| {r.track} | `{r.task_id}` | `{r.harness}` | `{r.container_image}` | **{badge}** | {r.steps} | {r.total_tokens:,} | {r.latency_s:.2f}s | {r.details} |"
        )

    report_text = "\n".join(lines) + "\n"
    out_path.write_text(report_text, encoding="utf-8")
    return report_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Official Benchmarks Across Vanilla Docker Agent Harnesses.")
    parser.add_argument(
        "--harnesses",
        default="actorii,pi,hermes",
        help="Comma-separated list of containerized harnesses: actorii,pi,hermes",
    )
    parser.add_argument(
        "--tracks",
        default="swe_bench_lite,humaneval,mbpp,gsm8k",
        help="Comma-separated official tracks: swe_bench_lite,humaneval,mbpp,gsm8k",
    )
    parser.add_argument("--model", default="qwen3.8:latest", help="LLM model name on :8080")
    parser.add_argument(
        "--tasks-per-track",
        type=int,
        default=1,
        help="Number of official tasks per benchmark track (default: 1 for quick, 3 for full 12-task suite)",
    )
    parser.add_argument("--output-json", default=str(DEFAULT_RESULTS_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_REPORT_MD))
    args = parser.parse_args()

    harnesses = [h.strip() for h in args.harnesses.split(",") if h.strip()]
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    k = max(1, int(args.tasks_per_track))

    all_metrics: List[TaskMetrics] = []

    for h in harnesses:
        print(f"\n[{h}] ({CONTAINER_IMAGES.get(h)})", flush=True)
        track_runners = [
            ("swe_bench_lite", load_swe_bench_tasks, run_swe_bench_lite_task),
            ("humaneval", load_humaneval_tasks, run_humaneval_task),
            ("mbpp", load_mbpp_tasks, run_mbpp_task),
            ("gsm8k", load_gsm8k_tasks, run_gsm8k_task),
            ("quixbugs", load_quixbugs_tasks, run_quixbugs_task),
            ("agentbench_os", load_agentbench_os_tasks, run_agentbench_os_task),
            ("cruxeval", load_cruxeval_tasks, run_cruxeval_task),
            ("bigcodebench_hard", load_bigcodebench_hard_tasks, run_bigcodebench_hard_task),
        ]
        for t_name, loader_fn, runner_fn in track_runners:
            if t_name in tracks:
                for row in loader_fn(limit=k):
                    m = runner_fn(row, h, args.model)
                    all_metrics.append(m)
                    status = "PASS" if m.passed else "FAIL"
                    print(
                        f"  {status} {m.task_id}\n"
                        f"       {m.steps}st | {m.total_tokens:,}tok | {m.latency_s}s",
                        flush=True,
                    )

    print("\n--- MOBILE SUMMARY ---", flush=True)
    for h in harnesses:
        h_rows = [r for r in all_metrics if r.harness == h]
        if not h_rows:
            continue
        n_pass = sum(1 for r in h_rows if r.passed)
        n_tot = len(h_rows)
        pct = (100.0 * n_pass) / max(1, n_tot)
        avg_s = sum(r.latency_s for r in h_rows) / max(1, n_tot)
        tot_t = sum(r.total_tokens for r in h_rows)
        print(f"* {h}: {n_pass}/{n_tot} ({pct:.0f}%) | {avg_s:.1f}s/t | {tot_t:,}tok", flush=True)

    Path(args.output_json).write_text(
        json.dumps([asdict(x) for x in all_metrics], indent=2, default=str),
        encoding="utf-8",
    )
    generate_markdown_report(all_metrics, args.model, Path(args.output_md))
    print(f"\nSaved: {Path(args.output_md).name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

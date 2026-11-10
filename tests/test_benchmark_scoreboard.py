"""Tests for the Bilby (ActorII) Benchmark Scoreboard CLI, auto-ingestor, and Studio `/benchmarks` endpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import scoreboard


def test_scoreboard_add_record_promote_and_render(tmp_path: Path) -> None:
    json_file = tmp_path / "scoreboard.json"
    md_file = tmp_path / "BENCHMARK_RESULTS.md"
    html_file = tmp_path / "scoreboard.html"
    readme_file = tmp_path / "README.md"

    readme_file.write_text(
        "# Demo\n\n<!-- BENCHMARK_SCOREBOARD:START -->\nOLD\n<!-- BENCHMARK_SCOREBOARD:END -->\n",
        encoding="utf-8",
    )
    json_file.write_text(
        json.dumps({"metadata": {"title": "Test Board"}, "benchmarks": [], "runs": []}),
        encoding="utf-8",
    )

    data = scoreboard.load_scoreboard(json_file)
    scoreboard.add_benchmark(
        data,
        bench_id="gaia-val",
        name="GAIA Validation",
        category="Memory",
        status="queued",
        upstream_url="https://example.org/gaia",
        description="Multi-step assistant tasks.",
        featured_in_readme=True,
    )
    scoreboard.save_scoreboard(data, json_file)
    scoreboard.render_all(json_file, md_file, html_file, readme_file)

    md_initial = md_file.read_text(encoding="utf-8")
    assert md_initial.startswith("> [!WARNING]\n")
    assert "GAIA Validation" in md_initial
    assert "gaia-val" in md_initial

    # Recording first run automatically promotes queued -> active and syncs README.md
    data = scoreboard.load_scoreboard(json_file)
    run = scoreboard.record_run(
        data,
        bench_id="gaia-val",
        passed=42,
        total=50,
        model="gpt-oss:20b",
        harness_version="v4.2.0",
        run_date="2026-11-10",
        mode="Docker Container",
        submetrics={"Level-1": "20/20", "Level-2": "22/30"},
    )
    assert run["score_pct"] == 84.0
    assert data["benchmarks"][0]["status"] == "active"
    assert data["benchmarks"][0]["total_tasks"] == 50

    scoreboard.save_scoreboard(data, json_file)
    scoreboard.render_all(json_file, md_file, html_file, readme_file)

    readme_after = readme_file.read_text(encoding="utf-8")
    assert "GAIA Validation" in readme_after
    assert "84.00% (42 / 50)" in readme_after

    html_after = html_file.read_text(encoding="utf-8")
    assert "GAIA Validation" in html_after
    assert "gpt-oss:20b" in html_after


def test_scoreboard_auto_extract_result_schemas() -> None:
    # 1. List schema (e.g. INDUSTRY_SUITE_RESULTS.json / NEXTGEN_RESULTS.json)
    m_list = scoreboard.extract_metrics_from_result_json([
        {"suite": "Track 1: SWE-bench Lite (Princeton)", "passed": True, "model": "gpt-oss:20b"},
        {"suite": "Track 1: SWE-bench Lite (Princeton)", "passed": False, "model": "gpt-oss:20b"},
        {"suite": "Track 2: OpenAI HumanEval", "passed": True, "model": "gpt-oss:20b"},
    ])
    assert m_list["passed"] == 2
    assert m_list["total"] == 3
    assert m_list["model"] == "gpt-oss:20b"
    assert m_list["submetrics"]["SWE-bench Lite"] == "1/2"
    assert m_list["submetrics"]["OpenAI HumanEval"] == "1/1"

    # 2. Dict with `results` list (e.g. TB21_5TASK_RESULTS.json)
    m_res = scoreboard.extract_metrics_from_result_json({
        "model": "qwen3.8:latest",
        "results": [{"passed": True}, {"passed": True}, {"passed": False}],
    })
    assert m_res == {"passed": 2, "total": 3, "model": "qwen3.8:latest", "submetrics": {}}

    # 3. Dict with explicit `passed` & `total` (e.g. REDTEAM_CONTAINMENT_RESULTS.json)
    m_red = scoreboard.extract_metrics_from_result_json({
        "model": "qwen3.8:latest",
        "passed": 6,
        "total": 6,
    })
    assert m_red["passed"] == 6 and m_red["total"] == 6

    # 4. Multi-trial (`avg@3` + `pass^3` + `pass@3`) schema for small N <= 10 suites
    m_trials = scoreboard.extract_metrics_from_result_json({
        "model": "qwen3.8:latest",
        "trials": 3,
        "results": [
            {"task_id": "TB2.1/fix-git", "harness": "actorii", "passed": True, "trial": 1},
            {"task_id": "TB2.1/cancel-async-tasks", "harness": "actorii", "passed": False, "trial": 1},
            {"task_id": "TB2.1/fix-git", "harness": "actorii", "passed": True, "trial": 2},
            {"task_id": "TB2.1/cancel-async-tasks", "harness": "actorii", "passed": True, "trial": 2},
            {"task_id": "TB2.1/fix-git", "harness": "actorii", "passed": True, "trial": 3},
            {"task_id": "TB2.1/cancel-async-tasks", "harness": "actorii", "passed": True, "trial": 3},
        ],
    })
    assert m_trials["passed"] == 5
    assert m_trials["total"] == 6
    assert m_trials["trials"] == 3
    assert m_trials["submetrics"]["pass^3"] == "1/2 (50%)"
    assert m_trials["submetrics"]["pass@3"] == "2/2 (100%)"
    assert m_trials["submetrics"]["fix-git"] == "3/3"
    assert m_trials["submetrics"]["cancel-async-tasks"] == "2/3"


def test_scoreboard_multitrial_formatting_and_rendering(tmp_path: Path) -> None:
    json_file = tmp_path / "scoreboard.json"
    md_file = tmp_path / "BENCHMARK_RESULTS.md"
    html_file = tmp_path / "scoreboard.html"
    readme_file = tmp_path / "README.md"
    readme_file.write_text(
        "# Demo\n\n<!-- BENCHMARK_SCOREBOARD:START -->\nOLD\n<!-- BENCHMARK_SCOREBOARD:END -->\n",
        encoding="utf-8",
    )
    json_file.write_text(
        json.dumps({"metadata": {"title": "Test Board"}, "benchmarks": [], "runs": []}),
        encoding="utf-8",
    )
    data = scoreboard.load_scoreboard(json_file)
    scoreboard.add_benchmark(
        data,
        bench_id="tb21-container",
        name="Terminal-Bench 2.1",
        category="Terminal & OS",
        status="active",
        total_tasks=5,
        featured_in_readme=True,
    )
    run = scoreboard.record_run(
        data,
        bench_id="tb21-container",
        passed=14,
        total=15,
        trials=3,
        model="qwen3.8-27b",
        mode="avg@3 Zero-Intervention Sweep",
        submetrics={"pass^3": "4/5 (80%)", "pass@3": "5/5 (100%)", "cancel-async-tasks": "2/3"},
    )
    assert run["trials"] == 3
    assert run["score_pct"] == 93.33
    scoreboard.save_scoreboard(data, json_file)
    scoreboard.render_all(json_file, md_file, html_file, readme_file)

    md_txt = md_file.read_text(encoding="utf-8")
    assert "5 (×3)" in md_txt
    assert "93.33% avg@3 (14 / 15 trials)" in md_txt
    assert "`pass^3`: **4/5 (80%)**" in md_txt

    readme_txt = readme_file.read_text(encoding="utf-8")
    assert "93.33% avg@3 (14 / 15 trials)" in readme_txt

    html_txt = html_file.read_text(encoding="utf-8")
    assert "avg@" in html_txt
    assert "pass^3" in html_txt


def test_studio_serves_benchmarks_page_and_json_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from websockets.datastructures import Headers
    from websockets.http11 import Request
    import websocket_server

    monkeypatch.delenv("ACTORII_STUDIO_TOKEN", raising=False)

    async def _check() -> None:
        resp_html = await websocket_server.process_http_request(
            None, Request("/benchmarks", Headers([("Host", "127.0.0.1:8090")]), method="GET")
        )
        assert resp_html is not None and resp_html.status_code == 200
        assert b"Active Benchmark Scoreboard" in resp_html.body

        resp_api = await websocket_server.process_http_request(
            None, Request("/api/benchmarks", Headers([("Host", "127.0.0.1:8090")]), method="GET")
        )
        assert resp_api is not None and resp_api.status_code == 200
        payload = json.loads(resp_api.body.decode("utf-8"))
        assert len(payload["benchmarks"]) >= 9
        assert len(payload["runs"]) >= 15

        # Cross-origin blocked
        resp_evil = await websocket_server.process_http_request(
            None,
            Request("/api/benchmarks", Headers([("Host", "127.0.0.1:8090"), ("Origin", "https://evil.example")]), method="GET"),
        )
        assert resp_evil is not None and resp_evil.status_code == 403

        # Token protected when ACTORII_STUDIO_TOKEN is configured
        monkeypatch.setenv("ACTORII_STUDIO_TOKEN", "bench-secret")
        resp_unauth = await websocket_server.process_http_request(
            None, Request("/benchmarks", Headers([("Host", "127.0.0.1:8090")]), method="GET")
        )
        assert resp_unauth is not None and resp_unauth.status_code == 401

        resp_auth = await websocket_server.process_http_request(
            None, Request("/benchmarks?token=bench-secret", Headers([("Host", "127.0.0.1:8090")]), method="GET")
        )
        assert resp_auth is not None and resp_auth.status_code == 200

    asyncio.run(_check())

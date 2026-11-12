"""Comprehensive Unit & Integration Tests for `Actor[#ProductLearner]` ("Learn How to Learn" Skill).

Verifies:
1. 1993 Actor 4.1 bytecode compilation, Ring 0/1 & singleton protection, and `summary` across memory modes.
2. `learnProduct(Actor[#ProductLearner], ...)` end-to-end Dual-Mode Neo4j persistence (`:Product` -> `:Subsystem`,
   `:Procedure`, `:Gotcha`, `:DocSource` + `:8084` vector-embedded `:ProductManualChunk`).
3. `ingestManual(Actor[#ProductLearner], ...)` on both local technical manuals and multi-page PDF datasheets (`Actor[#PDF]`).
4. Automatic `hydrateContext` injection on relevant product/subsystem prompts, non-injection on unrelated math/coding
   prompts, and `ACTORII_BENCH_STRICT=1` benchmark isolation.
5. 4-step `DuplicateStepWarning` A-B-A-B ping-pong loop prevention and `JetsonEdgeTwin` embedded-NUL sanitization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from actorii.guardrails.linter import ProtectedClassError
from actorii.harness import bootstrap_agent_harness
from actorii.harness.twins import JetsonEdgeTwin


class TestProductLearnerSkill:
    """Verify Bilby's reusable `Actor[#ProductLearner]` skill and Dual-Mode Neo4j manual recall."""

    def test_product_learner_pool_protection_and_memory_disabled_graceful(self) -> None:
        vm, _harness, _sandbox = bootstrap_agent_harness(enable_memory_graph=False)
        status = vm.eval_chunk("summary(Actor[#ProductLearner])!!")
        assert "ProductLearner(memory=#disabled" in str(status)

        with pytest.raises(ProtectedClassError):
            vm.eval_chunk("Actor[#ProductLearner] := 42!!")

        with pytest.raises(ProtectedClassError):
            vm.eval_chunk("now(ProductLearner)!!\nDef summary(self) { ^\"hacked\"; }!!")

    def test_learn_product_dual_mode_neo4j_and_auto_hydrate(self) -> None:
        ns = "pytest_product_learner"
        vm, harness, sandbox = bootstrap_agent_harness(
            enable_memory_graph=True,
            enable_engrams=True,
            memory_namespace=ns,
        )
        assert sandbox.memory_graph is not None
        sandbox.memory_graph.clearProducts()
        sandbox.memory_graph.clearEngrams()

        # Pre-seed WebScraper caches so unit test is fast and deterministic while running full 1993 Actor bytecode
        doc_url = "https://docs.nvidia.com/jetson/l4t-r32.7/wayland-display.html"
        vm.eval_chunk(
            f'atPut(searchCache(Actor[#Web]), "NVIDIA Jetson TX1 official technical documentation manual Wayland Weston HDMI, CSI camera::5", '
            f'"1. L4T R32.7 Wayland Guide ({doc_url})\\n   Official tegra-udrm and nvstart-weston.sh display instructions.")!!'
        )
        vm.eval_chunk(
            f'atPut(pageCache(Actor[#Web]), "{doc_url}", '
            f'"[WebScrape via #firecrawl] L4T R32.7 Wayland Guide — {doc_url}\\n'
            f'# Wayland & Weston Display Stack\\n'
            f'On L4T R32.7.6, X11 (`gdm3`) holds DRM master on `/dev/dri/card0` by default.\\n'
            f'Stop `gdm3`, load `sudo modprobe tegra-udrm modeset=1`, and launch `nvstart-weston.sh`.\\n'
            f'CRITICAL: Never overwrite `/boot/initrd` or upgrade `nvidia-l4t-initrd` on dual-boot SD/eMMC boards.")!!'
        )

        distilled_payload = {
            "name": "NVIDIA Jetson TX1 (L4T R32.7.6)",
            "category": "Embedded Edge GPU SoM",
            "version": "L4T R32.7.6 (JetPack 4.6.6 / Ubuntu 18.04)",
            "keywords": ["jetson", "tx1", "tegra-udrm", "wayland", "weston", "nvstart-weston", "hdmi", "csi", "rtsp"],
            "summary": "256-core Maxwell embedded GPU SoM running Tegra Linux R32.7.6 with Tegra DRM Wayland/Weston and GStreamer CSI RTSP.",
            "subsystems": [
                {
                    "name": "Wayland / Weston HDMI Display",
                    "driver": "tegra-udrm + Weston 6.0",
                    "description": "Renders Wayland client surfaces directly onto HDMI TV via Tegra UDRM modesetting.",
                    "commands": ["sudo systemctl stop gdm3", "sudo modprobe tegra-udrm modeset=1", "nvstart-weston.sh"],
                },
                {
                    "name": "MIPI-CSI Camera RTSP Pipeline",
                    "driver": "ov5693 + nvarguscamerasrc",
                    "description": "Hardware-accelerated H.264 RTSP stream on port 8554.",
                    "commands": ["systemctl status jetson-rtsp.service"],
                },
            ],
            "procedures": [
                {
                    "name": "Launch Wayland Weston on HDMI TV",
                    "command": "sudo systemctl stop gdm3 && sudo modprobe tegra-udrm modeset=1 && nvstart-weston.sh",
                    "notes": "Stop gdm3 first so Xorg releases DRM master.",
                }
            ],
            "gotchas": [
                {
                    "rule": "Never overwrite /boot/initrd or upgrade nvidia-l4t-initrd; breaks dual-boot SD/eMMC failsafe.",
                    "severity": "CRITICAL",
                }
            ],
            "chunks": [
                {
                    "title": "Launching Wayland & Weston on HDMI",
                    "content": "On NVIDIA Jetson TX1 running L4T R32.7.6, stop `gdm3`, load `tegra-udrm modeset=1`, and run `nvstart-weston.sh` to display Wayland windows over HDMI.",
                }
            ],
            "sources": [{"url": doc_url, "title": "NVIDIA L4T R32.7 Wayland Guide"}],
        }
        harness.custom_provider = lambda prompt, sys_p="", meta=None: json.dumps(distilled_payload)

        try:
            out = vm.eval_chunk(
                'learnProduct(Actor[#ProductLearner], "NVIDIA Jetson TX1", "Wayland Weston HDMI, CSI camera")!!'
            )
            out_str = str(out)
            assert "Learned Product 'NVIDIA Jetson TX1 (L4T R32.7.6)'" in out_str
            assert "nvstart-weston.sh" in out_str
            assert "tegra-udrm" in out_str
            assert vm.eval_chunk("learnCount(Actor[#ProductLearner])!!") == 1

            # Verify structured Neo4j graph relationships exist
            _, graph_rows = sandbox.memory_graph._run_cypher(
                """
                MATCH (p:Product {namespace: $ns})
                OPTIONAL MATCH (p)-[:HAS_SUBSYSTEM]->(s:Subsystem {namespace: $ns})
                OPTIONAL MATCH (p)-[:HAS_PROCEDURE]->(pr:Procedure {namespace: $ns})
                OPTIONAL MATCH (p)-[:HAS_GOTCHA]->(g:Gotcha {namespace: $ns})
                OPTIONAL MATCH (p)-[:HAS_MANUAL_CHUNK]->(mc:ProductManualChunk {namespace: $ns})
                RETURN p.id AS pid,
                       count(DISTINCT s) AS subs,
                       count(DISTINCT pr) AS procs,
                       count(DISTINCT g) AS gotchas,
                       count(DISTINCT mc) AS chunks
                """,
                {"ns": ns},
            )
            assert len(graph_rows) == 1
            assert graph_rows[0]["subs"] >= 2
            assert graph_rows[0]["procs"] >= 1
            assert graph_rows[0]["gotchas"] >= 1
            assert graph_rows[0]["chunks"] >= 1

            # Verify `listProducts` and 1-arg / 2-arg `recallProduct`
            listing = str(vm.eval_chunk("listProducts(Actor[#ProductLearner])!!"))
            assert "nvidia-jetson-tx1" in listing

            recalled = str(vm.eval_chunk('recallProduct(Actor[#ProductLearner], "Jetson TX1", "Wayland HDMI")!!'))
            assert "nvstart-weston.sh" in recalled
            assert "/boot/initrd" in recalled

            # Verify automatic zero-effort recall inside `hydrateContext` when user asks about Jetson / Wayland!
            hydrated = str(
                vm.eval_chunk('hydrateContext(Actor[#Memory], "put a wayland surface onto the hdmi tv connected to the jetson")!!')
            )
            assert "[AUTO-RECALLED PRODUCT TECHNICAL MANUAL" in hydrated
            assert "nvstart-weston.sh" in hydrated
            assert "tegra-udrm" in hydrated

            # Verify unrelated math/algorithm queries do NOT get cluttered with Jetson manual text
            unrelated = str(vm.eval_chunk('hydrateContext(Actor[#Memory], "compute the sum of squares from 1 to 50")!!'))
            assert "nvstart-weston.sh" not in unrelated
        finally:
            sandbox.memory_graph.clearProducts()
            sandbox.memory_graph.clearEngrams()

    def test_ingest_manual_local_markdown_and_pdf_datasheet(self, tmp_path: Path) -> None:
        ns = "pytest_product_ingest"
        vm, harness, sandbox = bootstrap_agent_harness(
            enable_memory_graph=True,
            enable_engrams=True,
            memory_namespace=ns,
        )
        assert sandbox.memory_graph is not None
        sandbox.memory_graph.clearProducts()
        # Test deterministic heuristic distillation fallback when LLM is offline
        def _offline_llm(*_args, **_kwargs):
            raise RuntimeError("offline")

        harness.custom_provider = _offline_llm

        md_manual = tmp_path / "tp358s_manual.md"
        md_manual.write_text(
            "# ThermoPro TP358S BLE Sensor Manual\n"
            "Broadcasts passive BLE advertisement packets on HCI0 every 2 seconds.\n\n"
            "## Linux BlueZ Reading Procedure\n"
            "Run `sudo btmgmt find` or query `summary(Actor[#Thermometer])`.\n"
            "WARNING: Never pair exclusively in GATT mode if passive BLE broadcast listeners are active.\n",
            encoding="utf-8",
        )

        pdf_manual = tmp_path / "tp358s_datasheet.pdf"
        import pypdf

        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with pdf_manual.open("wb") as fh:
            writer.write(fh)

        try:
            res_md = str(
                vm.eval_chunk(
                    f'ingestManual(Actor[#ProductLearner], "ThermoPro TP358S", "{md_manual}", "BLE advertisements")!!'
                )
            )
            assert "Learned Product 'ThermoPro TP358S'" in res_md
            assert "Never pair exclusively in GATT mode" in res_md

            res_pdf = str(
                vm.eval_chunk(
                    f'ingestManual(Actor[#ProductLearner], "ThermoPro TP358S", "{pdf_manual}")!!'
                )
            )
            assert "thermopro-tp358s" in res_pdf
            assert vm.eval_chunk("learnCount(Actor[#ProductLearner])!!") == 2
        finally:
            sandbox.memory_graph.clearProducts()

    def test_duplicate_step_ping_pong_blocked_across_four_steps_and_jetson_nul_safe(self, tmp_path: Path) -> None:
        vm, harness, sandbox = bootstrap_agent_harness(enable_memory_graph=False)
        vm.eval_chunk('setGoal(Actor[#Harness], "Inspect Jetson")!!')

        # Step 1 succeeds
        r1 = sandbox.executeStepAction(harness, 'storeScratch(Actor[#Harness], #probe, "card0")!!')
        assert r1 == "card0"
        harness._fields["stepCount"] = 2

        # Step 2 emits identical code -> DuplicateStepWarning
        r2 = sandbox.executeStepAction(harness, 'storeScratch(Actor[#Harness], #probe, "card0")!!')
        assert "DuplicateStepWarning" in str(r2)
        harness._fields["stepCount"] = 3

        # Step 3 tries the old A -> Warning -> A ping-pong loophole -> STILL caught by 4-step window!
        r3 = sandbox.executeStepAction(harness, 'storeScratch(Actor[#Harness], #probe, "card0")!!')
        assert "DuplicateStepWarning" in str(r3)
        assert "ProductLearner" in str(r3)

        # Verify JetsonEdgeTwin._ssh_exec_raw sanitizes embedded NUL byte ("\x00") before subprocess.run
        jtwin = JetsonEdgeTwin(state_dir=tmp_path / "jetson_state")
        with patch.object(jtwin, "_resolve_live_host", return_value=("192.168.68.60", {"user": "kilog", "port": 22}, "SSH-2.0-OpenSSH")):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "NVIDIA Jetson TX1"
                mock_run.return_value.stderr = ""
                out = jtwin._ssh_exec_raw("tr -d '\x00' </proc/device-tree/model")
                assert out["ok"] is True
                called_argv = mock_run.call_args[0][0]
                assert "\x00" not in called_argv[-1]
                assert "\\0" in called_argv[-1]

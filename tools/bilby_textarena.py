#!/usr/bin/env python3
"""
Bilby TextArena Runner (`bilby_textarena.py`)
=============================================
Run Bilby's local :8080 model (Qwen3.8 27B) on any of TextArena's 712
single-player, two-player, or multi-player text games—locally, against
deterministic algorithmic sparring bots (`minimax_ttt`, `tactical_c4`,
`nash_neg`), with post-fight RL Film-Study & Rematch loops, or on the
live textarena.ai online TrueSkill arena.

Usage (via ActorII/.venv-textarena):
  # Local 1-player match:
  .venv-textarena/bin/python tools/bilby_textarena.py --env GuessTheNumber-v0 --players 1

  # Local 2-player sparring vs deterministic algorithmic bot with film study + rematch:
  .venv-textarena/bin/python tools/bilby_textarena.py --env TicTacToe-v0 --players 2 \\
      --use-actorii --opponent-bot minimax_ttt --film-study --rematch

  # Live textarena.ai online arena:
  .venv-textarena/bin/python tools/bilby_textarena.py --online --env TicTacToe-v0 \\
      --model-name "Bilby-Qwen3.8-27B" --email "you@example.com"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import textarena as ta
except ImportError:  # Allows importing helpers/tests outside .venv-textarena
    ta = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = Path(__file__).resolve().parent
for _p in (SRC_DIR, TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sparring_opponents import SPARRING_BOT_REGISTRY, create_sparring_bot  # noqa: E402

NEO4J_HTTP_TX_URL = os.environ.get(
    "ACTORII_NEO4J_HTTP_URL",
    "http://127.0.0.1:7474/db/neo4j/tx/commit",
)
DEFAULT_STRATEGY_NAMESPACE = "textarena_bilby"
DEFAULT_SPARRING_RESULTS_PATH = REPO_ROOT / "benchmarks" / "SPARRING_RESULTS.json"


class LocalOpenAIAgent:
    """Lightweight OpenAI-compatible chat agent pointing at local :8080 / :8082 / :11434."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "qwen3.8:latest",
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 384,
        timeout: float = 120.0,
        enable_thinking: bool = False,
        injected_lessons: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.base_system_prompt = system_prompt or (
            "You are Bilby, a sharp competitive game-playing AI agent. "
            "Read the game rules and observation carefully. Follow the exact required "
            "action format (such as bracketed moves like `[42]` or `[e2 e4]`). "
            "Keep your reply concise and ALWAYS include your valid formatted move."
        )
        self.injected_lessons = injected_lessons.strip()

    @property
    def system_prompt(self) -> str:
        if not self.injected_lessons:
            return self.base_system_prompt
        return (
            f"{self.base_system_prompt}\n\n"
            f"FILM-STUDY LESSONS FOR REMATCH:\n{self.injected_lessons}"
        )

    def inject_lessons(self, lessons: str) -> None:
        self.injected_lessons = str(lessons or "").strip()

    def reset_for_rematch(self, lessons: str = "") -> None:
        if lessons:
            self.inject_lessons(lessons)

    def _chat(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if not self.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        msg = (body.get("choices") or [{}])[0].get("message") or {}
        content = str(msg.get("content") or "").strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if not content:
            reasoning = str(msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
            bracket_moves = re.findall(r"\[[^\]]+\]", reasoning)
            if bracket_moves:
                content = bracket_moves[-1]
        return content

    def reflect(self, reflection_prompt: str) -> str:
        return self._chat(
            "You are Bilby performing post-match RL film study. Be concise, concrete, and tactical.",
            reflection_prompt,
            max_tokens=512,
        )

    def __call__(self, observation: str) -> str:
        return self._chat(self.system_prompt, observation)


class ActorIIGameAgent:
    """Full Bilby (`ActorII` VM + `AgentHarness` + Neo4j) competitive TextArena agent."""

    def __init__(
        self,
        *,
        env_id: str,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "qwen3.8:latest",
        enable_thinking: bool = False,
        max_steps: int = 3,
        injected_lessons: str = "",
        namespace: str = DEFAULT_STRATEGY_NAMESPACE,
    ) -> None:
        os.environ["ACTORII_HOST_LOCKDOWN"] = "1"
        os.environ["ACTORII_BENCH_STRICT"] = "1"
        os.environ["ACTORII_LLM_ENDPOINT"] = f"{base_url.rstrip('/')}/chat/completions"
        os.environ["ACTORII_LLM_MODEL"] = model
        os.environ["ACTORII_THINKING_MODE"] = "medium" if enable_thinking else "off"
        os.environ["ACTORII_ENABLE_THINKING"] = "1" if enable_thinking else "0"

        from actorii.harness import bootstrap_agent_harness

        self.env_id = env_id
        self.model = model
        self.namespace = namespace
        self.move_history: List[str] = []
        self.turn_idx = 0
        self.prior_lessons = self._recall_prior_matches(env_id)
        self.prior_strategies = recall_strategies_neo4j(env_id, namespace=namespace)
        self.injected_lessons = injected_lessons.strip()
        self.vm, self.harness, self.sandbox = bootstrap_agent_harness(
            model=model,
            max_steps=max_steps,
            gas_limit=20000,
            enable_guardrails=True,
            memory_namespace=namespace,
        )

    def inject_lessons(self, lessons: str) -> None:
        self.injected_lessons = str(lessons or "").strip()
        try:
            self.vm.global_vars["_TAFilmLessons"] = self.injected_lessons
            self.vm.eval_chunk("storeScratch(Actor[#Harness], #taFilmLessons, _TAFilmLessons)!!")
        except Exception:
            pass

    def reset_for_rematch(self, lessons: str = "") -> None:
        self.move_history = []
        self.turn_idx = 0
        if lessons:
            self.inject_lessons(lessons)
        else:
            self.prior_strategies = recall_strategies_neo4j(self.env_id, namespace=self.namespace)

    def reflect(self, reflection_prompt: str) -> str:
        self.vm.global_vars["_TAGoal"] = (
            f"{reflection_prompt}\n\n"
            "Reply immediately with `COMPLETE: <2-3 concise tactical bullet lessons>`."
        )
        try:
            raw_reply = str(self.vm.eval_chunk("runGoal(Actor[#Harness], _TAGoal)!!") or "").strip()
        except Exception as exc:
            raw_reply = f"ERROR: {exc}"
        cleaned = re.sub(r"^\s*COMPLETE:\s*", "", raw_reply, flags=re.IGNORECASE).strip()
        return cleaned

    @staticmethod
    def _recall_prior_matches(env_id: str) -> str:
        stmt = (
            "MATCH (m:TextArenaMatch {env_id: $env_id}) "
            "RETURN m.mode, m.turns, m.rewards_json ORDER BY m.created_at DESC LIMIT 3"
        )
        try:
            payload = json.dumps({"statements": [{"statement": stmt, "parameters": {"env_id": env_id}}]}).encode("utf-8")
            req = urllib.request.Request(
                NEO4J_HTTP_TX_URL,
                data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            rows = (data.get("results") or [{}])[0].get("data") or []
            if not rows:
                return ""
            snippets = [f"mode={r['row'][0]}, turns={r['row'][1]}, rewards={r['row'][2]}" for r in rows]
            return "Prior Neo4j match history for this game: " + " | ".join(snippets)
        except Exception:
            return ""

    @staticmethod
    def _extract_move(reply: str) -> str:
        text = str(reply or "").strip()
        brackets = re.findall(r"\[[^\]]+\]", text)
        if brackets:
            return brackets[-1].strip()
        return text

    def _build_goal(self, observation: str, hist_str: str) -> str:
        lesson_blocks: List[str] = []
        if self.prior_lessons:
            lesson_blocks.append(self.prior_lessons)
        if self.prior_strategies:
            lesson_blocks.append(f"Stored Neo4j Tactical Strategies:\n{self.prior_strategies}")
        if self.injected_lessons:
            lesson_blocks.append(f"FILM-STUDY LESSONS FOR REMATCH (APPLY IMMEDIATELY):\n{self.injected_lessons}")
        lessons_section = "\n\n".join(lesson_blocks)

        return (
            f"You are Bilby playing competitive TextArena game `{self.env_id}` (Turn {self.turn_idx}).\n"
            f"Your previously played moves this match (DO NOT repeat an already-tried guess in deduction games): {hist_str}\n"
            f"{lessons_section}\n\n"
            "GAME OBSERVATION:\n"
            f"{observation}\n\n"
            "STRATEGY & OUTPUT RULES:\n"
            "- Think strategically (in negotiation, make constructive counter-offers instead of endlessly denying; "
            "in deduction/Wordle, strictly respect every green/yellow/gray letter constraint and NEVER repeat a prior guess; "
            "in board games, check for immediate winning moves and block opponent threats).\n"
            "- You may use 1 quick Actor 4.1 computation if needed, OR reply immediately with `COMPLETE: <exact bracketed move>` "
            "(e.g., `COMPLETE: [crane]` or `COMPLETE: [col 3]` or `COMPLETE: [Offer: 1 Wheat -> 1 Ore]`).\n"
            "- Your `COMPLETE:` reply MUST contain your exact valid bracketed move."
        )

    def __call__(self, observation: str) -> str:
        self.turn_idx += 1
        hist_str = ", ".join(self.move_history[-12:]) if self.move_history else "(none yet)"
        try:
            self.vm.global_vars["_TAHist"] = hist_str
            self.vm.eval_chunk("storeScratch(Actor[#Harness], #taMoveHistory, _TAHist)!!")
        except Exception:
            pass

        goal = self._build_goal(observation, hist_str)
        self.vm.global_vars["_TAGoal"] = goal
        try:
            raw_reply = str(self.vm.eval_chunk("runGoal(Actor[#Harness], _TAGoal)!!") or "").strip()
        except Exception as exc:
            raw_reply = f"ERROR: {exc}"
        move = self._extract_move(raw_reply)
        if move:
            self.move_history.append(move)
        return move


# ---------------------------------------------------------------------------
# Film-Study Transcript, Neo4j Strategy Persistence & Sparring Results
# ---------------------------------------------------------------------------


def _extract_p0_reward(rewards: Any) -> Optional[float]:
    if isinstance(rewards, dict):
        if 0 in rewards:
            return float(rewards[0])
        if "0" in rewards:
            return float(rewards["0"])
    return None


def classify_p0_outcome(rewards: Any) -> str:
    r0 = _extract_p0_reward(rewards)
    if r0 is None:
        return "UNKNOWN"
    if r0 > 0:
        return "WIN"
    if r0 < 0:
        return "LOSS"
    return "DRAW"


def format_match_transcript(
    env_id: str,
    transcript: List[Dict[str, Any]],
    rewards: Any,
    opponent_name: str = "",
    max_turns_shown: int = 40,
    reason: str = "",
) -> str:
    """Format a compact match transcript + outcome for post-fight RL film study."""
    outcome = classify_p0_outcome(rewards)
    opp_label = opponent_name or "opponent"
    reason_part = f" | Reason: {reason.strip()}" if str(reason or "").strip() else ""
    header = (
        f"Match Film-Study Transcript | Env: {env_id} | Opponent: {opp_label} | "
        f"Turns: {len(transcript)} | P0 Outcome: {outcome} | Rewards: {json.dumps(rewards, default=str)}{reason_part}"
    )
    lines = [header, "-" * min(len(header), 88)]
    shown = transcript[-max_turns_shown:]
    for entry in shown:
        turn = entry.get("turn", "?")
        pid = entry.get("player_id", "?")
        action = str(entry.get("action", "")).replace("\n", " ").strip()
        if len(action) > 140:
            action = action[:137] + "..."
        role = "Bilby(P0)" if str(pid) == "0" else f"Opp(P{pid})"
        lines.append(f"T{turn:02d} {role}: {action}")
    return "\n".join(lines)


def build_film_study_prompt(compact_transcript: str, env_id: str, outcome: str = "") -> str:
    norm_outcome = (outcome or "").upper()
    if not norm_outcome:
        if "P0 Outcome: WIN" in compact_transcript:
            norm_outcome = "WIN"
        elif "P0 Outcome: LOSS" in compact_transcript:
            norm_outcome = "LOSS"
        elif "P0 Outcome: DRAW" in compact_transcript:
            norm_outcome = "DRAW"

    if norm_outcome == "WIN":
        directive = (
            "Bilby (P0) WON this match! Do NOT invent a fake mistake or abandon the winning sequence.\n"
            "Analyze Bilby's (P0) winning play above and identify:\n"
            "1. The exact winning tactic or fork that secured victory for P0.\n"
            "2. 2-3 concrete, repeatable tactical rules to KEEP and sharpen in the Rematch."
        )
    else:
        directive = (
            f"Bilby (P0) did NOT win this match (P0 Outcome: {norm_outcome or 'UNKNOWN'}).\n"
            "Analyze Bilby's (P0) play in the match above and identify:\n"
            "1. The biggest tactical mistake or missed opportunity by P0.\n"
            "2. 2-3 concrete, actionable tactical rules to fix immediately in the Rematch."
        )
    return (
        f"Post-Fight RL Film Study for `{env_id}`:\n"
        f"{compact_transcript}\n\n"
        f"{directive}\n"
        "Keep it crisp and tactical (no fluff)."
    )


def synthesize_deterministic_lessons(
    env_id: str,
    transcript: List[Dict[str, Any]],
    rewards: Any,
    opponent_name: str = "",
) -> str:
    """Deterministic fallback tactical lessons when running offline or without live reflection."""
    outcome = classify_p0_outcome(rewards)
    env_low = env_id.lower()
    if "tictactoe" in env_low:
        return (
            f"- Outcome vs {opponent_name or 'opponent'}: {outcome}. "
            "Always claim center [4] if open, or a corner [0]/[2]/[6]/[8] when center is taken.\n"
            "- Before playing any square, scan all 8 lines for an immediate winning move or mandatory opponent 2-in-a-row block.\n"
            "- Avoid edge moves early; they allow opponent forks."
        )
    if "connectfour" in env_low:
        return (
            f"- Outcome vs {opponent_name or 'opponent'}: {outcome}. "
            "Prioritize center column [col 3] early to maximize horizontal and diagonal 4-in-a-row threats.\n"
            "- Always check for immediate winning drops first, then block any opponent 3-in-a-row threat.\n"
            "- Do not drop into a column if the cell directly above hands the opponent an immediate win."
        )
    if "negotiation" in env_low:
        return (
            f"- Outcome vs {opponent_name or 'opponent'}: {outcome}. "
            "Calculate received_value vs given_value using your private resource table before deciding.\n"
            "- Accept any feasible offer with positive net value; otherwise issue a constructive [Offer: ...] trading low-value surplus for high-value needs.\n"
            "- Never waste turns emitting blind [Deny] without a counter-offer."
        )
    return (
        f"- Outcome vs {opponent_name or 'opponent'}: {outcome} in {len(transcript)} turns.\n"
        "- Track opponent patterns from the transcript and verify action formatting before emitting your bracketed move."
    )


def record_strategy_neo4j(
    env_id: str,
    lessons: str,
    *,
    opponent: str = "",
    outcome: str = "",
    transcript_excerpt: str = "",
    namespace: str = DEFAULT_STRATEGY_NAMESPACE,
) -> None:
    stmt = """
    CREATE (s:TextArenaStrategy {
      namespace: $namespace,
      env_id: $env_id,
      opponent: $opponent,
      outcome: $outcome,
      lessons: $lessons,
      transcript_excerpt: $transcript_excerpt,
      created_at: timestamp()
    })
    RETURN elementId(s) AS id
    """
    payload = json.dumps(
        {
            "statements": [
                {
                    "statement": stmt,
                    "parameters": {
                        "namespace": namespace,
                        "env_id": env_id,
                        "opponent": opponent,
                        "outcome": outcome,
                        "lessons": lessons,
                        "transcript_excerpt": transcript_excerpt[:1500],
                    },
                }
            ]
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        NEO4J_HTTP_TX_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        resp.read()


def recall_strategies_neo4j(
    env_id: str,
    *,
    namespace: str = DEFAULT_STRATEGY_NAMESPACE,
    limit: int = 3,
) -> str:
    stmt = (
        "MATCH (s:TextArenaStrategy {namespace: $namespace, env_id: $env_id}) "
        "RETURN s.outcome, s.opponent, s.lessons ORDER BY s.created_at DESC LIMIT $limit"
    )
    try:
        payload = json.dumps(
            {
                "statements": [
                    {
                        "statement": stmt,
                        "parameters": {
                            "namespace": namespace,
                            "env_id": env_id,
                            "limit": limit,
                        },
                    }
                ]
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            NEO4J_HTTP_TX_URL,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        rows = (data.get("results") or [{}])[0].get("data") or []
        if not rows:
            return ""
        chunks = [
            f"[Prior {r['row'][0]} vs {r['row'][1] or 'opponent'}]\n{r['row'][2]}"
            for r in rows
            if r.get("row") and r["row"][2]
        ]
        return "\n".join(chunks)
    except Exception:
        return ""


def run_film_study(
    env_id: str,
    transcript: List[Dict[str, Any]],
    rewards: Any,
    *,
    agent: Any = None,
    opponent_name: str = "",
    namespace: str = DEFAULT_STRATEGY_NAMESPACE,
    record_neo4j: bool = True,
    reason: str = "",
) -> Dict[str, Any]:
    """Run post-match RL film study, extract lessons, and optionally persist to Neo4j."""
    compact_transcript = format_match_transcript(
        env_id=env_id,
        transcript=transcript,
        rewards=rewards,
        opponent_name=opponent_name,
        reason=reason,
    )
    outcome = classify_p0_outcome(rewards)
    prompt = build_film_study_prompt(compact_transcript, env_id)

    lessons = ""
    reflection_source = "deterministic_fallback"
    if agent is not None:
        try:
            if hasattr(agent, "reflect"):
                lessons = str(agent.reflect(prompt) or "").strip()
            else:
                lessons = str(agent(prompt) or "").strip()
            if lessons and not lessons.startswith("ERROR:"):
                reflection_source = "live_agent"
            else:
                lessons = ""
        except Exception:
            lessons = ""

    if not lessons:
        lessons = synthesize_deterministic_lessons(
            env_id=env_id,
            transcript=transcript,
            rewards=rewards,
            opponent_name=opponent_name,
        )

    neo4j_saved = False
    if record_neo4j:
        try:
            record_strategy_neo4j(
                env_id=env_id,
                lessons=lessons,
                opponent=opponent_name,
                outcome=outcome,
                transcript_excerpt=compact_transcript,
                namespace=namespace,
            )
            neo4j_saved = True
        except Exception:
            neo4j_saved = False

    return {
        "env_id": env_id,
        "namespace": namespace,
        "opponent": opponent_name,
        "outcome": outcome,
        "reflection_source": reflection_source,
        "transcript": compact_transcript,
        "lessons": lessons,
        "neo4j_saved": neo4j_saved,
    }


def build_sparring_record(
    *,
    env_id: str,
    model: str,
    use_actorii: bool,
    opponent_name: str,
    match_1: Dict[str, Any],
    film_study: Optional[Dict[str, Any]] = None,
    match_2: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    r1 = _extract_p0_reward(match_1.get("rewards"))
    r2 = _extract_p0_reward(match_2.get("rewards")) if match_2 else None
    delta: Dict[str, Any] = {
        "p0_reward_before": r1,
        "p0_reward_after": r2,
        "p0_reward_delta": (round(r2 - r1, 4) if (r1 is not None and r2 is not None) else None),
        "improved": (r2 > r1) if (r1 is not None and r2 is not None) else None,
        "outcome_before": classify_p0_outcome(match_1.get("rewards")),
        "outcome_after": classify_p0_outcome(match_2.get("rewards")) if match_2 else None,
    }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env_id": env_id,
        "model": model,
        "use_actorii": use_actorii,
        "opponent": opponent_name,
        "match_1": match_1,
        "film_study": film_study,
        "match_2": match_2,
        "before_vs_after": delta,
    }


def save_sparring_results(
    record: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    path = Path(output_path) if output_path is not None else DEFAULT_SPARRING_RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {"schema_version": 1, "updated_at": "", "runs": []}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
                data = existing
        except Exception:
            pass
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["runs"].append(record)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def record_match_neo4j(summary: Dict[str, Any]) -> None:
    stmt = """
    CREATE (m:TextArenaMatch {
      env_id: $env_id,
      mode: $mode,
      players: $players,
      model: $model,
      turns: $turns,
      rewards_json: $rewards_json,
      elapsed_s: $elapsed_s,
      created_at: timestamp()
    })
    RETURN elementId(m) AS id
    """
    payload = json.dumps(
        {
            "statements": [
                {
                    "statement": stmt,
                    "parameters": {
                        "env_id": summary["env_id"],
                        "mode": summary["mode"],
                        "players": summary["players"],
                        "model": summary["model"],
                        "turns": summary["turns"],
                        "rewards_json": json.dumps(summary["rewards"]),
                        "elapsed_s": summary["elapsed_s"],
                    },
                }
            ]
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        NEO4J_HTTP_TX_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        resp.read()


def run_local_match(
    *,
    env_id: str,
    agents: Dict[int, Any],
    players: int,
    max_turns: int,
    model_name: str,
    match_label: str = "Match 1",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if ta is None:
        raise RuntimeError(
            "textarena is not installed in this Python interpreter. "
            "Run via `.venv-textarena/bin/python tools/bilby_textarena.py ...`."
        )
    t0 = time.monotonic()
    turns = 0
    transcript: List[Dict[str, Any]] = []

    env = ta.make(env_id=env_id)
    try:
        env = ta.wrappers.LLMObservationWrapper(env=env)
    except ValueError:
        pass
    env.reset(num_players=players)
    done = False
    while not done and turns < max_turns:
        try:
            player_id, observation = env.get_observation()
        except ValueError:
            # Some TextArena 0.7.4 envs already return string observations instead of 3-tuples
            env = ta.make(env_id=env_id)
            env.reset(num_players=players)
            player_id, observation = env.get_observation()
            if not isinstance(observation, str):
                observation = "\n".join(str(x) for x in observation)
        action = agents[player_id](str(observation))
        turns += 1
        transcript.append({"turn": turns, "player_id": player_id, "action": action})
        print(f"[{match_label} | Turn {turns} | P{player_id}] -> {action[:160]}", flush=True)
        done, _ = env.step(action=action)
    rewards, game_info = env.close()
    elapsed = round(time.monotonic() - t0, 2)
    summary = {
        "env_id": env_id,
        "mode": "local",
        "players": players,
        "model": model_name,
        "turns": turns,
        "rewards": rewards,
        "game_info": game_info,
        "elapsed_s": elapsed,
        "transcript": transcript,
    }
    return summary, transcript


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Bilby on TextArena environments.")
    parser.add_argument("--env", default="GuessTheNumber-v0", help="TextArena env id (e.g. GuessTheNumber-v0, TicTacToe-v0)")
    parser.add_argument("--players", type=int, default=1, help="Number of players for local mode (default: 1)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="Bilby primary OpenAI API URL")
    parser.add_argument("--model", default="qwen3.8:latest", help="Bilby primary model name")
    parser.add_argument("--opponent-url", default="", help="Optional opponent OpenAI API URL for player >= 1")
    parser.add_argument("--opponent-model", default="", help="Optional opponent model name")
    parser.add_argument(
        "--opponent-bot",
        default="",
        choices=["", *sorted(SPARRING_BOT_REGISTRY.keys())],
        help="Deterministic algorithmic sparring bot for player >= 1 (minimax_ttt|tactical_c4|nash_neg)",
    )
    parser.add_argument("--film-study", action="store_true", help="Run post-match film-study reflection and save :TextArenaStrategy to Neo4j")
    parser.add_argument("--rematch", action="store_true", help="Automatically run Match 2 (Rematch) with film-study lessons injected")
    parser.add_argument(
        "--results-path",
        default=str(DEFAULT_SPARRING_RESULTS_PATH),
        help="Output path for structured Before-vs-After sparring results JSON",
    )
    parser.add_argument("--max-turns", type=int, default=30, help="Safety cap on total turns")
    parser.add_argument("--think", action="store_true", help="Enable extended <think> reasoning on each move")
    parser.add_argument("--use-actorii", action="store_true", help="Run Player 0 through full Bilby ActorII VM + AgentHarness")
    parser.add_argument("--record-neo4j", action="store_true", help="Persist match outcome to Neo4j :TextArenaMatch")
    parser.add_argument("--online", action="store_true", help="Connect to live textarena.ai online arena")
    parser.add_argument("--model-name", default="Bilby-ActorII-Qwen3.8-27B", help="Online arena display name")
    parser.add_argument("--model-token", default=os.environ.get("TEXTARENA_MODEL_TOKEN", ""), help="Online arena token")
    parser.add_argument("--email", default=os.environ.get("TEXTARENA_EMAIL", ""), help="Registration email for online arena")
    args = parser.parse_args()

    if ta is None:
        raise RuntimeError(
            "textarena is not installed in this Python interpreter. "
            "Run via `.venv-textarena/bin/python tools/bilby_textarena.py ...`."
        )

    if args.use_actorii:
        bilby_agent: Any = ActorIIGameAgent(
            env_id=args.env,
            base_url=args.base_url,
            model=args.model,
            enable_thinking=args.think,
        )
    else:
        bilby_agent = LocalOpenAIAgent(
            base_url=args.base_url,
            model=args.model,
            max_tokens=2048 if args.think else 384,
            enable_thinking=args.think,
        )

    if args.online:
        t0 = time.monotonic()
        turns = 0
        env = ta.make_online(
            env_id=args.env,
            model_name=args.model_name,
            model_token=args.model_token or None,
            model_description="Bilby (ActorII v4.1 + Qwen3.8 27B on valkyrie)",
            email=args.email or None,
            agent=bilby_agent,
        )
        try:
            env = ta.wrappers.LLMObservationWrapper(env=env)
        except ValueError:
            pass
        env.reset(num_players=1)
        done = False
        while not done and turns < args.max_turns:
            player_id, observation = env.get_observation()
            action = bilby_agent(observation)
            turns += 1
            print(f"[Turn {turns} | P{player_id} (online)] -> {action[:160]}", flush=True)
            done, _ = env.step(action=action)
        rewards, game_info = env.close()
        elapsed = round(time.monotonic() - t0, 2)
        summary = {
            "env_id": args.env,
            "mode": "online",
            "players": args.players,
            "model": args.model,
            "turns": turns,
            "rewards": rewards,
            "game_info": game_info,
            "elapsed_s": elapsed,
        }
        print("\n=== MATCH SUMMARY ===")
        print(json.dumps(summary, indent=2, default=str))
        if args.record_neo4j:
            try:
                record_match_neo4j(summary)
                print("Saved :TextArenaMatch node to Neo4j.")
            except Exception as exc:
                print(f"Warning: Neo4j save failed: {exc}")
        return 0

    # Local mode
    if args.opponent_bot:
        opp_agent: Any = create_sparring_bot(args.opponent_bot)
        opponent_name = args.opponent_bot
        if args.players < 2:
            args.players = 2
    elif args.opponent_url:
        opp_agent = LocalOpenAIAgent(base_url=args.opponent_url, model=args.opponent_model or args.model)
        opponent_name = args.opponent_model or args.opponent_url
    else:
        opp_agent = bilby_agent
        opponent_name = "self"

    agents: Dict[int, Any] = {0: bilby_agent}
    for pid in range(1, args.players):
        agents[pid] = opp_agent

    match_1_summary, match_1_transcript = run_local_match(
        env_id=args.env,
        agents=agents,
        players=args.players,
        max_turns=args.max_turns,
        model_name=args.model,
        match_label="Match 1",
    )

    print("\n=== MATCH 1 SUMMARY ===")
    print(json.dumps(match_1_summary, indent=2, default=str))

    if args.record_neo4j:
        try:
            record_match_neo4j(match_1_summary)
            print("Saved Match 1 :TextArenaMatch node to Neo4j.")
        except Exception as exc:
            print(f"Warning: Neo4j save failed: {exc}")

    film_study_report: Optional[Dict[str, Any]] = None
    match_2_summary: Optional[Dict[str, Any]] = None

    if args.film_study or args.rematch:
        g_info = match_1_summary.get("game_info") or {}
        p0_info = g_info.get(0) or g_info.get("0") or {}
        p0_reason = str(p0_info.get("reason") or "") if isinstance(p0_info, dict) else ""
        film_study_report = run_film_study(
            env_id=args.env,
            transcript=match_1_transcript,
            rewards=match_1_summary["rewards"],
            agent=bilby_agent,
            opponent_name=opponent_name,
            namespace=DEFAULT_STRATEGY_NAMESPACE,
            record_neo4j=True,
            reason=p0_reason,
        )
        print("\n=== FILM-STUDY LESSONS ===")
        print(film_study_report["lessons"])
        if film_study_report["neo4j_saved"]:
            print(f"Saved :TextArenaStrategy node to Neo4j (namespace={DEFAULT_STRATEGY_NAMESPACE}).")

    if args.rematch:
        lessons = film_study_report["lessons"] if film_study_report else ""
        if hasattr(bilby_agent, "reset_for_rematch"):
            bilby_agent.reset_for_rematch(lessons)
        if args.opponent_bot:
            # Fresh deterministic opponent instance for Match 2
            opp_agent_m2 = create_sparring_bot(args.opponent_bot)
            for pid in range(1, args.players):
                agents[pid] = opp_agent_m2

        match_2_summary, _ = run_local_match(
            env_id=args.env,
            agents=agents,
            players=args.players,
            max_turns=args.max_turns,
            model_name=args.model,
            match_label="Match 2 (Rematch)",
        )
        print("\n=== MATCH 2 (REMATCH) SUMMARY ===")
        print(json.dumps(match_2_summary, indent=2, default=str))
        if args.record_neo4j:
            try:
                record_match_neo4j(match_2_summary)
                print("Saved Match 2 :TextArenaMatch node to Neo4j.")
            except Exception as exc:
                print(f"Warning: Neo4j save failed: {exc}")

    if args.opponent_bot or args.film_study or args.rematch:
        sparring_record = build_sparring_record(
            env_id=args.env,
            model=args.model,
            use_actorii=args.use_actorii,
            opponent_name=opponent_name,
            match_1=match_1_summary,
            film_study=film_study_report,
            match_2=match_2_summary,
        )
        saved_path = save_sparring_results(sparring_record, Path(args.results_path))
        print(f"\nSaved Before-vs-After sparring results to {saved_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

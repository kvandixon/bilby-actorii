#!/usr/bin/env python3
"""
Bilby TextArena Runner (`bilby_textarena.py`)
=============================================
Run Bilby's local :8080 model (Qwen3.8 27B) on any of TextArena's 712
single-player, two-player, or multi-player text games—locally or on the
live textarena.ai online TrueSkill arena.

Usage (via ActorII/.venv-textarena):
  # Local 1-player match:
  .venv-textarena/bin/python tools/bilby_textarena.py --env GuessTheNumber-v0 --players 1

  # Local 2-player self-play (or vs CPU opponent on :11434):
  .venv-textarena/bin/python tools/bilby_textarena.py --env TicTacToe-v0 --players 2
  .venv-textarena/bin/python tools/bilby_textarena.py --env SimpleNegotiation-v0 --players 2 \\
      --opponent-url http://127.0.0.1:11434/v1 --opponent-model clef-flash-cpu:latest

  # Live textarena.ai online arena:
  .venv-textarena/bin/python tools/bilby_textarena.py --online --env TicTacToe-v0 \\
      --model-name "Bilby-Qwen3.8-27B" --email "you@example.com"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from typing import Any, Dict, Optional

import textarena as ta

NEO4J_HTTP_TX_URL = os.environ.get(
    "ACTORII_NEO4J_HTTP_URL",
    "http://127.0.0.1:7474/db/neo4j/tx/commit",
)


class LocalOpenAIAgent:
    """Lightweight OpenAI-compatible chat agent pointing at local :8080 / :8082 / :11434."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "qwen3.8:latest",
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt or (
            "You are Bilby, a sharp competitive game-playing AI agent. "
            "Read the game rules and observation carefully. Follow the exact required "
            "action format (such as bracketed moves like `[42]` or `[e2 e4]`). "
            "Keep your reply concise and ALWAYS include your valid formatted move."
        )

    def __call__(self, observation: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": observation},
            ],
        }
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
        # Strip any residual <think>...</think> tags if present
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Bilby on TextArena environments.")
    parser.add_argument("--env", default="GuessTheNumber-v0", help="TextArena env id (e.g. GuessTheNumber-v0, TicTacToe-v0)")
    parser.add_argument("--players", type=int, default=1, help="Number of players for local mode (default: 1)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="Bilby primary OpenAI API URL")
    parser.add_argument("--model", default="qwen3.8:latest", help="Bilby primary model name")
    parser.add_argument("--opponent-url", default="", help="Optional opponent OpenAI API URL for player >= 1")
    parser.add_argument("--opponent-model", default="", help="Optional opponent model name")
    parser.add_argument("--max-turns", type=int, default=30, help="Safety cap on total turns")
    parser.add_argument("--record-neo4j", action="store_true", help="Persist match outcome to Neo4j :TextArenaMatch")
    parser.add_argument("--online", action="store_true", help="Connect to live textarena.ai online arena")
    parser.add_argument("--model-name", default="Bilby-ActorII-Qwen3.8-27B", help="Online arena display name")
    parser.add_argument("--model-token", default=os.environ.get("TEXTARENA_MODEL_TOKEN", ""), help="Online arena token")
    parser.add_argument("--email", default=os.environ.get("TEXTARENA_EMAIL", ""), help="Registration email for online arena")
    args = parser.parse_args()

    bilby_agent = LocalOpenAIAgent(base_url=args.base_url, model=args.model)

    t0 = time.monotonic()
    turns = 0

    if args.online:
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
    else:
        opp_agent = (
            LocalOpenAIAgent(base_url=args.opponent_url, model=args.opponent_model or args.model)
            if args.opponent_url
            else bilby_agent
        )
        agents = {0: bilby_agent}
        for pid in range(1, args.players):
            agents[pid] = opp_agent

        env = ta.make(env_id=args.env)
        try:
            env = ta.wrappers.LLMObservationWrapper(env=env)
        except ValueError:
            pass
        env.reset(num_players=args.players)
        done = False
        while not done and turns < args.max_turns:
            player_id, observation = env.get_observation()
            action = agents[player_id](observation)
            turns += 1
            print(f"[Turn {turns} | P{player_id}] -> {action[:160]}", flush=True)
            done, _ = env.step(action=action)
        rewards, game_info = env.close()

    elapsed = round(time.monotonic() - t0, 2)
    summary = {
        "env_id": args.env,
        "mode": "online" if args.online else "local",
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


if __name__ == "__main__":
    raise SystemExit(main())

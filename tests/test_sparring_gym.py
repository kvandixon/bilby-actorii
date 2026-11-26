"""Unit tests for Bilby's local RL Film-Study & Algorithmic Sparring Gym.

Zero LLM / GPU calls — tests all 3 deterministic algorithmic bots,
transcript formatting, offline film-study reflection, lesson injection,
and Before-vs-After SPARRING_RESULTS.json output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from sparring_opponents import (  # noqa: E402
    ConnectFourTacticalBot,
    MinimaxTicTacToeBot,
    NashNegotiationBot,
    create_sparring_bot,
)
import bilby_textarena as bt  # noqa: E402


# ---------------------------------------------------------------------------
# 1. MinimaxTicTacToeBot
# ---------------------------------------------------------------------------


def _render_ttt_obs(board: list[str], player_id: int = 0) -> str:
    my_sym = "O" if player_id == 0 else "X"
    opp_sym = "X" if player_id == 0 else "O"
    cells = [board[i] if board[i] else str(i) for i in range(9)]
    return (
        f"[GAME] You are Player {player_id} in Tic Tac Toe.\n"
        f"As Player {player_id}, you will be '{my_sym}', while your opponent is '{opp_sym}'.\n\n"
        "[GAME] Current Board:\n\n"
        f" {cells[0]} | {cells[1]} | {cells[2]} \n"
        "---+---+---\n"
        f" {cells[3]} | {cells[4]} | {cells[5]} \n"
        "---+---+---\n"
        f" {cells[6]} | {cells[7]} | {cells[8]} \n"
    )


def test_minimax_ttt_takes_center_on_empty_board() -> None:
    bot = MinimaxTicTacToeBot()
    obs = _render_ttt_obs([""] * 9, player_id=0)
    assert bot(obs) == "[4]"


def test_minimax_ttt_takes_immediate_win() -> None:
    bot = MinimaxTicTacToeBot()
    # O has 0 and 1, empty 2 -> must win at [2]
    board = ["O", "O", "", "X", "X", "", "", "", ""]
    assert bot(_render_ttt_obs(board, player_id=0)) == "[2]"


def test_minimax_ttt_blocks_immediate_opponent_win() -> None:
    bot = MinimaxTicTacToeBot()
    # Player 1 (X): opponent O has 0 and 4 -> must block at [8]
    board = ["O", "X", "", "", "O", "", "", "", ""]
    assert bot(_render_ttt_obs(board, player_id=1)) == "[8]"


def test_minimax_ttt_is_unbeatable_in_self_play() -> None:
    bot0 = create_sparring_bot("minimax_ttt")
    bot1 = create_sparring_bot("minimax_ttt")
    board = [""] * 9
    turn = 0
    for _ in range(9):
        obs = _render_ttt_obs(board, player_id=turn)
        move_str = (bot0 if turn == 0 else bot1)(obs)
        cell = int(move_str.strip("[]"))
        assert board[cell] == ""
        board[cell] = "O" if turn == 0 else "X"
        turn = 1 - turn
    # Minimax vs Minimax must always end in a draw (no winner)
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    assert not any(board[a] and board[a] == board[b] == board[c] for a, b, c in lines)


# ---------------------------------------------------------------------------
# 2. ConnectFourTacticalBot
# ---------------------------------------------------------------------------


def _render_c4_obs(rows: list[list[str]], player_id: int = 0) -> str:
    sym = "X" if player_id == 0 else "O"
    body = "\n".join(" ".join(r) for r in rows)
    return (
        f"[GAME] You are Player {player_id} in Connect Four.\n"
        f"Your disc symbol: {sym}.\n"
        "[GAME] Board state:\n"
        "0 1 2 3 4 5 6\n"
        "-------------\n"
        f"{body}\n"
    )


def test_tactical_c4_prefers_center_on_empty_board() -> None:
    bot = create_sparring_bot("tactical_c4")
    board = [["."] * 7 for _ in range(6)]
    assert bot(_render_c4_obs(board, player_id=0)) == "[col 3]"


def test_tactical_c4_takes_immediate_horizontal_win() -> None:
    bot = ConnectFourTacticalBot()
    board = [["."] * 7 for _ in range(6)]
    board[5][0] = board[5][1] = board[5][2] = "X"
    board[5][4] = board[5][5] = "O"
    assert bot(_render_c4_obs(board, player_id=0)) == "[col 3]"


def test_tactical_c4_blocks_opponent_vertical_three_in_a_row() -> None:
    bot = ConnectFourTacticalBot()
    board = [["."] * 7 for _ in range(6)]
    # Opponent O has 3 stacked vertically in col 5
    board[5][5] = board[4][5] = board[3][5] = "O"
    board[5][3] = "X"
    assert bot(_render_c4_obs(board, player_id=0)) == "[col 5]"


def test_tactical_c4_blocks_opponent_horizontal_three_in_a_row() -> None:
    bot = ConnectFourTacticalBot()
    board = [["."] * 7 for _ in range(6)]
    # Player 1 (O): opponent X has 3 in a row at cols 1, 2, 3 (col 0 blocked by O)
    board[5][0] = "O"
    board[5][1] = board[5][2] = board[5][3] = "X"
    assert bot(_render_c4_obs(board, player_id=1)) == "[col 4]"


# ---------------------------------------------------------------------------
# 3. NashNegotiationBot
# ---------------------------------------------------------------------------


_NEG_BASE_OBS = (
    "You are Player 1 in the Negotiation Game.\n"
    "The resources and associated values you currently have are:\n"
    "\t+ [Wheat]   Qty: 19   Value: 5\n"
    "\t+ [Wood]    Qty: 14   Value: 8\n"
    "\t+ [Sheep]   Qty: 24   Value: 14\n"
    "\t+ [Brick]   Qty: 7    Value: 29\n"
    "\t+ [Ore]     Qty: 14   Value: 37\n"
)


def test_nash_neg_accepts_positive_value_offer() -> None:
    bot = create_sparring_bot("nash_neg")
    # Offered 1 Ore (value 37 to us) for 2 Wheat (cost 10 to us) -> net +27 -> Accept!
    obs = _NEG_BASE_OBS + "Player 0 made the following offer to Player 1: Offered items: 1 Ore -> Requested items: 2 Wheat\n"
    action = bot(obs)
    assert "[Accept]" in action


def test_nash_neg_counters_unfair_offer_instead_of_blind_deny() -> None:
    bot = NashNegotiationBot()
    # Offered 1 Wheat (val 5) for 1 Ore (val 37) -> unfair -> must counter-offer, NOT [Deny]
    obs = _NEG_BASE_OBS + "Player 0 made the following offer to Player 1: Offered items: 1 Wheat -> Requested items: 1 Ore\n"
    action = bot(obs)
    assert "[Deny]" not in action
    assert "[Offer:" in action
    assert "Wheat -> 1 Ore]" in action


def test_nash_neg_opening_offer_is_constructive() -> None:
    bot = NashNegotiationBot()
    action = bot(_NEG_BASE_OBS)
    assert "[Offer:" in action
    assert "[Deny]" not in action


def test_create_sparring_bot_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown sparring bot"):
        create_sparring_bot("nonexistent_bot")


# ---------------------------------------------------------------------------
# 4. Film-Study Transcript Formatting, Offline Reflection & JSON Output
# ---------------------------------------------------------------------------


def test_format_match_transcript_and_offline_film_study() -> None:
    transcript = [
        {"turn": 1, "player_id": 0, "action": "[0]"},
        {"turn": 2, "player_id": 1, "action": "[4]"},
        {"turn": 3, "player_id": 0, "action": "[1]"},
        {"turn": 4, "player_id": 1, "action": "[2]"},
    ]
    rewards = {0: -1, 1: 1}

    formatted = bt.format_match_transcript(
        env_id="TicTacToe-v0",
        transcript=transcript,
        rewards=rewards,
        opponent_name="minimax_ttt",
    )
    assert "TicTacToe-v0" in formatted
    assert "minimax_ttt" in formatted
    assert "P0 Outcome: LOSS" in formatted
    assert "T01 Bilby(P0): [0]" in formatted
    assert "T02 Opp(P1): [4]" in formatted

    report = bt.run_film_study(
        env_id="TicTacToe-v0",
        transcript=transcript,
        rewards=rewards,
        agent=None,
        opponent_name="minimax_ttt",
        record_neo4j=False,
    )
    assert report["outcome"] == "LOSS"
    assert report["namespace"] == "textarena_bilby"
    assert report["reflection_source"] == "deterministic_fallback"
    assert "[4]" in report["lessons"]


def test_sparring_results_json_roundtrip(tmp_path: Path) -> None:
    out_file = tmp_path / "SPARRING_RESULTS.json"
    match_1 = {
        "env_id": "ConnectFour-v0",
        "mode": "local",
        "players": 2,
        "model": "qwen3.8:latest",
        "turns": 14,
        "rewards": {0: -1, 1: 1},
        "elapsed_s": 12.3,
    }
    match_2 = {
        "env_id": "ConnectFour-v0",
        "mode": "local",
        "players": 2,
        "model": "qwen3.8:latest",
        "turns": 22,
        "rewards": {0: 1, 1: -1},
        "elapsed_s": 18.7,
    }
    film_study = {
        "env_id": "ConnectFour-v0",
        "namespace": "textarena_bilby",
        "opponent": "tactical_c4",
        "outcome": "LOSS",
        "lessons": "Claim [col 3] early and block vertical 3-in-a-row threats.",
        "neo4j_saved": False,
    }

    record = bt.build_sparring_record(
        env_id="ConnectFour-v0",
        model="qwen3.8:latest",
        use_actorii=True,
        opponent_name="tactical_c4",
        match_1=match_1,
        film_study=film_study,
        match_2=match_2,
    )
    saved_path = bt.save_sparring_results(record, output_path=out_file)
    assert saved_path == out_file

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert run["opponent"] == "tactical_c4"
    assert run["before_vs_after"]["p0_reward_before"] == -1.0
    assert run["before_vs_after"]["p0_reward_after"] == 1.0
    assert run["before_vs_after"]["p0_reward_delta"] == 2.0
    assert run["before_vs_after"]["improved"] is True
    assert run["before_vs_after"]["outcome_before"] == "LOSS"
    assert run["before_vs_after"]["outcome_after"] == "WIN"

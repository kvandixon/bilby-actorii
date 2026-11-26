#!/usr/bin/env python3
"""
Deterministic Algorithmic Sparring Bots for TextArena (`sparring_opponents.py`)
===============================================================================
Zero-VRAM, zero-LLM-call deterministic opponents for Bilby's local RL
Film-Study & Algorithmic Sparring Gym:

  - `MinimaxTicTacToeBot`     (`minimax_ttt`): Unbeatable minimax on 0..8 board
  - `ConnectFourTacticalBot`  (`tactical_c4`): Immediate win, opponent 3-in-a-row
                                               block, threat scoring, center bias
  - `NashNegotiationBot`      (`nash_neg`):    Private-value trade evaluation +
                                               constructive counter-offers (no blind [Deny])
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. MinimaxTicTacToeBot
# ---------------------------------------------------------------------------

_TTT_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

_TTT_PREFERENCE: Tuple[int, ...] = (4, 0, 2, 6, 8, 1, 3, 5, 7)


def _ttt_winner(board: Tuple[str, ...]) -> str:
    for a, b, c in _TTT_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return ""


@lru_cache(maxsize=None)
def _minimax_ttt(
    board: Tuple[str, ...],
    turn_sym: str,
    my_sym: str,
    opp_sym: str,
) -> Tuple[int, int]:
    """Return (score, best_cell) from `my_sym`'s perspective.

    Positive scores favor faster wins; negative scores favor slower losses.
    """
    winner = _ttt_winner(board)
    if winner == my_sym:
        return (10, -1)
    if winner == opp_sym:
        return (-10, -1)
    if "" not in board:
        return (0, -1)

    maximizing = turn_sym == my_sym
    next_sym = opp_sym if maximizing else my_sym
    best_score = -999 if maximizing else 999
    best_move = -1

    for idx in _TTT_PREFERENCE:
        if board[idx] != "":
            continue
        nxt = list(board)
        nxt[idx] = turn_sym
        child_score, _ = _minimax_ttt(tuple(nxt), next_sym, my_sym, opp_sym)
        # Prefer faster wins and slower losses
        if child_score > 0:
            adj_score = child_score - 1
        elif child_score < 0:
            adj_score = child_score + 1
        else:
            adj_score = 0

        if maximizing:
            if adj_score > best_score:
                best_score = adj_score
                best_move = idx
        else:
            if adj_score < best_score:
                best_score = adj_score
                best_move = idx

    return (best_score, best_move)


class MinimaxTicTacToeBot:
    """Unbeatable deterministic minimax bot for TextArena `TicTacToe-v0` (0..8 board)."""

    name = "minimax_ttt"

    @staticmethod
    def parse_symbols(observation: str) -> Tuple[str, str]:
        m = re.search(r"you will be ['\"]?([XO])['\"]?", observation, re.IGNORECASE)
        if m:
            my_sym = m.group(1).upper()
            return my_sym, ("O" if my_sym == "X" else "X")
        m2 = re.search(r"your (?:symbol|mark)[^XO]*([XO])", observation, re.IGNORECASE)
        if m2:
            my_sym = m2.group(1).upper()
            return my_sym, ("O" if my_sym == "X" else "X")
        if re.search(r"You are Player 1", observation, re.IGNORECASE):
            return "X", "O"
        return "O", "X"

    @staticmethod
    def parse_board(observation: str) -> Tuple[str, ...]:
        row_matches = re.findall(
            r"^\s*([XO0-8])\s*\|\s*([XO0-8])\s*\|\s*([XO0-8])\s*$",
            observation,
            flags=re.MULTILINE,
        )
        if len(row_matches) >= 3:
            last_three = row_matches[-3:]
            cells: List[str] = []
            for row in last_three:
                for token in row:
                    cells.append(token if token in ("X", "O") else "")
            return tuple(cells)
        return ("",) * 9

    def choose_move(self, board: Tuple[str, ...], my_sym: str, opp_sym: str) -> int:
        _, move = _minimax_ttt(board, my_sym, my_sym, opp_sym)
        if move == -1:
            for idx in _TTT_PREFERENCE:
                if board[idx] == "":
                    return idx
            return 0
        return move

    def __call__(self, observation: str) -> str:
        my_sym, opp_sym = self.parse_symbols(observation)
        board = self.parse_board(observation)
        move = self.choose_move(board, my_sym, opp_sym)
        return f"[{move}]"


# ---------------------------------------------------------------------------
# 2. ConnectFourTacticalBot
# ---------------------------------------------------------------------------

_C4_ROWS = 6
_C4_COLS = 7
_C4_COL_ORDER: Tuple[int, ...] = (3, 2, 4, 1, 5, 0, 6)


class ConnectFourTacticalBot:
    """Deterministic tactical bot for TextArena `ConnectFour-v0`.

    Priority order:
      1. Immediate winning drop
      2. Immediate block of opponent 3-in-a-row winning drop
      3. Tactical 4-window threat scoring (avoiding gift moves that let opponent win above)
      4. Center-column preference `[col 3]` -> 2, 4 -> 1, 5 -> 0, 6
    """

    name = "tactical_c4"

    @staticmethod
    def parse_symbols(observation: str) -> Tuple[str, str]:
        m = re.search(r"Your disc symbol:\s*([XO])", observation, re.IGNORECASE)
        if m:
            my_sym = m.group(1).upper()
            return my_sym, ("O" if my_sym == "X" else "X")
        if re.search(r"You are Player 1", observation, re.IGNORECASE):
            return "O", "X"
        return "X", "O"

    @staticmethod
    def parse_board(observation: str) -> List[List[str]]:
        row_pattern = re.compile(r"^\s*([.XO](?:\s+[.XO]){6})\s*$", re.MULTILINE)
        matches = row_pattern.findall(observation)
        if len(matches) >= _C4_ROWS:
            last_six = matches[-_C4_ROWS:]
            return [line.split() for line in last_six]
        return [["." for _ in range(_C4_COLS)] for _ in range(_C4_ROWS)]

    @staticmethod
    def drop_row(board: List[List[str]], col: int) -> int:
        if col < 0 or col >= _C4_COLS:
            return -1
        for r in range(_C4_ROWS - 1, -1, -1):
            if board[r][col] == ".":
                return r
        return -1

    @staticmethod
    def has_four(board: List[List[str]], sym: str) -> bool:
        # Horizontal
        for r in range(_C4_ROWS):
            for c in range(_C4_COLS - 3):
                if all(board[r][c + i] == sym for i in range(4)):
                    return True
        # Vertical
        for r in range(_C4_ROWS - 3):
            for c in range(_C4_COLS):
                if all(board[r + i][c] == sym for i in range(4)):
                    return True
        # Diagonal down-right
        for r in range(_C4_ROWS - 3):
            for c in range(_C4_COLS - 3):
                if all(board[r + i][c + i] == sym for i in range(4)):
                    return True
        # Diagonal up-right
        for r in range(3, _C4_ROWS):
            for c in range(_C4_COLS - 3):
                if all(board[r - i][c + i] == sym for i in range(4)):
                    return True
        return False

    @classmethod
    def _score_board(cls, board: List[List[str]], my_sym: str, opp_sym: str) -> int:
        score = 0
        # Center column control bonus
        center_count = sum(1 for r in range(_C4_ROWS) if board[r][3] == my_sym)
        score += center_count * 4

        def eval_window(window: List[str]) -> int:
            my_n = window.count(my_sym)
            opp_n = window.count(opp_sym)
            empty_n = window.count(".")
            if my_n == 4:
                return 1000
            if opp_n == 4:
                return -1000
            if my_n == 3 and empty_n == 1:
                return 12
            if my_n == 2 and empty_n == 2:
                return 3
            if opp_n == 3 and empty_n == 1:
                return -15
            if opp_n == 2 and empty_n == 2:
                return -3
            return 0

        for r in range(_C4_ROWS):
            for c in range(_C4_COLS - 3):
                score += eval_window([board[r][c + i] for i in range(4)])
        for r in range(_C4_ROWS - 3):
            for c in range(_C4_COLS):
                score += eval_window([board[r + i][c] for i in range(4)])
        for r in range(_C4_ROWS - 3):
            for c in range(_C4_COLS - 3):
                score += eval_window([board[r + i][c + i] for i in range(4)])
        for r in range(3, _C4_ROWS):
            for c in range(_C4_COLS - 3):
                score += eval_window([board[r - i][c + i] for i in range(4)])
        return score

    def choose_column(self, board: List[List[str]], my_sym: str, opp_sym: str) -> int:
        valid_cols = [c for c in _C4_COL_ORDER if self.drop_row(board, c) != -1]
        if not valid_cols:
            return 3

        # 1. Immediate win
        for c in valid_cols:
            r = self.drop_row(board, c)
            board[r][c] = my_sym
            won = self.has_four(board, my_sym)
            board[r][c] = "."
            if won:
                return c

        # 2. Immediate block of opponent 3-in-a-row
        for c in valid_cols:
            r = self.drop_row(board, c)
            board[r][c] = opp_sym
            opp_wins = self.has_four(board, opp_sym)
            board[r][c] = "."
            if opp_wins:
                return c

        # 3. Tactical scoring + center preference
        best_col = valid_cols[0]
        best_score = -10**9
        for c in valid_cols:
            r = self.drop_row(board, c)
            board[r][c] = my_sym
            score = self._score_board(board, my_sym, opp_sym) + (4 - abs(3 - c))
            # Avoid handing opponent an immediate win directly above our drop
            if r > 0:
                board[r - 1][c] = opp_sym
                if self.has_four(board, opp_sym):
                    score -= 500
                board[r - 1][c] = "."
            board[r][c] = "."
            if score > best_score:
                best_score = score
                best_col = c
        return best_col

    def __call__(self, observation: str) -> str:
        my_sym, opp_sym = self.parse_symbols(observation)
        board = self.parse_board(observation)
        col = self.choose_column(board, my_sym, opp_sym)
        return f"[col {col}]"


# ---------------------------------------------------------------------------
# 3. NashNegotiationBot
# ---------------------------------------------------------------------------


class NashNegotiationBot:
    """Deterministic fair-value negotiation bot for TextArena `SimpleNegotiation-v0`.

    Evaluates incoming offers against private resource valuations:
      - Accepts if feasible and net value >= 0 (received_value >= given_value > 0)
      - Otherwise makes a constructive, positive-EV counter-offer trading our
        lowest-value surplus resource for our highest-value target resource
        instead of emitting a blind `[Deny]`.
    """

    name = "nash_neg"

    @staticmethod
    def parse_inventory(observation: str) -> Dict[str, Tuple[int, int]]:
        """Return `{resource_name: (qty, unit_value)}` using latest values in observation."""
        inventory: Dict[str, Tuple[int, int]] = {}
        pattern = re.compile(
            r"\+\s*\[([A-Za-z]+)\]\s+Qty:\s*(\d+)\s+Value:\s*(\d+)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(observation):
            res = m.group(1).strip().title()
            qty = int(m.group(2))
            val = int(m.group(3))
            inventory[res] = (qty, val)
        return inventory

    @staticmethod
    def _parse_item_list(text: str) -> Dict[str, int]:
        items: Dict[str, int] = {}
        for m in re.finditer(r"(\d+)\s+([A-Za-z]+)", text):
            qty = int(m.group(1))
            res = m.group(2).strip().title()
            items[res] = items.get(res, 0) + qty
        return items

    @classmethod
    def parse_incoming_offer(
        cls,
        observation: str,
    ) -> Optional[Tuple[Dict[str, int], Dict[str, int]]]:
        """Return `(offered_to_us, requested_from_us)` for the latest pending offer, or None."""
        # Prefer TextArena's structured offer notification line
        ta_matches = list(
            re.finditer(
                r"Player\s+\d+\s+made the following offer to Player\s+\d+:\s*"
                r"Offered items:\s*(.+?)\s*->\s*Requested items:\s*([^\n]+)",
                observation,
                re.IGNORECASE,
            )
        )
        last_match = ta_matches[-1] if ta_matches else None

        if last_match is None:
            # Fallback to raw bracketed [Offer: X -> Y] syntax (ignoring rule example lines)
            raw_matches = [
                m
                for m in re.finditer(r"\[Offer:\s*([^\]]+?)\s*->\s*([^\]]+?)\]", observation)
                if "3 Sheep, 2 Ore" not in m.group(0)
            ]
            if raw_matches:
                last_match = raw_matches[-1]

        if last_match is None:
            return None

        # If a trade resolution happened after this offer, it is no longer pending
        tail = observation[last_match.end() :]
        if re.search(r"\b(accepted|denied|rejected|declined)\b", tail, re.IGNORECASE):
            return None

        offered = cls._parse_item_list(last_match.group(1))
        requested = cls._parse_item_list(last_match.group(2))
        if not offered and not requested:
            return None
        return offered, requested

    @staticmethod
    def evaluate_offer(
        inventory: Dict[str, Tuple[int, int]],
        offered: Dict[str, int],
        requested: Dict[str, int],
    ) -> Tuple[bool, int, int]:
        """Return `(should_accept, received_value, given_value)`."""
        for res, qty in requested.items():
            avail = inventory.get(res, (0, 0))[0]
            if qty > avail:
                return False, 0, 0

        received_val = sum(qty * inventory.get(res, (0, 0))[1] for res, qty in offered.items())
        given_val = sum(qty * inventory.get(res, (0, 0))[1] for res, qty in requested.items())
        should_accept = received_val > 0 and received_val >= given_val
        return should_accept, received_val, given_val

    @staticmethod
    def build_counter_offer(inventory: Dict[str, Tuple[int, int]]) -> str:
        """Construct a fair, positive-EV counter-offer using available inventory."""
        if not inventory:
            return "Let's start with a balanced exchange. [Offer: 1 Wheat -> 1 Ore]"

        available = [(res, qty, val) for res, (qty, val) in inventory.items() if qty > 0]
        if not available:
            return "Let's explore a small trade. [Offer: 1 Wheat -> 1 Ore]"

        # Give our lowest-valued available resource; request our highest-valued resource
        available_sorted = sorted(available, key=lambda x: (x[2], -x[1], x[0]))
        all_by_val_desc = sorted(
            [(res, qty, val) for res, (qty, val) in inventory.items()],
            key=lambda x: (-x[2], x[0]),
        )

        give_res, give_avail, give_val = available_sorted[0]
        want_res, _, want_val = all_by_val_desc[0]
        if want_res == give_res and len(all_by_val_desc) > 1:
            want_res, _, want_val = all_by_val_desc[1]

        # Offer enough units to be attractive while keeping net gain strictly positive when possible
        if give_val > 0 and want_val > give_val:
            max_profitable = max(1, (want_val - 1) // give_val)
            give_qty = min(give_avail, max(1, min(max_profitable, 3)))
        else:
            give_qty = 1

        return (
            f"That offer doesn't quite balance for me, so here is a constructive counter-offer: "
            f"[Offer: {give_qty} {give_res} -> 1 {want_res}]"
        )

    def __call__(self, observation: str) -> str:
        inventory = self.parse_inventory(observation)
        incoming = self.parse_incoming_offer(observation)
        if incoming is not None:
            offered, requested = incoming
            should_accept, _, _ = self.evaluate_offer(inventory, offered, requested)
            if should_accept:
                return "That trade increases our mutual value. [Accept]"
        return self.build_counter_offer(inventory)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SPARRING_BOT_REGISTRY = {
    "minimax_ttt": MinimaxTicTacToeBot,
    "tactical_c4": ConnectFourTacticalBot,
    "nash_neg": NashNegotiationBot,
}


def create_sparring_bot(bot_name: str):
    key = bot_name.strip().lower()
    if key not in SPARRING_BOT_REGISTRY:
        valid = ", ".join(sorted(SPARRING_BOT_REGISTRY.keys()))
        raise ValueError(f"Unknown sparring bot '{bot_name}'. Valid choices: {valid}")
    return SPARRING_BOT_REGISTRY[key]()

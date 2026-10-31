"""Writes Task 20's late positions as plain text for ``experiments/wasm_timing.mjs``.

Usage: ``python experiments/wasm_positions.py [--folder runs/p3_matches/b_endgame]``

Input: ``positions.json`` there (a game seed and the actions so far). Output:
``positions.txt``, one position per line: cards in all hands, hidden cards, the four
starting hands, then the actions, all decimal. WebAssembly draws deals from a different
random stream, so it cannot rebuild a position from its seed: it gets the hands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from big2._core import Big2State


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    ap.add_argument("--folder", default="runs/p3_matches/b_endgame")
    args = ap.parse_args()
    folder = Path(args.folder)
    positions = json.loads((folder / "positions.json").read_text())["positions"]
    lines = []
    for p in positions:
        s = Big2State.new_deal(p["game_seed"])
        hands = list(s.hands)
        for a in p["actions"]:
            s.apply(a)
        total = sum(s.hand_sizes)
        hidden = total - s.hand_sizes[s.current_player]
        lines.append(" ".join(str(n) for n in [total, hidden, *hands, *p["actions"]]))
    (folder / "positions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} positions into {folder / 'positions.txt'}")


if __name__ == "__main__":
    main()

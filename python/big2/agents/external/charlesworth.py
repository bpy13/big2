"""Charlesworth's released Big Two network as a local opponent (Task 21, B2).

Henry Charlesworth, *Application of self-play reinforcement learning to a four-player game
of imperfect information*, arXiv:1808.10442 (survey: ``docs/research/survey.md`` §2.1).
This module is our own NumPy code written from the paper's appendices (A: the 412 input
features, B: the 1,695-action index). It runs his published weights, which have no licence
file: they stay under ``runs/external/charlesworth/`` (gitignored), are read only through
:mod:`big2.agents.external.safe_pickle`, and are never committed or shipped
(``bot-stream/docs/decisions.md`` B14).

His game differs from ours in one rule, which the engine carries as the hidden
``four_card_plays`` flag: two pairs and four of a kind are four-card plays. Play this agent
with ``Rules(four_card_plays=True)``; the network was trained for it, and we play ours
without four-card plays elsewhere.

**Network** (weights file order): ``412 -> 512`` shared layer, then a policy branch
``512 -> 256 -> 1695`` and a value branch ``512 -> 256 -> 1`` (unused here). Hidden
activations are ReLU.

**Actions** are indexed by *positions in the sorted hand* (0 = lowest card, at most 13):
singles 0-12, pairs 13-45, triples 46-76, four-card hands 77-406, five-card hands 407-1693,
pass 1694. ``actionIndices.pkl`` holds his look-up tables; this module only uses the
reverse tables (action number within its size -> hand positions).

**Input** (412 binary features, in the order of the paper's Figure 3):

* 13 hand cards x 22: value (13: 3..2), suit (4: D C H S), flags in pair / in three of a
  kind / in four of a kind / in straight / in flush. Empty slots are zero.
* 3 opponents (the next player first) x 27: cards left (13, one-hot of 1..13), has played
  AD AC AH AS 2D 2C 2H 2S (8), has played pair / three of a kind / four-card hand (two pair
  *or* four of a kind) / straight / flush / full house (6).
* 16: Q K A 2 of each suit (D C H S) has been played by anyone.
* 29 for the previous hand: highest card's value (13) and suit (4; none for a full house,
  whose value is the three of a kind's), then control, single, pair, three of a kind, two
  pair, four of a kind, straight, flush, full house (9), then no / one / two passes since
  the last play. Control is the only bit set when the player leads a new trick.

The order inside the groups was first guessed from the paper's figure; it was then checked
against the author's published source (``henrycharlesworth/big2_PPOalgorithm`` at commit
eea4b04, read-only, nothing copied). The corrections: the 16 "played" bits come before the
previous-hand group, control is the group's first type bit, four of a kind sets the
opponents' four-card bit, a full house takes the three's value and no suit, and his
straights are any five consecutive ranks including J Q K A 2 (no low-2 straights).
``tests/test_charlesworth.py`` pins the layout on one hand-built state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from big2._core import Big2State, classify_cards
from big2.agents.external import safe_pickle

ROOT = Path(__file__).resolve().parents[4]
WEIGHTS_DIR = ROOT / "runs" / "external" / "charlesworth"
N_FEATURES = 412
N_ACTIONS = 1695
PASS_ACTION = 1694
#: First action number of each hand size (singles, pairs, triples, four-card, five-card).
SIZE_OFFSET = {1: 0, 2: 13, 3: 46, 4: 77, 5: 407}

# His straights: any five consecutive ranks (Big Two ranks 3=0 ... A=11, 2=12), so J Q K A 2
# counts and A 2 3 4 5 does not. This is what the network saw in training, whatever the
# engine's own straights rule says.
_STRAIGHTS = [tuple(range(k, k + 5)) for k in range(9)]

_TYPES = [
    "single",
    "pair",
    "triple",
    "two_pair",
    "four_of_a_kind",
    "straight",
    "flush",
    "full_house",
]
# The opponents' "has played" type bits: the four-card bit covers two pair and four of a kind.
_OPP_TYPES = ["pair", "triple", "four_card", "straight", "flush", "full_house"]


def _bits(mask: int) -> list[int]:
    return [i for i in range(52) if (mask >> i) & 1]


def card_type_flags(cards: int, straights: str) -> set[str]:
    """The feature types of a play: its kind, with a straight flush as straight + flush."""
    idx = _bits(cards)
    ranks = [i // 4 for i in idx]
    n = len(idx)
    if n == 1:
        return {"single"}
    if n == 2:
        return {"pair"}
    if n == 3:
        return {"triple"}
    if n == 4:
        return {"four_of_a_kind"} if len(set(ranks)) == 1 else {"two_pair"}
    c = classify_cards(cards, straights=straights)
    if c is None:
        raise ValueError(f"not a combination: {cards:#x}")
    kind = c[0]
    if kind == "straight_flush":
        return {"straight", "flush"}
    return {kind}


def _opp_kinds(flags: set[str]) -> set[str]:
    return {"four_card" if f in ("two_pair", "four_of_a_kind") else f for f in flags}


def load_weights(directory: Path = WEIGHTS_DIR):
    """The 10 weight arrays and the action tables, read with the restricted unpickler."""
    params = safe_pickle.load(str(directory / "modelParameters136500"))
    tables = safe_pickle.load(str(directory / "actionIndices.pkl"))
    return params, tables


@dataclass
class CharlesworthAgent:
    """Plays the network's highest-probability legal action (``temperature`` 0) or samples
    from its masked policy (``temperature`` 1).

    Fair play: only the acting player's hand and public history are read.
    """

    temperature: float = 0.0
    seed: int | None = None
    name: str = "charlesworth"
    directory: Path = WEIGHTS_DIR
    #: Hand positions never mapped to an action (should stay 0; the tests count it).
    unmapped: int = field(default=0, init=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        params, tables = load_weights(Path(self.directory))
        self.w1, self.b1, self.wp, self.bp, self.wo, self.bo = params[:6]
        self._to_action: dict[tuple[int, ...], int] = {}
        for size, table in zip((2, 3, 4, 5), tables[4:8], strict=True):
            for k, row in enumerate(table):
                self._to_action[tuple(int(p) for p in row)] = SIZE_OFFSET[size] + k

    # --- features -----------------------------------------------------------------

    def features(self, state: Big2State) -> np.ndarray:
        me = state.current_player
        assert me is not None
        rules = state.rules()
        straights = str(rules["straights"])
        hand = _bits(state.hand(me))
        x = np.zeros(N_FEATURES, dtype=np.float32)

        ranks = [c // 4 for c in hand]
        suits = [c % 4 for c in hand]
        in_straight = set()
        have = set(ranks)
        for seq in _STRAIGHTS:
            if have.issuperset(seq):
                in_straight.update(seq)
        for pos, (r, s) in enumerate(zip(ranks, suits, strict=True)):
            o = pos * 22
            x[o + r] = 1
            x[o + 13 + s] = 1
            n_rank = ranks.count(r)
            x[o + 17] = n_rank >= 2
            x[o + 18] = n_rank >= 3
            x[o + 19] = n_rank == 4
            x[o + 20] = r in in_straight
            x[o + 21] = suits.count(s) >= 5

        history = state.history
        sizes = state.hand_sizes
        for k in range(3):
            opp = (me + 1 + k) % 4
            o = 286 + k * 27
            x[o + min(max(sizes[opp], 1), 13) - 1] = 1
            played = [c for p, c in history if p == opp and c]
            top = 0
            kinds: set[str] = set()
            for c in played:
                top |= c
                kinds |= _opp_kinds(card_type_flags(c, straights))
            for j, card in enumerate(range(44, 52)):  # AD AC AH AS 2D 2C 2H 2S
                x[o + 13 + j] = (top >> card) & 1
            for j, name in enumerate(_OPP_TYPES):
                x[o + 21 + j] = name in kinds

        # The 16 "played" bits (Q K A 2, each D C H S) come before the previous hand.
        played_all = state.played
        for j, card in enumerate(range(36, 52)):
            x[367 + j] = (played_all >> card) & 1

        o = 383
        cur = state.current_play
        passes = 0
        for _, c in reversed(history):
            if c:
                break
            passes += 1
        if cur is None:
            x[o + 17] = 1  # control: leading a new trick, nothing else in the group is set
        else:
            _, cards, _ = cur
            flags = card_type_flags(cards, straights)
            top_card = max(_bits(cards))
            value = top_card // 4
            suit_bit = 13 + top_card % 4
            if "full_house" in flags:  # the value is the three of a kind's; no suit
                rks = [c // 4 for c in _bits(cards)]
                value = next(r for r in rks if rks.count(r) == 3)
                suit_bit = -1
            x[o + value] = 1
            if suit_bit >= 0:
                x[o + suit_bit] = 1
            for name in flags:
                x[o + 18 + _TYPES.index(name)] = 1
            x[o + 26 + min(passes, 2)] = 1
        return x

    # --- policy -------------------------------------------------------------------

    def logits(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        p = np.maximum(h @ self.wp + self.bp, 0.0)
        return p @ self.wo + self.bo

    def action_numbers(self, state: Big2State, legal: list[int]) -> list[int]:
        """The network's action number of each legal action (-1: not representable)."""
        hand = _bits(state.hand(state.current_player or 0))
        position = {c: i for i, c in enumerate(hand)}
        out = []
        for a in legal:
            if a == 0:
                out.append(PASS_ACTION)
                continue
            pos = tuple(position[c] for c in _bits(a))
            if len(pos) == 1:
                out.append(pos[0])
            else:
                out.append(self._to_action.get(pos, -1))
        return out

    def policy(self, state: Big2State) -> tuple[list[int], np.ndarray]:
        """``(legal actions, probability of each)``; the softmax runs over legal actions."""
        legal = list(state.legal_actions())
        nums = self.action_numbers(state, legal)
        self.unmapped += sum(n < 0 for n in nums)
        z = self.logits(self.features(state))
        zl = np.array([z[n] if n >= 0 else -np.inf for n in nums], dtype=np.float64)
        e = np.exp(zl - zl.max())
        return legal, e / e.sum()

    def act(self, state: Big2State) -> int:
        legal, p = self.policy(state)
        if len(legal) == 1:
            return legal[0]
        if self.temperature <= 0:
            return legal[int(np.argmax(p))]
        return self._rng.choices(legal, weights=p.tolist())[0]

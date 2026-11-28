"""Branch exploration of one deal-seat's decisions (plan Task 50, hk-games D161 choice 2),
game-agnostic.

A deal-seat is a tree of the seat's **own** decisions. From a decision, each own move ``m``
has a probability ``p(m)`` under a stand-in policy (a player of the deal's bracket); the
other seats are deterministic seeded bots, so the move alone fixes the next own decision (or
the deal's end). A node's ``path_p`` is the product of the own-move probabilities from the
first decision to it.

:func:`explore` grows the tree: first the main line (the source deal, already graded, so no
grade calls but it counts toward the cap), then best first: the unexplored edge with the
highest path probability is played on to the next own decision, which is graded. It stops
when :func:`coverage` reaches ``target`` or ``cap`` decisions are graded. Forced decisions
(one legal move) cost nothing and do not count toward the cap. Because the order does not
depend on the cap, a run at cap 80 also gives the tree at any smaller cap (its prefix).

Coverage is the probability that a stand-in player's whole path, from the first decision to
the deal's end, meets only graded decisions. Nothing here imports the rest of ``big2``.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: A child that is the deal's end (no further own decision).
END = "end"


@dataclass
class Node:
    """One own decision in the tree."""

    key: str
    path_p: float
    probs: dict[Hashable, float]
    forced: bool
    grade: Any = None
    main_line: bool = False
    #: own move -> child key, or :data:`END`; moves not yet played are absent.
    children: dict[Hashable, str] = field(default_factory=dict)


@dataclass
class Game:
    """The callbacks :func:`explore` needs. A *handle* is whatever the game uses to restore
    a decision (e.g. the action list from the deal's start)."""

    #: handle -> (key, {own move: probability}, forced)
    describe: Callable[[Any], tuple[str, dict[Hashable, float], bool]]
    #: (handle, own move) -> handle of the next own decision, or ``None`` at the deal's end
    step: Callable[[Any, Hashable], Any]
    #: handle -> the decision's grade (the expensive call)
    grade: Callable[[Any], Any]


def coverage(nodes: Mapping[str, Node], root: str) -> float:
    """Probability that the stand-in's path from ``root`` meets only graded decisions."""
    memo: dict[str, float] = {}

    def cov(k: str) -> float:
        if k in memo:
            return memo[k]
        n = nodes[k]
        if not n.forced and n.grade is None:
            memo[k] = 0.0
            return 0.0
        total = 0.0
        for m, p in n.probs.items():
            c = n.children.get(m)
            if c == END:
                total += p
            elif c is not None:
                total += p * cov(c)
        memo[k] = min(1.0, total)
        return memo[k]

    return cov(root)


@dataclass
class Result:
    """The explored tree of one deal-seat."""

    root: str
    nodes: dict[str, Node]
    graded: int
    coverage: float
    #: (graded decisions, coverage) after the main line and at each new grade: the
    #: coverage a run with that cap ends with (free end-of-deal edges after it not counted)
    curve: list[tuple[int, float]]
    new_grades: int


def explore(
    game: Game,
    root: Any,
    main_line: Sequence[Hashable],
    pregraded: Mapping[str, Any],
    cap: int = 40,
    target: float = 0.9,
) -> Result:
    """Grow the tree of one deal-seat (module docstring). ``main_line`` is the seat's own
    moves in the source deal, in order; ``pregraded`` maps keys to grades already known (the
    main line's, or an earlier run's), used instead of calling ``game.grade``."""
    nodes: dict[str, Node] = {}
    handles: dict[str, Any] = {}
    heap: list[tuple[float, int, str, Hashable]] = []
    seq = 0
    graded = 0
    new_grades = 0

    def add(handle: Any, path_p: float, main: bool) -> str:
        nonlocal graded, new_grades, seq
        key, probs, forced = game.describe(handle)
        n = Node(key, path_p, dict(probs), forced, main_line=main)
        nodes[key] = n
        handles[key] = handle
        if not forced:
            if key in pregraded:
                n.grade = pregraded[key]
            else:
                n.grade = game.grade(handle)
                new_grades += 1
            graded += 1
        return key

    def push_edges(key: str, skip: Hashable | None = None) -> None:
        nonlocal seq
        n = nodes[key]
        for m, p in n.probs.items():
            if p > 0 and m not in n.children and m != skip:
                heapq.heappush(heap, (-n.path_p * p, seq, key, m))
                seq += 1

    def play(key: str, m: Hashable, main: bool) -> str | None:
        """Plays own move ``m`` at ``key``; returns the child's key (None at the end).
        Forced decisions on the way are added and passed through at no cost."""
        n = nodes[key]
        nxt = game.step(handles[key], m)
        if nxt is None:
            n.children[m] = END
            return None
        c = add(nxt, n.path_p * n.probs.get(m, 0.0), main)
        n.children[m] = c
        return c

    # the main line: every own decision of the source deal
    root_key = add(root, 1.0, True)
    cur: str | None = root_key
    path = [root_key]
    for m in main_line:
        if cur is None:
            raise ValueError("main line is longer than the deal")
        cur = play(cur, m, True)
        if cur is not None:
            path.append(cur)
    if cur is not None:
        raise ValueError("main line ends before the deal does")
    for k in path:
        push_edges(k)
    cov_now = coverage(nodes, root_key)
    curve = [(graded, cov_now)]

    def pass_forced(k: str) -> None:
        """Forced decisions have one move: play it at once (no cost)."""
        while k is not None and nodes[k].forced:
            (only,) = nodes[k].probs
            nk = play(k, only, False)
            if nk is None:
                return
            k = nk
        if k is not None:
            push_edges(k)

    while heap and graded < cap and cov_now < target:
        _, _, key, m = heapq.heappop(heap)
        if m in nodes[key].children:
            continue
        before = graded
        c = play(key, m, False)
        if c is not None:
            pass_forced(c)
        cov_now = coverage(nodes, root_key)
        if graded > before:
            # a run with cap == graded stops here, so this is its coverage
            curve.append((graded, cov_now))
    return Result(root_key, nodes, graded, cov_now, curve, new_grades)


def coverage_at(curve: Sequence[tuple[int, float]], cap: int) -> float:
    """Coverage of the tree a run with this ``cap`` would have grown (the curve's prefix;
    the main line is always in, even when it alone has more graded decisions than ``cap``)."""
    best = curve[0][1]
    for g, c in curve:
        if g <= cap:
            best = c
    return best


__all__ = ["END", "Game", "Node", "Result", "coverage", "coverage_at", "explore"]

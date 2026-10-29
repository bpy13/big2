"""Agents that play Big Two.

An agent is anything with a ``name`` and an ``act(state) -> int`` method returning
a legal action (card mask, ``0`` = pass) for ``state.current_player``.

**Fair play:** ``state`` is the engine's full state, hidden hands included, for
speed. Agents must only use what the player to act can see: their own hand
(``state.hand(state.current_player)``) and public information (``history``,
``hand_sizes``, ``current_play``, ``played``). The built-in agents do; the
search agents only touch hidden cards through ``determinize``, which re-deals them.
A test (``test_agents.py``) checks that search agents ignore hidden cards.

Agents can be built from short spec strings, which is how the CLI and the
tournament runner create them::

    make_agent("greedy")
    make_agent("greedy:structured=1,epsilon=0.1")
    make_agent("ismcts:iterations=2000,exploration=0.25")
    make_agent("ismcts:iterations=1000,rollout=structured")
    make_agent("pimc:iterations=200")
    make_agent("puct:simulations=2000,net=models/big2-pvnet-r0.pknn")
    make_agent("qnet:temperature=1.0")  # models/big2-qnet-duty.pknn unless path=…
    make_agent("dmc:checkpoint=runs/dmc/v1/best.pt")
    make_agent("dmcsearch:checkpoint=runs/dmc/v1/best.pt,determinizations=32,top=6")

The ``dmc`` and ``dmcsearch`` kinds need PyTorch and a checkpoint from
``big2 rl train`` (see docs/rl.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from big2._core import Big2State, analyze
from big2.agents.patwa import PatwaGreedy, PatwaSmart

if TYPE_CHECKING:
    from big2._core import AnalysisResult


class Agent(Protocol):
    name: str

    def act(self, state: Big2State) -> int: ...


@dataclass
class RandomAgent:
    """Uniformly random legal action."""

    seed: int | None = None
    name: str = "random"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def act(self, state: Big2State) -> int:
        action: int = self._rng.choice(state.legal_actions())
        return action


@dataclass
class GreedyAgent:
    """Rule-based baseline: shed the weakest cards first (``crates/big2/src/bots/``).

    With ``structured=True`` it is ``StructuredGreedy``: it keeps the pairs, triples
    and five-card groups of the hand's best plan (min-plays planner), checking at
    most ``plan_depth`` plays per decision.
    """

    epsilon: float = 0.0
    seed: int | None = None
    name: str = "greedy"
    structured: bool = False
    plan_depth: int = 8
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def act(self, state: Big2State) -> int:
        return state.greedy_action(
            self.epsilon, self._rng.getrandbits(63), self.structured, self.plan_depth
        )


@dataclass
class SearchAgent:
    """Plays the solver's top move. ``algorithm`` is ``"ismcts"``, ``"pimc"`` or
    ``"puct"`` (``iterations`` are its simulations; ``options`` must hold ``net``).

    Extra keyword options are passed to :func:`big2._core.analyze`
    (``exploration``, ``threads``, ``time_limit``, ``objective``,
    ``endgame_threshold``, ``rollout_epsilon``, ``net``, ``c_puct``, ...).
    """

    algorithm: str = "ismcts"
    iterations: int = 1000
    seed: int = 0
    options: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    _calls: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            extra = "".join(f",{k}={v}" for k, v in sorted(self.options.items()))
            self.name = f"{self.algorithm}:iterations={self.iterations}{extra}"

    def analyze(self, state: Big2State) -> AnalysisResult:
        """Full solver output for the player to act."""
        self._calls += 1
        return analyze(
            state,
            algorithm=self.algorithm,
            iterations=self.iterations,
            seed=(self.seed * 1_000_003 + self._calls) % 2**64,  # the engine takes a u64
            **self.options,
        )

    def act(self, state: Big2State) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            # A forced move (usually a pass) needs no search; under a time limit it
            # would otherwise cost the full budget. The call still counts, so every
            # later search gets the same seed as before this shortcut existed.
            self._calls += 1
            return int(legal[0])
        return self.analyze(state)["moves"][0]["cards"]


def _parse_value(v: str) -> Any:
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def make_agent(spec: str, seed: int | None = None) -> Agent:
    """Builds an agent from a spec string like ``"ismcts:iterations=2000"``.

    Kinds: ``random``, ``greedy[:epsilon=…,structured=1,plan_depth=…]``,
    ``ismcts[:…,rollout=structured]``, ``pimc[:…]``,
    ``puct:net=….pknn[,simulations=2000,c_puct=1.5,…]`` (the network inside the search;
    ``leaf_network_weight=0.5`` mixes a rollout into the leaf value, ``rollout_moves=…``
    cuts it short, ``rollout=…`` picks its policy as for ISMCTS),
    ``patwa-greedy``, ``patwa-smart[:<weight>=…]`` (see :mod:`big2.agents.patwa`),
    ``qnet[:path=….pknn,temperature=…]`` (see :mod:`big2.agents.qnet`),
    ``pvnet:checkpoint=….pt[,mode=score|prior,scoring=…]`` (a network v2 played greedily,
    see :class:`big2.rl.agent.PvNetAgent`),
    ``dmc:checkpoint=…[,objective=…]``, ``dmcsearch:checkpoint=…[,determinizations=…,top=…]``.
    """
    kind, _, rest = spec.partition(":")
    opts = {
        k.strip(): _parse_value(v.strip())
        for k, v in (p.split("=", 1) for p in rest.split(",") if p)
    }
    kind = kind.strip().lower()
    if kind == "random":
        return RandomAgent(seed=seed, name=spec)
    if kind == "greedy":
        return GreedyAgent(
            epsilon=float(opts.pop("epsilon", 0.0)),
            seed=seed,
            name=spec,
            structured=bool(opts.pop("structured", False)),
            plan_depth=int(opts.pop("plan_depth", 8)),
        )
    if kind in ("ismcts", "pimc"):
        iterations = int(opts.pop("iterations", 1000 if kind == "ismcts" else 100))
        return SearchAgent(
            algorithm=kind, iterations=iterations, seed=seed or 0, options=opts, name=spec
        )
    if kind == "puct":
        if "net" not in opts:
            raise ValueError(f"puct needs net=<path to .pknn> in {spec!r}")
        simulations = int(opts.pop("simulations", 2000))
        return SearchAgent(
            algorithm="puct", iterations=simulations, seed=seed or 0, options=opts, name=spec
        )
    if kind in ("patwa-greedy", "patwa-smart"):
        # Approximations of Patwa (2026)'s rule-based opponents (big2.agents.patwa).
        if kind == "patwa-greedy":
            return PatwaGreedy(name=spec)
        return PatwaSmart(name=spec, **opts)
    if kind == "qnet":
        # NumPy copy of the Rust QNetPolicy, reading the exported .pknn (big2.agents.qnet).
        from big2.agents.qnet import DEFAULT_QNET, QNetAgent

        return QNetAgent(
            path=str(opts.pop("path", DEFAULT_QNET)),
            temperature=float(opts.pop("temperature", 1.0)),
            seed=seed,
            name=spec,
        )
    if kind == "pvnet":
        if "checkpoint" not in opts:
            raise ValueError(f"pvnet needs checkpoint=<path to a v2 .pt> in {spec!r}")
        from big2.rl.agent import PvNetAgent

        return PvNetAgent(str(opts.pop("checkpoint")), name=spec, **opts)
    if kind in ("dmc", "dmcsearch"):
        if "checkpoint" not in opts:
            raise ValueError(f"{kind} needs checkpoint=<path to .pt> in {spec!r}")
        # Imported lazily so the rest of the package works without PyTorch.
        if kind == "dmc":
            from big2.rl.agent import DMCAgent

            return DMCAgent(str(opts.pop("checkpoint")), name=spec, **opts)
        from big2.rl.search import RolloutSearchAgent

        return RolloutSearchAgent(str(opts.pop("checkpoint")), seed=seed or 0, name=spec, **opts)
    raise ValueError(
        f"unknown agent kind {kind!r} in {spec!r} "
        "(random, greedy, ismcts, pimc, puct, patwa-greedy, patwa-smart, qnet, pvnet, dmc, "
        "dmcsearch)"
    )


__all__ = [
    "Agent",
    "RandomAgent",
    "GreedyAgent",
    "SearchAgent",
    "PatwaGreedy",
    "PatwaSmart",
    "make_agent",
]

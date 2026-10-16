# poker — card-game solver & RL lab

A research codebase for studying imperfect-information card games, starting with
**Big Two** (鋤大D). It pairs a fast Rust rules engine with:

1. **A search-based solver** — information-set Monte Carlo tree search with an exact
   endgame solver — that recommends moves with estimated win rates.
2. **A playable UI** against solver bots, with a study mode showing the top 3 moves.
3. **Reinforcement learning** agents, compared head-to-head with the solver.

> Status: **Phase 1 complete** (rules engine, solver, CLI). Next: Phase 2 (web UI
> with study mode), then Phase 3 (reinforcement learning).

## Quick start

```sh
conda env create -f environment.yml
conda activate poker
make dev     # builds the Rust extension and installs the `poker` Python package
make test
```

## Using it

```sh
# Play a deal in the terminal against three solver bots ("?" asks for a hint)
poker big2 play
poker big2 play --hints --bots greedy        # easier bots, top-3 hint every turn

# Ask the solver about a position, from one seat's point of view
poker big2 analyze --seat 0 --hand "3d 4d 4h 5h 6d 7d 7s 8d 8s Jd Jh Kc 2s"
poker big2 analyze --seat 1 --hand "..." --history "0:3d; 1:pass; 2:5c"

# Duplicate tournament (every deal played 4×, seats rotated)
poker big2 tournament ismcts:iterations=1000 greedy greedy greedy --deals 200 --workers 6
```

Every command takes the rule variants as options (`--straights no_twos`,
`--scoring original`, `--pass-rule locked_out`, ...); see `poker big2 <cmd> --help`.

From Python:

```python
from poker._core import analyze
from poker.games.big2 import Rules, pretty

state = Rules().new_deal(seed=1)
for m in analyze(state, iterations=20_000, threads=4)["moves"][:3]:
    print(pretty(m["cards"]), f"win {m['win_rate']:.1%}", f"score {m['mean_score']:+.1f}")
```

## Documentation

| Document | What it covers |
|---|---|
| [docs/rules/big2.md](docs/rules/big2.md) | The exact Big Two rules implemented, with variants |
| [docs/solver.md](docs/solver.md) | What the solver computes, how, and how far to trust it |
| [docs/study-results/](docs/study-results/) | Measured results of each experiment |
| [docs/architecture.md](docs/architecture.md) | Code layout, correctness strategy, performance |
| [docs/decisions/](docs/decisions/) | Why the project is built the way it is |
| [CLAUDE.md](CLAUDE.md) | Commands, layout and conventions for contributors |

## Repository layout

```
crates/        Rust: cards, games (traits), big2 (engine), search (solvers), pybind
python/poker/  Python: games helpers, agents, eval (tournaments), cli
experiments/   Scripts that produce the results in docs/study-results/
docs/          Rules spec, solver notes, architecture, decisions, results
```

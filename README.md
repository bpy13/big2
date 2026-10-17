# poker — card-game solver & RL lab

A research codebase for studying imperfect-information card games, starting with
**Big Two** (鋤大D). It pairs a fast Rust rules engine with:

1. **A search-based solver** — information-set Monte Carlo tree search with an exact
   endgame solver — that recommends moves with estimated win rates.
2. **A playable UI** against solver bots, with a study mode showing the top 3 moves.
3. **Reinforcement learning** agents (DMC, PPO, search distillation), compared
   head-to-head with the solver.

## Findings so far

| Agent | vs the field (head-to-head) | ms / move |
|---|---:|---:|
| ISMCTS search solver | +1.40 | 14 |
| Distilled network + search | +1.31 | 188 |
| PIMC search | +0.76 | 2 |
| Distilled network | +0.17 | 1 |
| DMC self-play network | −1.07 | 1 |
| PPO self-play network | −1.49 | 1 |

Search beats self-play learning at this compute budget (1–1.5 CPU-hours of
training); distilling the solver into a network gives nearly the solver's
strength at a fraction of the cost. Details: [Phase 1](docs/study-results/phase1-solver.md),
[Phase 3](docs/study-results/phase3-rl.md).

> Status: **Phases 1–3 complete** — rules engine, search solver, CLI, web UI with
> study mode, and reinforcement-learning agents compared head-to-head with the solver.

## Quick start

```sh
conda env create -f environment.yml
conda activate poker
make dev     # builds the Rust extension, installs the `poker` package (with PyTorch) and UI deps
make test
```

Works on macOS, Windows and Linux. On Windows, install the Visual Studio Build
Tools (C++ workload) first, for the Rust linker. To train on an NVIDIA GPU, run
`make torch-cuda` before `make dev` — see [docs/rl.md §6](docs/rl.md#6-hardware-cpu-nvidia-gpu-apple-gpu).

## Using it

```sh
# Web UI: play against solver bots, with study mode (top-3 hints, quiz, post-game review)
make serve                                   # then open http://127.0.0.1:8000

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
| [docs/ui.md](docs/ui.md) | Web UI, study mode, review, game logs, API |
| [docs/rl.md](docs/rl.md) | Reinforcement learning: methods, models, commands |
| [docs/study-results/](docs/study-results/) | Measured results of each experiment |
| [docs/architecture.md](docs/architecture.md) | Code layout, correctness strategy, performance |
| [docs/decisions/](docs/decisions/) | Why the project is built the way it is |
| [CLAUDE.md](CLAUDE.md) | Commands, layout and conventions for contributors |

## Repository layout

```
crates/        Rust: cards, games (traits), big2 (engine), search (solvers), pybind
python/poker/  Python: games helpers, agents, eval (tournaments), cli, server (FastAPI), rl
web/           React + TypeScript UI (Vite); browser tests in web/tests
experiments/   Scripts that produce the results in docs/study-results/
models/        Trained network used by the web UI
docs/          Rules spec, solver notes, architecture, decisions, results
```

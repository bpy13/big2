# poker — card-game solver & RL lab

A research codebase for studying imperfect-information card games, starting with
**Big Two** (鋤大D). It pairs a fast Rust rules engine with:

1. **A search-based solver** — information-set Monte Carlo tree search with an exact
   endgame solver — that recommends moves with estimated win rates.
2. **A playable UI** against solver bots, with a study mode showing the top 3 moves.
3. **Reinforcement learning** agents, compared head-to-head with the solver.

> Status: Phase 1 in progress — rules engine done; solver next.

## Quick start

```sh
conda env create -f environment.yml
conda activate poker
make dev     # builds the Rust extension and installs the `poker` Python package
make test
```

## Documentation

| Document | What it covers |
|---|---|
| [docs/rules/big2.md](docs/rules/big2.md) | The exact Big Two rules implemented, with variants |
| [docs/decisions/](docs/decisions/) | Why the project is built the way it is |
| [CLAUDE.md](CLAUDE.md) | Commands, layout and conventions for contributors |

## Engine performance

Apple M1, one core, default rules (`make bench`):

| Benchmark | Result |
|---|---|
| Random playout from a fresh deal | ~29 µs per game, ~2.5 M moves/s |
| Greedy-bot playout from a fresh deal | ~18 µs per game |
| Legal moves for a 13-card leading hand | ~2 µs |

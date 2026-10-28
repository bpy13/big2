# Architecture

```mermaid
flowchart TB
    subgraph rust["Rust (crates/)"]
        cards["cards<br/>Rank, Suit, Card, text notation"]
        games["games<br/>Game / Policy / Rewind / Evaluator traits"]
        nn["nn<br/>MLP inference (.pknn)"]
        big2["big2<br/>rules · combos · movegen · State · bots · features · value net"]
        search["search<br/>ISMCTS · PIMC · max^n endgame · belief sampling"]
        pybind["pybind<br/>PyO3 module big2._core"]
        cards --> big2
        games --> big2
        nn --> big2
        games --> search
        big2 --> pybind
        search --> pybind
    end
    subgraph py["Python (python/big2/)"]
        helpers["games/big2.py<br/>parsing · Rules · observation → state"]
        agents["agents/<br/>random · greedy · ismcts · pimc"]
        evalm["eval/<br/>duplicate tournaments"]
        cli["cli/<br/>big2 play · analyze · tournament"]
        server["server/<br/>FastAPI · game sessions · review"]
        rl["rl/<br/>DMC · PPO · distillation · network search"]
    end
    web["web/<br/>React + TypeScript"]
    pybind --> helpers --> agents --> evalm --> cli
    agents --> server --> web
    pybind --> rl
```

## Layers

**`crates/cards`** knows what a card is and how to write one (`"Ah"`, `"Tc"`), but
not how cards are ordered: that depends on the game.

**`crates/games`** defines the interface every game implements (`Game`): current
player, legal actions, apply, outcome, and `determinize` — sample hidden cards
consistent with one player's view. The search crate only talks to this trait, so a
second game (e.g. Hold'em) gets the solvers for free.

**`crates/nn`** runs small fully connected networks exported from PyTorch
(`.pknn` files), so the search can call a learned value network without Python.

**`crates/big2`** is the Big Two engine. Hands are `u64` masks with card index
`rank*4 + suit` (ranks 3…2), so "higher index = stronger single". A combination's
strength is precomputed into one integer key, so "does X beat Y" is one comparison.
Move generation builds combinations from per-rank suit sets rather than testing
subsets. `docs/rules/big2.md` is the spec; tests cite its sections.

**`crates/search`** holds the solvers (see [solver.md](solver.md)). They return an
`Analysis`: every legal action with simulations, mean score, win rate and 95%
confidence intervals.

**`crates/pybind`** exposes all of the above to Python as plain values: card masks
are `int`, an action is the mask of cards played (`0` = pass), rules are keyword
strings. `python/big2/_core.pyi` documents and types it.

**`python/big2`** is everything above the engine: helpers, agents built from spec
strings (`"ismcts:iterations=2000"`), the tournament runner, and the CLI.

## Correctness strategy

| Layer | Check |
|---|---|
| Combination rules | Unit tests per rule-doc example; move generator vs brute-force subset search on random hands |
| Turn flow, scoring | Scenario tests per rule-doc section; property tests (cards conserved, zero-sum, termination) under random rule variants |
| Whole engine | Differential test against an independent pure-Python implementation (`python/tests/reference/`), checking the full legal-move set at every step (5,000 games agree) |
| Solvers | Endgame solver vs naive max^n; immediate wins found; reproducible with a seed; **identical output for positions that differ only in hidden cards** (the solver cannot peek) |

## Performance (Apple M1, one core)

| Operation | Speed |
|---|---|
| Random playout of a full deal | ~29 µs (~2.5 M moves/s) |
| Greedy playout of a full deal | ~18 µs |
| Legal moves, 13-card leading hand | ~2 µs |
| ISMCTS, 1,000 simulations (opening position) | ~20 ms |

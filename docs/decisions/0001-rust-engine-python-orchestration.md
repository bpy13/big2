# 0001 — Rust engine, Python orchestration

**Status:** accepted (2026-10-16)

## Context

Both the search solver (Phase 1) and RL self-play (Phase 3) spend nearly all their
time in three operations: generating legal moves, applying a move, and playing a
game to the end. A Monte Carlo solver's accuracy grows with the number of games it
can simulate per decision, so engine speed translates directly into solver strength
and tighter win-rate estimates. Pure Python runs these operations roughly 50–100×
slower than compiled code.

At the same time, RL training (PyTorch), the CLI, the web server and the analysis
notebooks are most productive in Python.

## Decision

* Rules, move generation, state, and the search solvers are written in Rust
  (`crates/`), built into one Python extension module `poker._core` with PyO3 and
  maturin.
* Everything else is Python (`python/poker/`): agents, evaluation, CLI, FastAPI
  server, RL.
* The Rust toolchain lives inside the `poker` conda environment (conda-forge
  `rust`), not in a global rustup install.

## Consequences

* Two toolchains. `make build` hides this; the Python side never sees Rust types,
  only plain ints, strings and lists.
* Correctness is guarded twice: Rust unit and property tests, plus a differential
  test against an independent pure-Python rules oracle
  (`python/tests/reference/`).

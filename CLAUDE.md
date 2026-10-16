# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Environment

Everything runs inside the `poker` conda env (Python, Rust, Node, maturin):

```sh
conda env create -f environment.yml   # once
conda activate poker
make dev                              # build the Rust extension + install Python package (editable)
```

## Commands

| Task | Command |
|---|---|
| All tests | `make test` (Rust + Python) |
| Rust tests only | `cargo test` |
| Python tests only | `pytest` |
| Lint / format | `make lint`, `make fmt` |
| Engine benchmark | `make bench` |

## Layout

* `crates/` — Rust: `cards` (vocabulary), `games` (Game/Policy traits), `big2`
  (rules engine), `search` (solvers), `pybind` (PyO3 module `poker._core`).
* `python/poker/` — Python package: agents, evaluation, CLI, server, RL.
* `web/` — React + TypeScript UI.
* `docs/` — rules spec (`docs/rules/big2.md` is authoritative), architecture,
  decisions (ADRs), study results.

## Conventions

* The rules doc is the spec. A rules change updates `docs/rules/big2.md`, the Rust
  engine, and the Python reference oracle together, each with a test.
* Big Two card index: `rank * 4 + suit`, ranks `3=0 … 2=12`, suits `♦=0 ♣=1 ♥=2 ♠=3`.
  Hands are `u64` masks. Text notation: `3d Tc Ah 2s`.
* Public Rust items get rustdoc; public Python functions get docstrings.
* Commit in small, descriptive steps.

# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Environment

Everything runs inside the `big2` conda env (Python, Rust, Node, maturin):

```sh
conda env create -f environment.yml   # once
conda activate big2
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
| Web UI + API | `make serve` (http://127.0.0.1:8000) |
| UI dev server | `uvicorn big2.server.app:app --reload` + `make web-dev` |
| Browser tests | `make test-e2e` (once: `cd web && npx playwright install chromium`) |
| RL training | `big2 rl train` / `big2 rl ppo` / `big2 rl distill-gen` + `distill-train` (see docs/rl.md) |
| NVIDIA GPU PyTorch | `make torch-cuda` before `make dev` (trainers take `--device auto\|cpu\|cuda\|mps`) |

## Layout

* `crates/` — Rust: `cards` (vocabulary), `games` (Game/Policy traits), `big2`
  (rules engine), `search` (solvers), `pybind` (PyO3 module `big2._core`).
* `python/big2/` — Python package: agents, evaluation, CLI, server, RL.
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
* Do not rebuild the extension (`pip install -e .`) while a training job runs: the
  running process has the old `_core` library loaded.
* Must run on macOS, Windows and Linux: use `pathlib`, no shell-specific Makefile
  recipes, and keep tensors' device explicit (tests run the trainers on every
  device present).
* Node tools must run inside the conda env (`conda activate big2`); the system
  Node is too old for Vite.

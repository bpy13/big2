# Reinforcement learning

Phase 3 trains neural networks to play Big Two and compares them with the
Phase 1 search solver. Full results: [study-results/phase3-rl.md](study-results/phase3-rl.md).

**In one paragraph:** two self-play methods (DMC and PPO) learned to beat the greedy
bot but stayed about 2 points per game behind the solver after 1–1.5 hours of
laptop training. Distilling the solver into the network worked much better: the
distilled network nearly matches the solver (−0.5 ± 1.0 points per game) while
choosing moves ~13× faster, and with a little search on top it ties the solver.
The web UI uses the distilled network.

## 1. Search vs learning

| | Search solver (Phase 1) | Network (Phase 3) |
|---|---|---|
| When it thinks | At the table, every move | Beforehand, during training |
| Time per move | 2–15 ms (more for hints) | ~1 ms |
| Hidden cards | Uniform random guesses | Learns what the history implies |
| Model of opponents | Greedy rollout bot | Whatever it was trained against |
| Uncertainty | 95% intervals from simulations | None built in |

## 2. What the network sees and outputs

Features are 0/1 values computed in Rust (`crates/big2/src/features.rs`) from the
acting player's viewpoint, with seats relative to them (next, across, previous):
your hand; the cards you have not seen; the cards each seat has played; each
opponent's card count; the play to beat and who made it; whether you lead;
passes so far this trick; the last 12 actions. A test checks that two positions
differing only in hidden cards encode identically — the network cannot cheat.

Each candidate move is encoded as its cards, its kind, and the hand you would
have left after it.

```
state (1,103) ─ 512 ─ 256 ─┐
                            ├─ 256 ─ 128 ─ 2   (expected score, win logit)
move (61 + 52) ─ 128 ──────┘
```

The state tower runs once per decision; only the small head runs per legal move.
About 0.8 M parameters, 3.3 MB on disk. The network plays the move with the
highest predicted score (or win probability, with `objective=win`).

## 3. Methods

### 3.1 Deep Monte Carlo self-play (DouZero) — `big2 rl train`

DouZero (Zha et al., ICML 2021) beat all earlier programs at Dou Dizhu, a close
relative of Big Two, with a simple recipe:

1. **Play** games of the current network against itself; at each decision score
   every legal move and play the best (ε = 5% random moves to explore).
2. **Label** each (position, move) with the final score of the player who made it.
3. **Fit** the network to those labels (squared error; plus cross-entropy for the
   win head), keeping the last 4 iterations' data. Repeat.

One iteration = 1,024 games (~45,000 decisions). On the M1 CPU (shared with
other jobs) it took ~9 s: ~2.3 s playing the games and ~6.6 s of gradient steps.
The Rust batched environment (`Big2VecEnv`) plays ~13,000 random games/s, so the
network — mostly learning, partly choosing moves — is the bottleneck; see §6.

### 3.2 PPO self-play — `big2 rl ppo`

Proximal Policy Optimization (Schulman et al., 2017), as in Charlesworth's Big
Two agent (2018). Instead of move values, the network outputs move
probabilities (softmax over the legal moves) plus a position value:

1. Play self-play games, *sampling* moves from the policy.
2. Advantage = final score − predicted value: did it go better than expected?
3. For 4 epochs, raise the probability of better-than-expected moves and lower
   the others, clipped so one batch cannot change the policy much; regress the
   value toward the final score; small entropy bonus for exploration.

### 3.3 Search distillation (expert iteration) — `big2 rl distill-gen` + `distill-train`

Self-play learns from one noisy number per decision. The solver already
estimates every promising move from hundreds of simulated games. Distillation
uses those estimates as targets:

1. **Generate:** worker processes play games in which ISMCTS (1,000 simulations)
   analyzes every decision. Store the state and, for each move simulated at least
   20 times, its mean score, win rate and simulation count. Play the solver's best
   move (25% of the time, a random one of its top 3, for variety).
2. **Train:** regress the network toward those values, weighting each move by its
   simulation count.

This is the "expert iteration" idea behind AlphaZero, with ISMCTS as the expert.
The network becomes a fast copy of the solver, including its assumption that
opponents play like the greedy bot.

### 3.4 Network + search — `dmcsearch:` agents

`python/big2/rl/search.py` combines a network with Monte Carlo search:

1. The network ranks the legal moves; keep the best 6.
2. Sample 32 deals of the hidden cards.
3. Play each (deal, candidate) to the end with **the network playing every seat**
   (instead of the greedy bot), all 192 games in lockstep with batched network calls.
4. Pick the candidate with the best average result.

## 4. Models in `models/`

| File | Method | Last-card duty | Committed |
|---|---|---|---|
| `big2-distilled-duty.pt` | Distillation, 4 h from `big2-distilled-10h.pt`; the web UI's default | on (current default) | yes |
| `big2-distilled-10h.pt` | Distillation, +291 min ([long-training.md](study-results/long-training.md)) | off | yes |
| `big2-dmc-selfplay-10h.pt` | DMC self-play, +10 h | off | no (retrain) |
| `big2-ppo-10h.pt` | PPO self-play, +10 h | off | no (retrain) |
| `big2-distilled.pt` | Distillation (Phase 3) | off | yes |
| `big2-dmc-selfplay.pt` | DMC self-play, 90 min | off | no (retrain) |
| `big2-ppo.pt` | PPO self-play, 60 min | off | no (retrain) |
| `big2-distilled-finetuned.pt` | Distilled, then 60 min DMC self-play | off | no (retrain) |

A network does not know other rule variants than the one it was trained under.
The web UI lists the networks in `python/big2/server/models.py` whose files
exist, each with its rules, and offers a network only for games under those rules.
The trainers take `--last-card-duty/--no-last-card-duty` (default: on).

### 4.1 The network in Rust: `big2 rl export`

The engine (and so the app, through WebAssembly) plays a Q-network without Python:

```sh
big2 rl export --checkpoint models/big2-distilled-duty.pt --out runs/p1_qnet/distilled-duty.pknn
```

writes a `.pknn` version 2 file (format in `crates/nn/src/lib.rs`) with three towers,
`state`, `action` and `head`. `big2_big2::qnet::QNet::from_bytes` reads it;
`score_moves(state, legal)` returns each move's predicted score in points and win
logit, and `QNetPolicy { net, temperature }` plays it (best move at temperature 0,
otherwise a softmax over the scores in points).

**Size and accuracy.** The state tower's first layer (two thirds of the weights) is
stored as int8 with one f16 scale per 32 weights, the other layers as f16, biases as
f32. The rounding of the first layer is offset by a bias correction measured on
2,000 self-play positions. For `big2-distilled-duty.pt` (2026-10-29):

| | Value |
|---|---|
| File size | 1.16 MB (f32: 3.4 MB; `--f32` writes that) |
| Same best move as PyTorch, 1,000 positions | 99.9 % (seeds 7 and 99: 99.8 %, 99.5 %) |
| Score difference, RMS | 0.030 points |
| Time per decision, native (`cargo bench -p big2-big2 --bench engine -- qnet`) | 33 µs |

Plain int8 with one scale per row (0.85 MB) was tried first and rejected: 93 % same
move, 0.75 points RMS. `python/tests/test_export.py` checks the comparison: it builds
the `qnet_dump` example (`crates/big2/examples/qnet_dump.rs`), which plays seeded
games in Rust and writes the features and scores, and scores the same features in
PyTorch.

## 5. Commands

```sh
# Self-play (all trainers take --device auto|cpu|cuda|mps, see §6)
big2 rl train --run-dir runs/dmc/v1 --minutes 90             # DMC
big2 rl train --run-dir runs/dmc/ft --init models/big2-distilled.pt   # DMC from given weights
big2 rl ppo --run-dir runs/ppo/v1 --minutes 60                # PPO

# Distillation (run the two side by side; the trainer picks up new data as it appears)
big2 rl distill-gen --out-dir runs/distill/data --workers 4 --minutes 60
big2 rl distill-train --data-dir runs/distill/data --run-dir runs/distill/v1 --minutes 75

# Continue a run (weights, optimizer, counters); every trainer takes --resume
big2 rl train --run-dir runs/dmc/v1_more --resume runs/dmc/v1/latest.pt --minutes 600
big2 rl ppo --run-dir runs/ppo/v1_more --resume runs/ppo/v1/latest.pt --minutes 600
big2 rl distill-train --run-dir runs/distill/more --resume runs/distill/v1/latest.pt
# Sharing the CPU between jobs: --threads N (PyTorch), --workers N (distill-gen)

# Use a network
big2 tournament dmc:checkpoint=models/big2-distilled.pt ismcts:iterations=1000 \
    dmc:checkpoint=models/big2-distilled.pt ismcts:iterations=1000 --deals 100 --workers 6

# Reproduce the Phase 3 study
python experiments/phase3_rl.py round-robin --deals 100 --workers 7
python experiments/phase3_rl.py agreement --positions 300
```

Agent specs: `dmc:checkpoint=PATH[,objective=win]` (any network, including PPO and
distilled ones) and `dmcsearch:checkpoint=PATH[,determinizations=32,top=6]`.

Each run directory holds `log.jsonl` (one line per iteration or evaluation),
`config.json`, `latest.pt` (resumable) and `best.pt` (best evaluation vs greedy).

### 3.5 Value network for search leaves — `big2 rl value-train`

Trains a small MLP on full-information positions from distilled-network
self-play to predict every seat's final score and the winner, and exports it
for the Rust search (`analyze(..., value=...)`; see
[solver.md §6.2](solver.md#62-learned-leaf-values-value)).

## 6. Hardware: CPU, NVIDIA GPU, Apple GPU

Every trainer takes `--device`:

| `--device` | Learning (gradient steps) | Choosing moves (self-play, evaluation) |
|---|---|---|
| `auto` (default) | NVIDIA GPU if PyTorch sees one, else CPU | same |
| `cuda` | NVIDIA GPU | NVIDIA GPU |
| `mps` | Apple GPU | CPU (small batches are faster there) |
| `cpu` | CPU | CPU |

The line printed at start-up and `config.json` show which device was used.
`distill-gen` is unaffected: it runs the Rust solver, which uses CPU cores only
(`--workers`).

**Where the time goes.** Training has two network workloads (`log.jsonl` records
both as `collect_s` and `train_s`):

* *Choosing moves.* `--envs` games (default 256) are played in lockstep: each step
  the Rust environment hands over every pending decision with all its legal moves,
  and the network scores them in **one batched call** (the state tower runs once
  per decision, only the small move head per legal move). A GPU makes the call
  cheap, after which per-step overhead dominates; raise `--envs` (e.g. 1024–4096)
  so each call does more work.
* *Learning.* Minibatches of 2,048 samples — this is where a GPU helps most, and in
  the Phase 3 DMC runs it was ~75% of the time.

**NVIDIA GPU setup** (Windows or Linux):

```sh
conda env create -f environment.yml && conda activate big2
make torch-cuda      # PyTorch's CUDA build; prints "CUDA available: True"
make dev
pytest python/tests/test_rl.py    # also runs the trainers on the GPU
big2 rl train --run-dir runs/dmc/gpu --minutes 600 --envs 1024
```

On Windows, `pip install torch` alone installs a CPU-only build — hence
`make torch-cuda`. Building the Rust extension needs the Visual Studio Build Tools
(workload "Desktop development with C++"). WSL2 with Ubuntu also works, using the
Linux steps.

## 7. Limits

* **Compute.** DouZero trained on billions of decisions with many GPUs. These runs
  used 1–1.5 hours on a laptop CPU, several at once. The results describe this
  budget, not the methods' ceilings.
* **Evaluation vs greedy is biased.** Agents that learned from the solver inherit
  its greedy-opponent model and score extra against greedy bots. Compare agents
  head-to-head (`round-robin`).
* **Self-play only finds what it plays against.** Nothing guarantees a network is
  hard to exploit by a different style, including human play.

## References

* Zha et al. *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement
  Learning.* ICML 2021.
* Schulman et al. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
* Charlesworth. *Application of Self-Play Reinforcement Learning to a Four-Player
  Game of Imperfect Information* (Big Two). arXiv:1808.10442, 2018.
* Anthony, Tian, Barber. *Thinking Fast and Slow with Deep Learning and Tree
  Search* (expert iteration). NeurIPS 2017.
* Silver et al. *Mastering the game of Go without human knowledge* (AlphaGo
  Zero). Nature, 2017.
* Tesauro, Galperin. *On-line Policy Improvement using Monte-Carlo Search.*
  NeurIPS 1996.

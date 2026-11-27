# Expert search budget (Phase 6, Tasks 37-40)

## 0. The self-play network in Rust (Task 37)

Network: `runs/selfplay_luck/hour_24.pt` (Task 36, hour 24; `qnet_v2`, state 512/256, action 128,
head 256), exported with `big2 rl export` to `runs/p6_net/selfplay-h24.pknn` (mixed int8/f16,
1.17 MB, gitignored, not shipped). The self-play trainer's checkpoint keys (`arch`, `model`,
`net_config`, plus `train_config`, `info`) match r3's, so `export.py` needed no mapping.

**Agreement with PyTorch** (`python/tests/test_export.py::test_selfplay_h24_network_in_rust_matches_pytorch`,
1,000 positions from greedy-prior self-play, seed 0, 5.3 legal moves on average):

| measure | result | bar |
|---|---|---|
| Rust prior top move = PyTorch's | 99.8 % | >= 99 % |
| value outputs (each seat's expected cards left), RMS difference | 0.0081 | <= 0.01 |
| per-seat cards-left distribution, L1 mean / p99 / max | 0.0024 / 0.0116 / 0.113 | (info) |

**Speed of one forward pass** (all legal moves of one position, prior and cards-left heads),
1,400 positions (seed 7, after 100 warm-up positions), one thread, this machine, release build:

| where | median ms | 95th percentile ms |
|---|---|---|
| native (`puct_timing` example, step `forward`) | 0.059 | 0.126 |
| Node 22 WebAssembly (`experiments/forward_timing.mjs`) | 0.246 | 0.553 |

WebAssembly is about 4x slower than native. Sources: `runs/p6_net/forward_native.json`, step
`forward 1500 7` of `crates/search-wasm`.

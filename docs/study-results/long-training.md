# Ten more hours of training

**Date:** 2026-10-18 · **Code:** `7bd39a2` · **Machine:** Apple M1, CPU only, all
three runs (plus distillation data generation) sharing the 8 cores
**Reproduce:** `poker rl train --resume …`, `poker rl ppo --resume …`,
`poker rl distill-train --resume …` (see [rl.md](../rl.md));
`python experiments/long_training.py --deals 200 --workers 7`
(raw JSON in `runs/long_training/`).
**Rules:** the defaults of the time, i.e. last-card duty **off** (the default is on since 2026-10-18).

Each Phase 3 network ([phase3-rl.md](phase3-rl.md)) was trained for another 10
hours from its checkpoint, then played 200 duplicate deals (A B A B, seats
rotated) against the solver (ISMCTS, 1,000 simulations) and against its own
Phase 3 version.

## Summary

1. **All three networks improved, and the two best now tie the solver.** No
   network beats it.
2. **Self-play DMC gained the most**: from about −2.2 to −0.2 points per game
   against the solver, without any solver data.
3. **The distilled network clearly beats its Phase 3 version (+1.3 ± 0.8)** but
   is still level with the solver it learned from, as expected for a copy of its
   teacher.
4. **The distillation run stopped after 4.9 hours**, not 10, because of a race in its data
   loader (fixed since: a file renamed by the data generator between listing and
   reading crashed the trainer).

## Results (points per game, from the network's side, 95% CI over deals)

| Network | Training | vs solver, Phase 3 | **vs solver, now** | **vs its Phase 3 version** |
|---|---|---:|---:|---:|
| DMC self-play | +600 min, 5.6 M games in total | −2.22 | **−0.17 ± 0.82** | +0.69 ± 0.95 |
| PPO self-play | +600 min, 2.7 M games in total | −2.76 | −1.42 ± 0.69 | +0.51 ± 0.71 |
| Distilled | +291 min on the newest 800 k of 4.0 M solver-labelled decisions | −0.54 | **−0.47 ± 0.76** | **+1.28 ± 0.84** |

Phase 3 numbers are from its round-robin (100 deals), so the "before" column is
noisier than the others.

Against three greedy bots (the trainers' own evaluation, 256 deals, final
checkpoint): DMC +5.7, PPO +4.3, distilled +7.7 (Phase 3: +4.0, +3.1, +6.5).

## Reading it

* **Self-play was not at a ceiling after 1.5 hours.** DMC kept improving for the
  whole 10 hours and closed most of the gap to the solver. It learns from noisy
  single-game returns, so it needs many games; on this laptop that means hours.
  A GPU (`--device cuda`) would mostly speed up its learning step.
* **Distillation is capped by its teacher.** Its gain over the Phase 3 version
  came from 5× more data, but its strength against the solver did not change.
  To go past the solver it needs a stronger teacher (expert iteration: network +
  search as the labeller).
* **PPO remains the weakest.** Sampling its own moves adds variance on top of the
  noisy returns.

## Where the networks are used

All four networks are in the web UI's network picker
([ui.md](../ui.md#solver-and-network-side-by-side)): the distilled +10 h network
is the default (`models/big2-distilled-10h.pt`, committed); the DMC and PPO
networks are listed when their files are present in `models/`.

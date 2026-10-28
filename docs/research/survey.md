# Big Two (鋤大D) bot survey: published work up to late 2026

**Date:** 2026-10-28. Written for the bot stream of hk-games. This file builds on
`big2\docs\research\prior-work.md` (2026-10-18) and does not repeat it: that file holds the
first comparison of our solver and networks with the literature and a five-item improvement
list. This file adds what that survey missed, reads the key papers in full, and proposes
measurable bars for "state of the art".

**How I read each source.** "Full text" means I read the arXiv HTML or the publisher page with
the results tables. "Abstract only" means the full text was paywalled or blocked to our fetch
(IEEE Xplore returned empty pages or HTTP 418; ScienceDirect and the NTUST thesis server
returned 403; two Tichu thesis PDFs were binary-only). Every abstract-only source is flagged
in its table row. Numbers are the papers' own; nothing was re-run here. Ideas of our own are
marked **(our idea)** and live mostly in `hard-problems.md`.

**Rules reminder.** Our rules (`big2\docs\rules\big2.md`): four players, 13 cards each, 3♦
leads, singles/pairs/triples/five-card hands, no bombs, tiered pairwise settlement by default
(1–7 cards ×1, 8–9 ×2, 10–12 ×3, 13 ×4), with variants for straights, flush order, pass
rule, last-card duty and scoring. Every paper below uses a different scoring rule, so no
published number transfers to our scoring without a re-run.

## Summary

1. **Published Big Two AI is still five works, and none evaluates in a way we can reuse.**
   Charlesworth 2018 (PPO self-play, 664 human games), Chen & Lu 2022 (Big2AI: MCTS/ISMCTS
   plus opponent move prediction; abstract only), Luo & Tan 2024 (opponent model plus action
   filtering for DMC; abstract only), Chen & Lu 2025 (Big2MDP: Markov decision process with
   explicit "free-playing right" exploration; abstract only, new since `prior-work.md`), and
   Patwa 2026 (PPO vs value methods, read in full; single seed, scripted opponents, no code).
   Two of them exist only as Android prototypes. Only Charlesworth published code and weights.
2. **The method frontier moved in Dou Dizhu, not Big Two.** The line is now DouZero (2021)
   → PerfectDou (2022) → VRPO network (2026, +0.152 vs PerfectDou) → Ataraxos (Nature 2026):
   the same VRPO-trained policy network plus a belief network and a 200-rollout test-time
   search, +0.199 vs PerfectDou and +0.107 over its own network without search. Dou Dizhu
   compute for the whole Ataraxos stack was **2 days on 4 H100 for the policy and 8 hours for
   the belief network**, which is within one to two weeks on our RTX 4060 Ti (own estimate).
3. **The cheap wins in the Dou Dizhu literature are the ones we have not done.** Full-information
   critic (PerfectDou), opponent-hand prediction (DouZero+, Luo & Tan), candidate pruning to
   top-k (DanZero+, AlphaDou), oracle guiding (OADMCDou, IJCAI 2024: +0.104 ADP vs DouZero on
   a small budget), and variance-reduced advantages (VRPO). All run on one GPU in days.
4. **Search on top of a learned network is the common thread of every leader** (Ataraxos,
   DanZero+'s PPO over DMC candidates, our own distilled-net + search tie). Our Phase 3 result
   that network + rollout search only ties ISMCTS says our network is the weak part, not the
   idea.
5. **Belief is worth something but not much from history alone.** Our measurement: a cheating
   search gains +2.95 points per game, but the history moves the belief only from 0.37 to
   0.40 per card. Ataraxos ablates its belief net only in Stratego, not Dou Dizhu. Expect a
   learned belief to be a one-point improvement at most in Big Two; it matters more for
   teaching ("your 2♠ is the highest card still out") than for strength.
6. **"State of the art for Big Two" has no agreed yardstick.** The only reusable published
   targets are Charlesworth's released network (as an opponent under his scoring) and Patwa's
   win rates against his three scripted bots (re-implementable from his descriptions). §6
   proposes four bars, from "beats every published Big Two agent under its own protocol" to
   "Ataraxos-class method parity measured our way".
7. **The phone is the binding constraint, not training.** A 0.8 M-parameter MLP runs in under a
   millisecond in Rust; ISMCTS at 5,000 simulations fits ~0.5 s on a phone (app stream
   estimate). An Ataraxos-style search (200 full rollouts through a 1.9 M-parameter
   transformer per move) does not fit 2 s on one WebAssembly thread (own estimate, §5). The
   expert bot needs either a distilled policy-value network inside a small PUCT search or a
   server.
8. **LLMs are not a route to strength**, but they are a data point for teaching: fine-tuned
   7–9 B models reach 0.85 win rate in Dou Dizhu from DouZero trajectories (LLM4CardGame,
   2025), and theory-of-mind prompting improves LLM agents in Guandan. Neither beats the
   teacher. Not on a phone.

## 1. What changed since `prior-work.md` (2026-10-18)

| Item | Status then | Status now |
|---|---|---|
| Chen & Lu 2022 (Big2AI) | not accessible | Abstract confirmed via Semantic Scholar: MCTS + ISMCTS with card-superiority analysis, dynamic weights, learned game features and multi-opponent multi-move prediction; Android prototype; "highest win rate and fewest lost points" vs computer and human players, no numbers in the abstract. Marked open access, but the PDF is blocked to our fetch |
| Chen & Lu 2025 (Big2MDP) | unknown | New: IEEE Trans. on Games 17(2):267–281, DOI 10.1109/TG.2024.3424431. MDP framework that "balances scoring opportunities against losing risk, predicts opponents' moves, and decides when to seek the free-playing right". Android; beats "existing AI methods" vs computers and humans. Paywalled; abstract only |
| Luo & Tan 2024 (ODMC) | anonymous "Applied Soft Computing 158" | Authors Qian Luo and Tien-Ping Tan (Universiti Sains Malaysia), DOI 10.1016/j.asoc.2024.111545. Same group as OADMCDou (IJCAI 2024). Still abstract only (403) |
| Patwa 2026 | abstract + partial | Read in full (arXiv HTML v2, 28 June 2026). Details in §2 |
| Charlesworth 2018 | paper only | Code, environment and a trained checkpoint are public at `github.com/henrycharlesworth/big2_PPOalgorithm` (TensorFlow 1, `modelParameters136500`); no licence file visible; playable web version linked |
| NTUST 2024 PhD thesis (SSBC) | unknown | A Taiwanese thesis proposes "Strategic Stage-Based Control" for 大老二 (opening/middle/endgame rules, not RL). Server returned 403; search snippet only |
| Ataraxos | press + abstract | arXiv HTML v2 read in full, including the Dou Dizhu supplement. Details in §3 |
| VRPO (Fan & Farina 2026) | not covered | Read in full for Dou Dizhu. Its numbers equal the "policy network" row of Ataraxos exactly (+0.152 vs PerfectDou, +0.286 vs DouZero), so Ataraxos' Dou Dizhu network is a VRPO network |
| OADMCDou (IJCAI 2024), AlphaDou, DouRN, GuanZero, DanZero+ details, LLM work | not covered | §3 |

## 2. Big Two: the five published works

### 2.1 Charlesworth 2018 (full text)

Henry Charlesworth, *Application of self-play reinforcement learning to a four-player game of
imperfect information*, arXiv:1808.10442 (also AAAI 2019 RLG workshop).
<https://arxiv.org/abs/1808.10442>

- **Scoring:** the winner gets the sum of the other three players' remaining cards; each loser
  gets minus their own count. No tiers, no pairwise settlement.
- **Actions:** a fixed list of 1,695 indexed actions (1,287 five-card, 330 four-card, 31
  three-card, 33 two-card, 13 singles, pass) with a legality mask. Note the 330 four-card
  actions: his rules allow four-card plays (two pairs or four of a kind), which ours do not.
- **Input:** 412 hand-made binary features (own hand with pair/straight flags, per-opponent card
  counts and "has played a top-8 card", last play type, pass count, control).
- **Network:** 512 → two 256 branches (policy over 1,695, value). PPO, γ 0.995, λ 0.95,
  entropy 0.02, lr 2.5e-4 annealed.
- **Compute:** 150 M steps ≈ 3 M games, "a single PC with four cores and a GPU", about 2 days.
- **Evaluation:** 7 amateur humans, 664 games against three copies of the agent; humans
  averaged −0.96 ± 0.38 (standard error) under his scoring; the agent won 22.5% of games
  (25% is par). One human was positive on 31 games.
- **Open:** environment, training code and checkpoint on GitHub (no licence file seen on the
  page; treat as "all rights reserved" until checked).

### 2.2 Chen & Lu 2022 and 2025 (abstract only)

Lien-Wu Chen and Yi Lu, *Challenging artificial intelligence with multi-opponent and
multi-movement prediction for the card game Big2*, IEEE Access 10:40661–40676, 2022,
DOI 10.1109/ACCESS.2022.3166932; and *Markov decision process-based artificial intelligence
with card-playing strategy and free-playing right exploration for four-player card game Big2*,
IEEE Trans. on Games 17(2):267–281, 2025, DOI 10.1109/TG.2024.3424431.

- 2022: "Big2AI" combines card-superiority analysis, dynamic weight adjustment, learned game
  features and prediction of several moves for several opponents, on MCTS and ISMCTS.
  Evaluated in an Android prototype against "existing AI players" and humans. No numbers in
  the abstract; 4 citations.
- 2025: "Big2MDP" models hidden hands as an MDP, balances scoring against risk, predicts
  opponent moves and explicitly decides when to seek the free-playing right (the lead).
  Same Android evaluation style. This is the only published work that names control of the
  lead as a first-class decision.
- Neither releases code. Neither reports a confidence interval in its abstract. Both are
  Taiwanese 大老二, so their scoring probably differs from ours (unverified).

### 2.3 Luo & Tan 2024 (abstract only)

Qian Luo and Tien-Ping Tan, *Improved learning efficiency of deep Monte-Carlo for complex
imperfect-information card games*, Applied Soft Computing 158, 2024.
<https://www.sciencedirect.com/science/article/abs/pii/S1568494624003193>

- An **opponent model** predicts hidden information and feeds the DMC Q-network; **ODMC**
  filters actions with a *minimum combination search* (a dynamic-programming hand
  decomposition) so DMC scores fewer candidates. Tested on Dou Dizhu and Big 2; "comparable
  strength to DMC with less training time". No numbers available to us.
- The same group's IJCAI 2024 paper (§3.3) is readable and shows their evaluation style:
  10,000 decks, ADP with 95% intervals.

### 2.4 Patwa 2026 (full text)

Aalok Patwa, *Self-play reinforcement learning under imperfect information in Big 2*,
arXiv:2605.28863 v2, 28 June 2026 (listed at ICML 2026 virtual). <https://arxiv.org/abs/2605.28863>

- **Scoring:** as Charlesworth (winner gets the sum, losers minus their count). Value methods
  divide by 13.
- **State:** 277-dim vector (hand padded to 13, trick indicator, seen cards, three opponent
  counts, pass count, three per-opponent played-card indicators). **Actions:** enumerated per
  state; each legal action is an 80-dim feature vector scored by a dot product with the state
  embedding (card embeddings + masked self-attention over the hand).
- **Training:** 5,000 batches × 64 games = 320,000 games per agent; a 6-core i7 laptop; 7–13
  hours per run. PPO lr 3e-5 warm-up + cosine, clip 0.2, γ 0.99, λ 0.95, 4 epochs, minibatch
  256. Value methods: no replay buffer, ε 0.5 → 0, target sync every 10 batches.
- **Opponents:** *random*; *greedy* = lowest non-pass action by (type, rank key), ignoring hand
  structure; *smart* = deterministic scorer that prefers immediate wins, shedding cards,
  keeping 2s and avoiding orphan singles, with break penalties for pairs, triples and
  five-card structures.
- **Protocol:** 1,000 four-player games per cell, evaluated seat randomised, **single seed, no
  intervals**. The author says small differences should be read with caution.
- **Numbers (win rate, mean score):** PPO with entropy 0.05 — vs random 90.1% (13.10), vs
  greedy **64.8% (5.40)**, vs smart **43.5% (2.00)**. Entropy 0: 58.2% vs greedy. MC-Q 50.2%,
  Q-learning 49.8%, SARSA 44.2% vs greedy. Current-policy self-play beat checkpoint self-play
  (61.6%) and smart-only training (55.4%).
- **Limits stated by the author:** no CFR, no search, no opponent modelling, no humans, no
  cross-play between independently trained agents, single configuration per ablation.
- **Code:** none linked.

### 2.5 Reading across the five

- Nobody has published a Big Two search agent with numbers we can check, so our ISMCTS
  results (`phase1-solver.md`) are the only public-style search baseline with intervals.
- Nobody has compared a learned agent with a search agent head-to-head. We have
  (`phase3-rl.md`, `long-training.md`): after 10 hours of CPU self-play the DMC network ties
  ISMCTS-1000 (−0.17 ± 0.82).
- The two Taiwanese works are the only ones from a Chinese-rules community, and the only ones
  with human play outside a lab. Their method (MCTS/ISMCTS plus opponent move prediction plus
  hand-made evaluation) is what our solver already is, plus the opponent model we measured as
  nearly worthless from history alone (`search-methods.md`).

## 3. Dou Dizhu and Guandan: where the methods come from

Dou Dizhu is three players (landlord vs two peasants), 17/20 cards, bombs and rockets that
beat anything, and a fixed "win/lose with multiplier" payoff. Guandan is four players in two
teams with two decks and a level card. Both are shedding games with the same combination
grammar as Big Two. Big Two differs in three ways that matter: four independent players, no
bombs (so the lead changes only by rank), and a card-count score. The methods still transfer;
the evaluation protocols do not.

### 3.1 The leaders, read in full

| System | Paper | Method in one line | Compute | Evaluation and result |
|---|---|---|---|---|
| DouZero (2021) | Zha et al., ICML 2021, <https://arxiv.org/abs/2106.06135>; code Apache-2.0 <https://github.com/kwai/DouZero> | Deep Monte Carlo: Q(state, action) with LSTM over history, Monte Carlo returns, many actors | "a single server with four GPUs", "days" (the LLM4CardGame paper quotes 30 days on 4 × 1080 Ti) | First on Botzone among 344 agents; pretrained ADP/WP/SL weights released |
| PerfectDou (2022) | Yang et al., NeurIPS 2022, <https://arxiv.org/abs/2203.16406>; weights + eval code Apache-2.0 <https://github.com/Netease-Games-AI-Lab-Guangzhou/PerfectDou> (training code withheld) | PPO + GAE with a critic that sees all hands; min-steps reward shaping; 621 abstract actions | 880 CPU cores + 8 GPUs, 2.5e9 steps (`prior-work.md`) | Beat DouZero (WP 0.543). Note: the community fork `godmoves/perfectdou` reports DouZero-ResNet beating the released PerfectDou model 52.08% / +0.124, so the released checkpoint is not the paper's strongest |
| OADMCDou (2024) | Luo, Tan, Zha, Zhang, IJCAI 2024, <https://www.ijcai.org/proceedings/2024/660> | **Oracle guiding** (train with opponents' cards visible, then fade the oracle out) + **adaptive DMC** (gradient weight clipping) | Small: "three days" per run on unstated hardware | vs DouZero ADP **+0.104 ± 0.041** (10,000 decks, 95% CI); vs DouZero+ +0.345, vs WagerWin +0.399 (both re-run by the authors on less compute than their papers); ADMC alone matched DouZero with lower loss |
| AlphaDou (2024) | Chang, arXiv:2407.10279, <https://arxiv.org/abs/2407.10279>; code GPL-3.0 <https://github.com/RuBP17/AlphaDou> | DMC with **two heads (win probability and expected points)**; keep moves whose Q is within 5% of the best, then pick the highest win probability; separate bidding networks | 4 Xeon Gold 6330 + **one RTX 4090**; duration not reported | vs DouZero (random bidding) WP 0.597, ADP 0.434 (4,000 decks, role-swapped); vs DouZero-ResNet WP 0.522. No PerfectDou comparison. Weights not clearly released |
| VRPO (2026) | Fan & Farina, arXiv:2605.19235, <https://arxiv.org/abs/2605.19235> | PPO with a centralised action-value critic and an Expected-SARSA(λ) trace instead of GAE (removes action-sampling variance); decisions split into steps to tame 27,472 actions | **~55 h on 4 × RTX 5090**, 2.65e9 timesteps, 40,000 iterations × 2,048 trajectories | Role-averaged gain vs PerfectDou **+0.152 ± 0.004**, vs DouZero +0.286 ± 0.004 (100,000 duplicate deals); passed PerfectDou at ~40% of its step budget. No code statement found |
| **Ataraxos (2026)** | Sokota, Vinitsky, Hu, Fan, Kolter, Farina, Nature 658, <https://www.nature.com/articles/s41586-026-11036-y>; arXiv:2511.07312 <https://arxiv.org/abs/2511.07312>; code MIT <https://github.com/AtaraxosAI/doudizhu> (no weights) | Damped self-play policy-value net (VRPO or IPPO) + **belief transformer** (3.3 M params, predicts the holder of each unseen card in rank order, NLL loss, trained on self-play snapshots) + **test-time search**: ⌈200/\|A\|⌉ sampled hands per option, full-length rollouts with the policy net, one regularised policy-improvement step (step size 5) | Dou Dizhu: policy **2 days on 4 H100**; belief **8 h on 4 H100**. Actor 1.9 M params (transformer depth 4, width 128) | vs PerfectDou **+0.199 ± 0.015**, vs DouZero +0.350 ± 0.016, vs its own network without search **+0.107 ± 0.012** (10,000 duplicate deals). Search helps the peasant side most (coordination). No belief ablation for Dou Dizhu; time per move not reported |

Smaller Dou Dizhu entries, abstract or README only: DouZero+ (opponent model = next player's
per-rank card counts from an LSTM; coach network discards lopsided deals; mainly helps
peasants; code at <https://github.com/submit-paper/Doudizhu>, no licence file seen;
<https://arxiv.org/abs/2204.02558>, full text read); DouRN (residual blocks on DouZero, "beats
DouZero", no numbers in the abstract; code <https://github.com/Yingchaol/Douzero_Resnet>;
<https://arxiv.org/abs/2403.14102>).

### 3.2 Guandan

| System | Paper | Method | Compute | Result |
|---|---|---|---|---|
| DanZero (2022) | Zhao et al., arXiv:2210.17087 | DMC, distributed | 80 actors × 2 cores, one RTX 3070, 30 days | 100% vs five rule bots, 82–90% vs the top three of the first Chinese Guandan AI competition |
| DanZero+ (2023, full text) | Zhao et al., arXiv:2312.02561 <https://arxiv.org/abs/2312.02561>; code <https://github.com/submit-paper/Danzero_plus> (no licence file, no weights, Docker image on Baidu) | **PPO choosing among the DMC model's top-k moves, k = 2**; k = 3 and 5 were worse (hyper-parameters tuned for k = 2) | 40 actors × 2 cores, "under a day" | 92.7% / 86.2% / 90.8% vs the top three bots (DMC: 90.1 / 82.6 / 87.5); 55.1% vs its own DMC teacher; "71 of 100" games vs graduate students (the paper's own counts are inconsistent) |
| GuanZero (2024, full text) | Yanggong, Pan, Wang, arXiv:2402.13582 <https://arxiv.org/abs/2402.13582> | DMC + **behaviour-regulating flags** (cooperating, dwarfing, assisting) as one-hot inputs; LSTM over the last 20 actions | "less than a week", hardware unstated | 82% vs the rule champion CGAIAC, 75–77% vs DouZero-style agents (1,000 mini-games, swapped); no DanZero comparison; no code |

The DanZero+ lesson is the sharpest: **plain PPO over thousands of legal moves was impractical;
PPO over the top two DMC candidates gained 2–4 points of win rate in under a day.** Our Phase 3
PPO failure fits that pattern.

### 3.3 LLM-based play (full text for the first)

- *Can large language models master complex card games?* (THUDM, arXiv:2509.01328 v5,
  <https://arxiv.org/abs/2509.01328>; code <https://github.com/THUDM/LLM4CardGame>): LoRA
  fine-tuning of Qwen2.5-7B, Llama-3.1-8B and GLM4-9B on 1 M teacher trajectories per game
  (DouZero for Dou Dizhu, DanZero for Guandan). Dou Dizhu win rates 0.81–0.88 (mixed-data
  models), Guandan 0.60–0.70; API models 0.02–0.35. 11–29 GPU-hours on 8 H800 per game. The
  text does not state the teacher's own score on the same opponents, so "approaches the
  teacher" is unquantified. Not on a phone; not a strength route.
- Guandan theory-of-mind prompting (Yim et al., arXiv:2408.02559) and RL+LLM commentary for
  Guandan (Tao et al., arXiv:2406.17807): abstracts only. The commentary paper is the one
  relevant precedent for "reasons" generated beside an RL agent, but it generates text, which
  D42 forbids in the engine. Keys come first; an LLM could rewrite keys into prose on a server
  later.

## 4. Other shedding games and general methods

### 4.1 Tichu, President (Daihinmin), Pusoy Dos

- **Tichu:** two student theses only. Müller (ETH 2020, 29 pages): imitation learning + PPO
  with Tichu-specific tricks, "a decent human level", sample efficiency is the problem. Wyss
  (Bern 2024): DQN vs PPO, PPO with behavioural-cloning pre-training reaches amateur level;
  cites an earlier mini-Tichu MCTS. Both PDFs were binary-only to our tools; **search-snippet
  only.** Links: <https://pub.tik.ee.ethz.ch/students/2020-FS/SA-2020-19.pdf>,
  <https://prg.inf.unibe.ch/wp-content/uploads/2024/08/BA_NicolasWyss.pdf>.
- **President / 大貧民:** the UEC Computer Daihinmin Competition (UECda) has run since 2006
  (IPSJ reports 58482, 58498); strong entries use Monte Carlo methods, and Wakatsuki 2020
  reports "new applications of MCTS to computer Daihinmin" (Int. J. Smart Computing and AI
  4(1):18–35; metadata only, the competition site did not resolve). This is the longest-running
  benchmark for a Big-Two-like game, but its rules (8-cut, revolution, card exchange) differ
  enough that its agents are not opponents for us.
- **Pusoy Dos:** no AI publication found (two searches). It is Big Two with Philippine
  ordering; our engine covers it with a rule variant if the app ever wants it.

### 4.2 General methods that transfer

| Method | Source | What it gives us | Caveat |
|---|---|---|---|
| ISMCTS | Cowling, Powley, Whitehouse, IEEE TCIAIG 2012 (our solver) | A training-free baseline that produces per-move values with intervals | Strategy fusion; rollout policy is the opponent model |
| Re-determinizing ISMCTS | Goodman, arXiv:1902.06075 <https://arxiv.org/abs/1902.06075> (full abstract) | Stops hidden information leaking into opponent nodes; won the 2018 Hanabi competition at 40 ms per move with a learned leaf evaluator | Hanabi is cooperative; the leak matters most when opponents' nodes see our cards |
| Expert iteration | Anthony, Tian, Barber, NeurIPS 2017 <https://arxiv.org/abs/1705.08439> | Search labels → network → stronger search, repeated. Our distillation is one round of it | Needs the network inside the search to compound; we have not done that yet |
| Sampling information sets for value targets | Bertram, Fürnkranz, Müller, KI 2024 <https://arxiv.org/abs/2407.05876> | For a fixed budget of perfect-information evaluations, ~3 sampled deals per position across many positions beat many samples on few positions | Measured on Reconnaissance Blind Chess |
| Belief networks | Ataraxos (§3.1); DouZero+; Luo & Tan | A learned sampler for hidden hands, used by search and by hints | Our measurement says history reveals little in Big Two |
| ReBeL / Student of Games | Brown et al. 2020 <https://arxiv.org/abs/2007.13544>; Schmid et al. 2023 <https://arxiv.org/abs/2112.03178> | Public-belief-state search with equilibrium guarantees | Two-player zero-sum only; a four-player belief over 39 cards in three hands is too large for their value nets. Not a route for us (unchanged from `prior-work.md`) |
| Pluribus-style multi-player search | Brown & Sandholm, Science 2019 | Shows depth-limited search with a few continuation strategies works empirically for 6 players | No guarantee; poker's range structure does not carry to card-count games |
| Max^n and multi-player UCT | Luckhardt & Irani 1986; our `endgame` and tree policy | Exact endgames per sampled deal; each player maximises own score | Our Phase 1 found no gain vs greedy opponents; untested vs strong ones |
| DanZero+ top-k, ODMC min-combination, PerfectDou min-steps | §3 | Three ways to shrink the move list to "moves consistent with a good hand decomposition" | Big Two has no bombs, so the decomposition is cleaner than in Dou Dizhu |
| WebAssembly inference | `tract` (Sonos, Rust, ONNX/NNEF, wasm32 target, Apache-2.0 / MIT) <https://github.com/sonos/tract>; or our own `crates/nn` (`.pknn`) | Runs a small network in the browser worker with no Python | A transformer at 2 M params is fine per call; 8,000 calls per move is not (§5) |

## 5. Comparison table: fit for our hardware and for a phone

Hardware: RTX 4060 Ti 16 GB, i7-12700K (20 threads), 32 GB, local only. Phone: WebAssembly,
one thread, 0.5–2 s per hint, `bot_action` budgets lower. "Own estimate" means I scaled the
paper's compute by a rough ratio: one H100 ≈ 4–6 × a 4060 Ti on small transformers (own
estimate, not measured); a 5090 ≈ 2.5–3 ×.

| Method (year) | Game | Compute (training / play) | Evaluation | Strength evidence | Open code / weights / licence | Fit: our box | Fit: phone |
|---|---|---|---|---|---|---|---|
| Charlesworth PPO (2018) | Big Two | 4 cores + GPU, 2 days / 1 ms | 664 human games, 10k-game self-eval | humans −0.96 ± 0.38 vs it | code + TF1 checkpoint; no licence file | yes (hours) | yes (MLP); needs a port |
| Big2AI (2022), Big2MDP (2025) | Big Two | not stated / Android real time | prototype vs bots and humans, no intervals | "highest win rate" (abstract) | none | re-implementable from the descriptions | yes by design (search on a phone) |
| Luo & Tan ODMC (2024) | Big Two, Dou Dizhu | "less than DMC" / 1 ms | abstract only | "comparable to DMC" | none | yes | yes (network) |
| Patwa PPO (2026) | Big Two | 6-core laptop, 7–13 h / 1 ms | 1,000 games × 3 scripted bots, one seed | 64.8% vs greedy, 43.5% vs smart | none; bots re-implementable | yes (hours) | yes (small net) |
| Our ISMCTS-1000 (2026) | Big Two | none / 14 ms on M1 | 100–300 duplicate deals, CIs, head-to-head | +8.6 vs greedy; ties DMC-10h | ours (Rust) | — | yes: 5,000 sims ≈ 0.5 s (app estimate), `Instant` fix pending |
| Our distilled net (2026) | Big Two | 4 h CPU + 1 h solver data / 1 ms | same | −0.47 ± 0.76 vs ISMCTS | ours; PyTorch → `.pknn` export exists for the value net | — | yes |
| DouZero DMC (2021) | Dou Dizhu | 4 GPUs, days–weeks / 1 ms | Botzone + head-to-head | first of 344 | Apache-2.0, weights | yes at reduced scale (weeks) | yes (MLP + LSTM) |
| PerfectDou (2022) | Dou Dizhu | 880 cores + 8 GPUs / 1 ms | 10k decks role-swap | 0.543 WP vs DouZero | Apache-2.0 weights + eval; no training code | the recipe yes; the scale no | yes |
| OADMCDou (2024) | Dou Dizhu | "three days", unstated / 1 ms | 10k decks, 95% CI | +0.104 ± 0.041 vs DouZero | none found | yes | yes |
| AlphaDou (2024) | Dou Dizhu | 4 Xeon + 1 RTX 4090 / 1 ms | 4k decks role-swap | WP 0.597 vs DouZero | GPL-3.0 code (licence blocks shipping; reading is fine per decisions) | yes (same class of GPU) | yes |
| DanZero+ (2023) | Guandan | 80 actors 30 days + 40 actors < 1 day / 1 ms | 1,000 games vs 8 bots; 100 vs students | 92.7% vs best bot; 71% vs students | code, no licence, no weights | DMC stage: months on CPU; PPO stage: yes | yes |
| GuanZero (2024) | Guandan | "< 1 week" / 1 ms | 1,000 games swapped | 75–82% | none | yes | yes |
| VRPO (2026) | Dou Dizhu, HUNL | 4 × 5090, 55 h / 1 ms | 100k duplicate deals | +0.152 vs PerfectDou | none found | ~2–4 weeks on one 4060 Ti (own estimate); Big Two is smaller than Dou Dizhu in actions, so less | yes (1.9 M transformer) |
| Ataraxos (2026) | Dou Dizhu (+Stratego, Hanabi) | 4 H100 × 2 d + 4 H100 × 8 h / 200 rollouts per move on a GPU | 10k duplicate deals | +0.199 vs PerfectDou; +0.107 from search | MIT code, no weights | ~1–2 weeks policy + 1–2 days belief (own estimate) | **no** as published: 200 rollouts × ~40 plies × a 2 M-param transformer ≈ 8,000 forward passes per move, ≈ 8–30 s on one wasm thread (own estimate). Needs 20–50 rollouts and a smaller net, or a server |
| LLM4CardGame (2025) | Dou Dizhu, Guandan | 8 H800, 11–29 h / seconds per move on a GPU | win rate vs unstated opponents | 0.85 (DouDizhu) | code; data unclear | fine-tuning a 7 B model fits 16 GB with LoRA | no |
| ReBeL / SoG | poker etc. | hundreds of GPUs | — | — | withheld / partial | no | no |

## 6. What "state of the art for Big Two" can mean, and the bars we propose

Published Big Two evaluation is: three scripted bots and one seed (Patwa), seven amateurs
(Charlesworth), or an Android prototype with no numbers (Chen & Lu). Nobody has run a
duplicate match with intervals against another published agent. So "state of the art" can
mean four different things, and we should state which one we claim.

**Meaning A — beats every published Big Two agent under its own protocol.** Feasible for two of
five. The other three have no code and no numbers.

**Meaning B — method parity with the Dou Dizhu leaders**, measured with their protocol
(duplicate deals, role or seat rotation, 10,000+ deals, standard errors). This is the claim
Ataraxos makes; we can make it only against our own re-implementations.

**Meaning C — strongest in a ladder we control, with an exploitability proxy.** No published
Big Two work has this; it is the honest internal yardstick.

**Meaning D — human evidence from the app**, the only meaning players care about.

Proposed bars (all with 95% intervals over deals; "duplicate" = every deal played with the
lineup rotated through all four seats, as in `phase1-solver.md`):

| Bar | What we run | Pass condition | Cost |
|---|---|---|---|
| **B1 Patwa's bots, Patwa's scoring** | Add `scoring=sum_of_others` and Patwa's *greedy* and *smart* bots as agents (his definitions are in §2.4); 10,000 games per cell, seat randomised, 3 seeds | Win rate vs 3 greedy > 64.8% and vs 3 smart > 43.5%, interval excluding his number; report mean score too | 1–2 days to implement the bots and scoring; evaluation minutes |
| **B2 Charlesworth's network as an opponent** | Port his checkpoint (412 features, 1,695-action head) to ONNX or a NumPy loader; play under his rules (four-card plays allowed, his scoring); 1,000 duplicate deals, our bot in one seat vs three copies, and two-vs-two | Our bot's mean score > 0 with interval excluding 0 in both line-ups; report win rate (his agent took 22.5% of games against humans) | 2–3 days (rule variant for four-card plays, feature re-implementation, checkpoint conversion); the licence must be checked before use, and we only run it, never ship it |
| **B3 Internal ladder with an exploitability proxy** | Under our tiered scoring: (a) head-to-head vs ISMCTS-1000 and vs the DMC-10h network, 1,000 duplicate deals; (b) *best-response probe*: train a DMC agent for 10 GPU-hours against three frozen copies of the candidate, report its gain | (a) ≥ +1.0 point per game vs ISMCTS-1000, interval excluding 0; (b) the probe gains less against the candidate than against ISMCTS-1000 and the DMC-10h network (lower = harder to exploit) | (a) minutes; (b) ~10 GPU-hours per candidate |
| **B4 Ataraxos-class method bar** | Re-implement the stack at our scale: damped self-play net (VRPO-style critic), belief net, 200-rollout search; compare network-only vs network+search vs ISMCTS-1000, 10,000 duplicate deals | Search adds a measurable gain over its own network (Ataraxos: +0.107 in Dou Dizhu); the stack beats ISMCTS-1000 by more than B3(a) | 2–4 weeks of GPU time plus the engineering; this is the research core of a later phase |
| **B5 Humans in the app** | Log every game; after 500 human games at the expert level, compute the humans' mean score with a standard error, as Charlesworth did | Humans average below 0 against the expert bot with the interval excluding 0 | free once the app ships |
| **B6 Phone budget** | On a mid-range Android phone, one wasm thread: agreement with a 100,000-simulation reference and value lost per decision, at 0.5 s and 2 s | At 2 s: agreement ≥ the laptop ISMCTS-1000 figure (69.7%, `phase3-rl.md`) and mean value lost ≤ 0.3 points | 1 day once the wasm build exists |

Recommendation: claim **A (B1 + B2) and C (B3)** for the first release and treat **B4** as the
research target. B4 is what "at least the state of the art" honestly means in 2026, because
the Dou Dizhu leaders are the only credible yardstick for the method. B5 is the one that
matters to the product.

## 7. Open agents we could run as opponents (checked live, 2026-10-28)

| Agent | Live? | Usable how | Licence |
|---|---|---|---|
| Charlesworth big2_PPOalgorithm | yes; checkpoint present | opponent under his rules (B2) | no licence file seen; ask or treat as read-only |
| Patwa 2026 | no code | re-implement his three bots and his scoring (B1) | n/a |
| DouZero, PerfectDou, AlphaDou, DouZero+, DanZero+, Ataraxos code | yes | Dou Dizhu only; useful as reference implementations of DMC, full-info critic, two-head Q, top-k PPO, belief net and damped self-play. Never linked or shipped (holdem decisions, carried over) | Apache-2.0 / Apache-2.0 / GPL-3.0 / none / none / MIT |
| RLCard | yes, MIT | Dou Dizhu rule agent and environments; nothing for Big Two | MIT |

## 8. Where the claims are weak

- **Three of five Big Two papers are abstract-only to us** (Chen & Lu 2022 and 2025, Luo &
  Tan 2024). Their "beats existing AI" claims have no numbers we can see. If Brian has IEEE
  access, the 2025 paper's "free-playing right" decision rule is worth reading before we
  design our lead-control features (`hard-problems.md` §5).
- **Patwa's results are one seed, 1,000 games, scripted opponents.** His own table shows a
  6.6-point swing from an entropy setting; cross-seed noise could be of the same size.
- **Charlesworth's rules differ from ours** (four-card plays; scoring). His human study is the
  only Big Two human evidence, and it is 7 amateurs.
- **The released PerfectDou checkpoint loses to DouZero-ResNet** in a community evaluation, so
  "beats PerfectDou" claims depend on which checkpoint was used; VRPO and Ataraxos do not
  say.
- **Ataraxos' Dou Dizhu numbers are its own**, with no belief ablation for that game and no
  per-move time. Its code has no weights, so reproduction means retraining.
- **Compute is unstated** for OADMCDou, GuanZero, Chen & Lu, Luo & Tan and AlphaDou's duration.
- **Our own numbers were measured with last-card duty off** and on an M1; the app uses duty on.
  Every bar in §6 must be re-run under the current defaults.

## Sources

- Charlesworth, H. *Application of self-play reinforcement learning to a four-player game of
  imperfect information.* arXiv:1808.10442, 2018. <https://arxiv.org/abs/1808.10442>; code
  <https://github.com/henrycharlesworth/big2_PPOalgorithm>
- Chen, L.-W., Lu, Y. *Challenging artificial intelligence with multi-opponent and
  multi-movement prediction for the card game Big2.* IEEE Access 10, 2022.
  DOI 10.1109/ACCESS.2022.3166932 (abstract via Semantic Scholar)
- Chen, L.-W., Lu, Y. *Markov decision process-based artificial intelligence with card-playing
  strategy and free-playing right exploration for four-player card game Big2.* IEEE Trans.
  Games 17(2), 2025. DOI 10.1109/TG.2024.3424431 (abstract via Semantic Scholar)
- Luo, Q., Tan, T.-P. *Improved learning efficiency of deep Monte-Carlo for complex
  imperfect-information card games.* Applied Soft Computing 158, 2024.
  <https://www.sciencedirect.com/science/article/abs/pii/S1568494624003193>
- Luo, Q., Tan, T.-P., Zha, D., Zhang, T. *Enhanced DouDiZhu card game strategy using oracle
  guiding and adaptive deep Monte Carlo method.* IJCAI 2024.
  <https://www.ijcai.org/proceedings/2024/660>
- Patwa, A. *Self-play reinforcement learning under imperfect information in Big 2.*
  arXiv:2605.28863, 2026. <https://arxiv.org/abs/2605.28863>
- Zha, D. et al. *DouZero.* ICML 2021. <https://arxiv.org/abs/2106.06135>;
  <https://github.com/kwai/DouZero>
- Zhao, Y. et al. *DouZero+.* arXiv:2204.02558. <https://arxiv.org/abs/2204.02558>;
  <https://github.com/submit-paper/Doudizhu>
- Yang, G. et al. *PerfectDou.* NeurIPS 2022. <https://arxiv.org/abs/2203.16406>;
  <https://github.com/Netease-Games-AI-Lab-Guangzhou/PerfectDou>;
  community evaluation <https://github.com/godmoves/perfectdou>
- Chang, C. *AlphaDou.* arXiv:2407.10279. <https://arxiv.org/abs/2407.10279>;
  <https://github.com/RuBP17/AlphaDou>
- Chen, Y., Lyu, Y., Zhang, D. *DouRN.* arXiv:2403.14102. <https://arxiv.org/abs/2403.14102>
- Fan, Z., Farina, G. *GAE falls short in imperfect-information self-play reinforcement
  learning* (VRPO). arXiv:2605.19235, 2026. <https://arxiv.org/abs/2605.19235>
- Sokota, S., Vinitsky, E., Hu, H., Fan, Z., Kolter, J. Z., Farina, G. *Scalable
  decision-making for games of imperfect information* (Ataraxos). Nature 658, 2026;
  arXiv:2511.07312. <https://arxiv.org/abs/2511.07312>; <https://github.com/AtaraxosAI/doudizhu>
- Zhao, Y. et al. *DanZero.* arXiv:2210.17087; *DanZero+.* arXiv:2312.02561.
  <https://arxiv.org/abs/2312.02561>; <https://github.com/submit-paper/Danzero_plus>
- Yanggong, Y., Pan, H., Wang, L. *Mastering the game of Guandan with deep reinforcement
  learning and behavior regulating* (GuanZero). arXiv:2402.13582.
  <https://arxiv.org/abs/2402.13582>
- THUDM. *Can large language models master complex card games?* arXiv:2509.01328, 2025.
  <https://arxiv.org/abs/2509.01328>; <https://github.com/THUDM/LLM4CardGame>
- Yim, Y. et al. *Evaluating and enhancing LLM agents based on theory of mind in Guandan.*
  arXiv:2408.02559. Tao, M. et al. *Enhancing commentary strategies for imperfect-information
  card games.* arXiv:2406.17807. (abstracts)
- Cowling, P., Powley, E., Whitehouse, D. *Information set Monte Carlo tree search.* IEEE
  TCIAIG 4(2), 2012. Whitehouse, D. *Monte Carlo tree search for games with hidden information
  and uncertainty.* PhD thesis, York, 2014.
  <https://etheses.whiterose.ac.uk/id/eprint/8117/1/Feb%2016%20-%20FINAL.pdf>
- Goodman, J. *Re-determinizing information set Monte Carlo tree search in Hanabi.*
  arXiv:1902.06075. <https://arxiv.org/abs/1902.06075>
- Anthony, T., Tian, Z., Barber, D. *Thinking fast and slow with deep learning and tree
  search.* NeurIPS 2017. <https://arxiv.org/abs/1705.08439>
- Bertram, T., Fürnkranz, J., Müller, M. *Efficiently training neural networks for imperfect
  information games by sampling information sets.* KI 2024. <https://arxiv.org/abs/2407.05876>
- Brown, N. et al. *ReBeL.* arXiv:2007.13544. Schmid, M. et al. *Student of Games.*
  arXiv:2112.03178. Brown, N., Sandholm, T. *Superhuman AI for multiplayer poker.* Science 2019.
- Müller, P. *Tichu bot.* ETH Zurich semester thesis, 2020.
  <https://pub.tik.ee.ethz.ch/students/2020-FS/SA-2020-19.pdf> (snippet only). Wyss, N.
  *Reinforcement learning for Tichu.* Bern BA thesis, 2024.
  <https://prg.inf.unibe.ch/wp-content/uploads/2024/08/BA_NicolasWyss.pdf> (snippet only)
- Wakatsuki, M. *New applications of the Monte-Carlo tree search to computer Daihinmin.* Int.
  J. Smart Computing and AI 4(1), 2020 (metadata only). UECda reports: IPSJ records 58482,
  58498.
- tract: <https://github.com/sonos/tract>. RLCard: <https://github.com/datamllab/rlcard>.
- Our own: `big2\docs\study-results\phase1-solver.md`, `phase3-rl.md`, `search-methods.md`,
  `long-training.md`; `big2\docs\research\prior-work.md`.

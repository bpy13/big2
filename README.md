# Big Two RL Lab

An interactive four-player Big Two game with approximate self-play strategy hints and
checkpointable regret-minimization training. Training is the focus: save a strategy,
resume it later, and compare self-play behavior over training iterations.

## Rules

Four players receive 13 cards each. Ranks run from 3 up to Ace, then 2; suits are
diamonds < clubs < hearts < spades. The holder of 3♦ opens and must include it in the
first play. Singles, pairs, triples, and five-card combinations are legal. Five-card
categories rank straight < flush < full house < four of a kind < straight flush.
Straights use consecutive ranks from 3 through Ace; twos do not appear in straights,
and Aces are high. Players may pass; after three consecutive passes, the last player
to play leads the next trick. A deal ends as soon as a player empties their hand.

The default score multiplies remaining cards by tiers: 1–7 cards ×1, 8–9 ×2, 10–12
×3, and 13 ×4. The alternate uses 1–9 ×1, 10–12 ×2, and 13 ×3. Each pair settles
the difference in penalty: the higher-penalty player pays the lower-penalty player.
Pairwise transfers sum to zero.

## Setup

```sh
conda env create -f environment.yml
conda activate big2
python -m pip install -e ".[dev]"
streamlit run app.py
```

**Train** is the first tab. Choose a scoring mode, run outcome-sampling MCCFR batches,
and inspect information-set coverage and sampled self-play diagnostics. Use **Save
checkpoint** to write the current regrets, average strategy, iteration count, and RNG
state to `models/`. Use **Load saved checkpoint** to resume, or download/upload the JSON
file to move it between machines. The `models/` directory is ignored by Git. Set
`BIG2_MODEL_DIR` to change the local checkpoint directory.

In **Play**, choose a seat and play against three bots sampling from the learned
average strategy. Bots play until your turn; the hint shows its most likely legal plays
and probabilities. Unvisited information sets use a uniform fallback strategy.

## Learning method and limits

Big Two is a four-player hidden-information game. The app uses OpenSpiel's
outcome-sampling Monte Carlo Counterfactual Regret Minimization (MCCFR), which learns
regrets and average action probabilities from sampled traversals. The hints are
approximate strategies, **not** exact ground truth or guaranteed Nash equilibria.
Self-play win rates and score transfers are empirical diagnostics, not exploitability
bounds.
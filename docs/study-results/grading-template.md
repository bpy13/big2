# Grade lines, the trivial flag and a performance measure (plan Task 43)

Status 2026-11-28: built on Task 42's labels (7 bands x 300 same-rung deals, every decision
graded by PUCT-2,000 with the belief network, decision 0010), Task 40's ladder games (38,400
games on 9,600 deals) and 4,000 new games of the network-plus-search expert (queue job 442). The
proposal is `runs/p6_grading/proposal.json`; the question to Brian is in `docs/inbox.md`. Code:
`python/big2/teach/grading/lines.py`, `accuracy.py`, `rating.py` (game-agnostic), driver
`experiments/grading_rating.py` (+ `grading_rating_job.ps1`), report `experiments/grading_report.py`
(writes `runs/p6_grading/report.md`, `tables.json`, `proposal.json`), tests
`python/tests/test_grading_lines.py` (8).

Brian, 2026-11-27: "re-adjust what moves should qualify as good or inaccurate or bad or etc." and
"we should have a better representation of the performance rather than using how often the
best/good is played, as a lot of the moves are trivial". Decision 0008 (amended): the rating is
points won with the luck subtracted, on a chess-like 0 to 3000+ scale; grade lines for single
moves stay EV loss against the expert.

## 1. Method

- **Reference.** A decision's loss is the EV of the reference's best move minus the EV of the
  move played, in points, from PUCT-2,000 (`c_puct` 2.5, belief network, exact endgame at 12 or
  fewer hidden cards); `docs/study-results/mistakes.md` §1. A deal's score has SD 27.6 points
  per seat (ladder games), so 0.25 SD, the "decides the deal" mark, is 6.9 points.
- **Trivial** (`lines.trivial`): a decision is *forced* (one legal move, usually a pass) or
  *flat* (Task 42's rule: every evaluated move's loss is under 0.2 points or its 95 % range covers
  0). Trivial decisions get no grade word and count in no score.
- **Lines** (`lines.derive_lines`): upper limits of great / good / inaccuracy / mistake, beyond
  the last a blunder. Rule: great = the reference's noise (median 95 % half-width of a paired
  loss, at least 0.2); good = the loss the top rung (`net`, the raw hour-24 network) exceeds on
  10 % of its graded decisions; inaccuracy = the loss it exceeds on 3 %; mistake = the loss the
  app's `medium` rung exceeds on 3 %. Values are rounded to the nearest half point; a line under
  1.5x the one below is lifted to 1.5x.
- **APL** (`accuracy.apl`): mean points lost per graded decision. **Accuracy**: mean over graded
  decisions of `100 * exp(-loss / s)`, `s` such that a loss on the inaccuracy line scores 50.
  **Rating from APL**: least squares of the rungs' luck-removed rating against `ln(APL)`.
- **Luck-removed rating** (`rating.py`): the luck of a deal is the score each seat gets when the
  hour-24 network plays all four seats of that deal, averaged over `m` sampled games at
  temperature 1 (`m = 16` here; one top-move game for `m = 0`). A seat's *margin* is its score
  minus its luck. Strengths `s` (points a deal) are the least-squares fit of margin =
  `s_seat - mean(s at the table)` (additive, zero-sum). Rating = `3000 * (s - s_random) /
  (s_net - s_random)`: random play is 0, the raw network 3000, and the expert with search lands
  wherever its margin puts it. 95 % ranges by bootstrap over deals. The luck term depends on the
  deal alone, so every player at the table is corrected by the same rule (no cap at the expert).

## 2. Trivial decisions

| band (rung Elo) | decisions | forced | flat | trivial | graded |
|---|---:|---:|---:|---:|---:|
| net-t4.34 (1096) | 16,952 | 47.1 % | 1.7 % | 48.7 % | 8,689 |
| easy (1161) | 13,159 | 46.4 % | 1.5 % | 47.9 % | 6,858 |
| medium (1200) | 13,844 | 43.0 % | 1.7 % | 44.6 % | 7,663 |
| hard (1222) | 13,096 | 41.5 % | 1.8 % | 43.2 % | 7,435 |
| net-t1 (1230) | 13,472 | 47.3 % | 1.5 % | 48.9 % | 6,889 |
| expert (1236) | 12,543 | 43.6 % | 1.3 % | 44.8 % | 6,922 |
| net (1248) | 13,138 | 46.1 % | 1.6 % | 47.7 % | 6,875 |

Almost half of all decisions are trivial, nearly all of them forced passes. The rule is Task 42's
(default 1 of the task); nothing argued for a change: flat decisions are only 1.3-1.8 %, and the
proposed great line (0.5) already absorbs small differences on the graded ones.

## 3. Lines

Noise: the median 95 % half-width of a graded, non-best move's loss is 0.483 points (0.44-0.53 by
band). Loss exceeded on a share of each band's graded decisions:

| band | 10 % | 5 % | 3 % | 1 % |
|---|---:|---:|---:|---:|
| net-t4.34 | 5.67 | 8.72 | 11.40 | 18.49 |
| easy | 4.39 | 7.11 | 9.84 | 19.70 |
| medium | 4.29 | 7.18 | **9.73** | 15.74 |
| hard | 4.01 | 6.70 | 9.21 | 14.82 |
| net-t1 | 2.92 | 5.44 | 7.75 | 13.49 |
| expert | 3.45 | 5.44 | 7.02 | 11.40 |
| net | **2.72** | 5.08 | **6.99** | 11.96 |

The rule gives 0.48 / 2.72 / 6.99 / 9.73. Rounded to the half point: 0.5 / 2.5 / 7.0 / 9.5; the
mistake line is under 1.5 x 7.0, so it is lifted to **10.5**.

**Proposed lines: great <= 0.5, good <= 2.5, inaccuracy <= 7.0, mistake <= 10.5, blunder above**
(D102: 0.2 / 0.69 / 2.75 / 6.86). Reasons: 0.5 is the reference's own noise, so "great" means
"not distinguishable from the best"; 2.5 is what the strongest bot without search exceeds on
one graded decision in nine; 7.0 is both the top rung's 3 % tail and the old "decides the deal"
mark (0.25 SD of a deal's score, D102's blunder line); 10.5 keeps the 1.5x spacing (0.38 SD of a
deal). Compared with D102 every line but the first moves up by 2-4x: under the new reference even
the raw network loses more than D102's 2.75 on 10 % of graded decisions, so D102's words were
calling a tenth of the expert's own moves mistakes.

Word shares over graded decisions, D102 lines:

| band | great | good | inaccuracy | mistake | blunder |
|---|---:|---:|---:|---:|---:|
| net-t4.34 | 48.5 % | 9.3 % | 20.6 % | 14.1 % | 7.4 % |
| easy | 60.0 % | 8.9 % | 15.6 % | 10.2 % | 5.3 % |
| medium | 54.7 % | 9.2 % | 19.7 % | 11.0 % | 5.4 % |
| hard | 59.2 % | 8.9 % | 17.1 % | 10.0 % | 4.9 % |
| net-t1 | 65.8 % | 8.5 % | 15.1 % | 7.0 % | 3.6 % |
| expert | 64.8 % | 7.6 % | 15.1 % | 9.3 % | 3.2 % |
| net | 69.6 % | 7.5 % | 13.0 % | 6.8 % | 3.1 % |

Proposed lines:

| band | great | good | inaccuracy | mistake | blunder |
|---|---:|---:|---:|---:|---:|
| net-t4.34 | 54.6 % | 22.2 % | 16.0 % | 3.5 % | 3.6 % |
| easy | 66.0 % | 17.0 % | 11.8 % | 2.4 % | 2.7 % |
| medium | 60.8 % | 21.0 % | 12.9 % | 2.8 % | 2.5 % |
| hard | 65.1 % | 18.7 % | 11.5 % | 2.5 % | 2.1 % |
| net-t1 | 71.4 % | 17.1 % | 7.9 % | 1.7 % | 1.8 % |
| expert | 69.9 % | 16.5 % | 10.5 % | 1.7 % | 1.3 % |
| net | 74.9 % | 14.2 % | 8.0 % | 1.6 % | 1.4 % |

Repeatability (Task 42's relabels of 2,522 graded decisions): the same word on a new seed 84.6 %
under D102, **88.1 %** under the proposed lines; under the rules-only belief 82.2 % and 87.5 %.
Most remaining changes are moves that were best on one seed and lost a little on the other.

## 4. Accuracy, APL and today's best-or-good share

`s` = 7.0 / ln 2 = 10.1, so a move's accuracy is 100 at loss 0, 78 at the good line, 50 at the
inaccuracy line, 35 at the mistake line, 14 at a 20-point blunder. A game is one seat of one deal.

| band | rung Elo (Task 40) | luck-removed rating | graded decisions | APL | per-game APL SD | accuracy | per-game SD | best-or-good share (loss <= 2.5) | games for +-100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| net-t4.34 | 1096 | 1557 | 8,689 | 1.919 | 1.98 | 87.2 | 9.4 | 76.9 % | 1,350 |
| easy | 1161 | 1989 | 6,858 | 1.489 | 2.23 | 90.1 | 9.5 | 83.0 % | 2,857 |
| medium | 1200 | 2305 | 7,663 | 1.467 | 1.46 | 90.5 | 8.5 | 81.8 % | 1,259 |
| hard | 1222 | 2425 | 7,435 | 1.326 | 1.61 | 91.5 | 8.7 | 83.9 % | 1,867 |
| net-t1 | 1230 | 2871 | 6,889 | 1.004 | 1.31 | 93.5 | 7.5 | 88.5 % | 2,160 |
| expert | 1236 | 2712 | 6,922 | 1.017 | 1.21 | 93.0 | 7.7 | 86.4 % | 1,806 |
| net | 1248 | 3000 | 6,875 | 0.886 | 1.13 | 94.2 | 7.0 | 89.0 % | 2,068 |

- **Rating from APL**: rating = 2828 - 1818 x ln(APL), RMS residual 103 rating points over the
  seven rungs. The per-game spread of APL is larger than its mean, so an APL-based estimate
  needs 1,300-2,900 games for +-100 (95 %); about 150-300 games for +-300.
- **Ordering.** Spearman rank correlation with the rungs' order, over the seven rungs: APL and
  accuracy 0.964 against Task 40's per-deal Elo and 1.000 against the luck-removed rating;
  today's best-or-good share 0.929 and 0.964. Over single games (one seat of one deal, 28,000
  games, against the seat's rung rating): APL and accuracy 0.295, best-or-good share 0.247. APL
  and accuracy order the rungs perfectly and single games a little better than the share does;
  a single game is a noisy measure under any of them.
- The APL-based estimate caps at the expert (its APL is 0 by construction), which is why it is a
  secondary "performance" number, not the rating (decision 0008 amended).

## 5. Luck-removed rating

Luck = mean score of each seat over 16 sampled games of the hour-24 network playing all four
seats (26 ms a game; 12,700 deals in 9 minutes on 8 threads, queue job 443). Games: Task 40's
38,400 ladder games plus 4,000 new games of `puct2000` (PUCT-2,000 with the belief network,
Task 42's reference, standing in for Task 39's unpicked budget) against `net` and `expert`,
ABAB, 500 duplicate deals each (job 442). 42,400 games on 10,600 deals.

| bot | strength (points a deal) | **rating** | 95 % range | half A | half B |
|---|---:|---:|---:|---:|---:|
| random | -20.74 | **0** | (anchor) | -62 | 62 |
| net-t7.07 (ladder probe) | -9.48 | 1149 | [1049, 1240] | 1092 | 1205 |
| net-t4.34 | -5.48 | 1557 | [1470, 1637] | 1518 | 1596 |
| greedy (`easy`) | -1.24 | **1989** | [1936, 2046] | 2016 | 1962 |
| net-t2.66 (ladder probe) | +1.02 | 2220 | [2088, 2352] | 2142 | 2298 |
| Q-network (`medium`) | +1.85 | **2305** | [2252, 2360] | 2355 | 2254 |
| ISMCTS-500 (`hard`) | +3.03 | **2425** | [2346, 2518] | 2518 | 2331 |
| ISMCTS-10,000 (old `expert`) | +5.85 | **2712** | [2661, 2767] | 2743 | 2682 |
| net-t1 | +7.41 | 2871 | [2815, 2925] | 2860 | 2883 |
| hour-24 network, top move (`net`) | +8.67 | **3000** | (anchor) | 3025 | 2975 |
| network + PUCT-2,000 (new expert) | +9.13 | **3047** | [2981, 3111] | 3066 | 3029 |

`c` = 102 rating points per point of margin a deal (the 29.4-point gap between random and the
network spans 3000). Where the six bots sit: random 0, easy 1989, medium 2305, hard 2425, old
expert 2712, raw network 3000; the new expert 3047 (it beats the raw network by 0.27 +-0.19 a
deal in direct play, Task 52 measured +0.34). `net-t1` (2871) rates above the old `expert`
(2712) on points although Task 40's per-deal Elo had them level (1230 vs 1236): the two measures
differ (points vs who finishes ahead), and the direct pairing is -0.95 +-0.21 a deal for `expert`.

**Unbiased.** (1) The luck term is a function of the deal alone. In a duplicate design (every name
in every seat of the deal once) the luck terms of one name sum to exactly zero over the deal, so
the fitted strengths with luck removed equal the raw-points fit to the last digit (checked:
`m = 0`, `m = 4` and `m = 16` give the same ratings); only the per-game noise changes. (2) Equal
players come out equal: each name split into halves A and B by rotation parity rates the same
within noise (columns above; the widest gap, `hard`, is 187 points on 800+800 games, 1.5 SE).
(3) Same-rung tables (Task 42's deals, four equal seats): the mean margin by seat is within
+-2.7 points of 0 everywhere, inside 2 SE except `easy` seats 0/1 (-2.65 / +2.56, SE 1.3). (4) A
player stronger than the network rates above it: `puct2000` at 3047.

**Variance cut.** Over every seat of every game, score SD 27.6 -> margin SD 22.2, a 35.3 % cut
of the variance; the duplicate leave-one-out baseline (same hand, the deal's other three games)
cuts 32.9 %. One sampled self-play game varies with SD 16.0 around the deal's expectation, so the
cut is 3.6 % with one top-move game, 29.0 % with 4 samples, 35.3 % with 16, and at most 37.4 %
with the exact expectation: **the cards decide a bit over a third of a seat's score variance;
the rest is play.** On same-rung tables the cut is 12 % for the weakest band and 44-74 % for
`medium` and above (weak players' own randomness dominates their results).

**Additive model.** Observed vs predicted margin over the 29 pairings (two seats of `a` per game):
25 agree within 1.5 SE; `easy`-`hard` (-1.30 vs -2.14, SE 0.41), `hard`-`net-t1` (-1.50 vs -2.19,
0.36) and `expert`-`hard` (+0.94 vs +1.41, 0.39) are off by 1.2-2 SE; a 1-point margin is 100
rating points.

**A player's rating after each deal.** `R <- R + K * c * (margin - expected margin)` with
expected margin = `(R - mean of the four seats' ratings) / c` and `K = max(1 / (n + 21), 1 / 400)`
(the start rating counts as 20 deals; after 400 deals a fixed step). Precision of a rating that
is the mean margin of n deals (95 % half-width, margin SD 21.1 against the model):

| deals | 10 | 50 | 100 | 400 | 2,000 |
|---|---:|---:|---:|---:|---:|
| +- rating | 1,335 | 597 | 422 | 211 | 94 |

So +-100 needs about 1,800 deals, +-300 about 200. The fixed step 1/400 leaves a jitter SD of 76
points on a player whose strength does not change and follows a changed strength within a few
hundred deals. The app should show the half-width beside the rating (or hide the rating until
about 50 deals).

## 6. Proposal (in `runs/p6_grading/proposal.json`) and hand-over

1. Trivial = forced or flat (Task 42's rule); no grade word, in no score.
2. Lines great / good / inaccuracy / mistake = 0.5 / 2.5 / 7.0 / 10.5 points, blunder above.
3. Accuracy = mean of `100 * exp(-loss / 10.1)` over graded decisions; APL beside it; the
   APL-based rating (2828 - 1818 ln APL) shown as "performance this game" if at all.
4. Rating = points won minus luck (16 self-play samples of the expert's network on the deal),
   additive model, random 0 / raw network 3000, `c` = 102 a point, update with K = max(1/(n+21),
   1/400); show the +- half-width.

Hand-over for hk-games plan Tasks E2 and E3: `runs/p6_grading/proposal.json` (keys `trivial`,
`lines`, `accuracy_s`, `elo_from_apl`, `rating`) and Brian's answer in `docs/inbox.md`. The luck
term needs the hour-24 network on the device (16 forward-pass games of about 40 decisions, about
0.2 s in WebAssembly at 0.25 ms a pass).

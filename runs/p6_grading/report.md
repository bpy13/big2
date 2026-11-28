### Trivial decisions

| band | decisions | forced | flat | trivial | graded |
|---|---:|---:|---:|---:|---:|
| net-t4.34 | 16,952 | 47.1 % | 1.7 % | 48.7 % | 8,689 |
| easy | 13,159 | 46.4 % | 1.5 % | 47.9 % | 6,858 |
| medium | 13,844 | 43.0 % | 1.7 % | 44.6 % | 7,663 |
| hard | 13,096 | 41.5 % | 1.8 % | 43.2 % | 7,435 |
| net-t1 | 13,472 | 47.3 % | 1.5 % | 48.9 % | 6,889 |
| expert | 12,543 | 43.6 % | 1.3 % | 44.8 % | 6,922 |
| net | 13,138 | 46.1 % | 1.6 % | 47.7 % | 6,875 |

### Lines

Noise half-width (median 1.96 x loss SE, played move not best): 0.483. Raw rule values: 0.48, 2.72, 6.99, 9.73; rounded to 0.5 and lifted to 1.5x: **0.5 / 2.5 / 7.0 / 10.5** (D102: 0.2 / 0.69 / 2.75 / 6.86).

| band | loss exceeded on 10 % | 5 % | 3 % | 1 % |
|---|---:|---:|---:|---:|
| net-t4.34 | 5.67 | 8.72 | 11.40 | 18.49 |
| easy | 4.39 | 7.11 | 9.84 | 19.70 |
| medium | 4.29 | 7.18 | 9.73 | 15.74 |
| hard | 4.01 | 6.70 | 9.21 | 14.82 |
| net-t1 | 2.92 | 5.44 | 7.75 | 13.49 |
| expert | 3.45 | 5.44 | 7.02 | 11.40 |
| net | 2.72 | 5.08 | 6.99 | 11.96 |

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

Same word on a new seed: 84.6 % (2522); under `belief=rules`: 82.2 % (2522).

Word shares over graded decisions, proposed lines:

| band | great | good | inaccuracy | mistake | blunder |
|---|---:|---:|---:|---:|---:|
| net-t4.34 | 54.6 % | 22.2 % | 16.0 % | 3.5 % | 3.6 % |
| easy | 66.0 % | 17.0 % | 11.8 % | 2.4 % | 2.7 % |
| medium | 60.8 % | 21.0 % | 12.9 % | 2.8 % | 2.5 % |
| hard | 65.1 % | 18.7 % | 11.5 % | 2.5 % | 2.1 % |
| net-t1 | 71.4 % | 17.1 % | 7.9 % | 1.7 % | 1.8 % |
| expert | 69.9 % | 16.5 % | 10.5 % | 1.7 % | 1.3 % |
| net | 74.9 % | 14.2 % | 8.0 % | 1.6 % | 1.4 % |

Same word on a new seed: 88.1 % (2522); under `belief=rules`: 87.5 % (2522).

### Accuracy and APL

accuracy_s = 10.099 (a loss on the inaccuracy line scores 50).

| band | rung Elo (Task 40) | luck-removed rating | graded decisions | APL | per-game APL SD | accuracy | per-game accuracy SD | best-or-good share | games for +-100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| net-t4.34 | 1096 | 1557 | 8,689 | 1.919 | 1.978 | 87.2 | 9.4 | 76.9 % | 1,350 |
| easy | 1161 | 1989 | 6,858 | 1.489 | 2.232 | 90.1 | 9.5 | 83.0 % | 2,857 |
| medium | 1200 | 2305 | 7,663 | 1.467 | 1.460 | 90.5 | 8.5 | 81.8 % | 1,259 |
| hard | 1222 | 2425 | 7,435 | 1.326 | 1.608 | 91.5 | 8.7 | 83.9 % | 1,867 |
| net-t1 | 1230 | 2871 | 6,889 | 1.004 | 1.309 | 93.5 | 7.5 | 88.5 % | 2,160 |
| expert | 1236 | 2712 | 6,922 | 1.017 | 1.213 | 93.0 | 7.7 | 86.4 % | 1,806 |
| net | 1248 | 3000 | 6,875 | 0.886 | 1.131 | 94.2 | 7.0 | 89.0 % | 2,068 |

Rating from APL: rating = 2828 + (-1818) x ln(APL); RMS residual 103 rating points.

Spearman rank correlation of each measure with the rungs' order:

| measure | vs Task 40 Elo (7 rungs) | vs luck-removed rating (7 rungs) | single games vs rating |
|---|---:|---:|---:|
| apl | 0.964 | 1.000 | 0.295 |
| accuracy | 0.964 | 1.000 | 0.295 |
| best_good_share | 0.929 | 0.964 | 0.247 |

### Luck-removed rating

Luck: the hour-24 network plays all four seats of the deal; m=16 sampled games at temperature 1. Games 42,400 on 10,600 deals. c = 102.0 rating points per point of margin a deal; margin SD 21.11; deals for +-100 (95 %): 1783.

95 % half-width of a player's rating after 10 deals +-1335, 50 deals +-597, 100 deals +-422, 400 deals +-211, 2000 deals +-94; steady step k_min = 0.0025 leaves a jitter SD of 76 on a constant player; the start rating counts as 20 deals.

| bot | strength (points a deal) | rating | 95 % range | half A | half B |
|---|---:|---:|---:|---:|---:|
| random | -20.74 | 0 | [0, 0] | -62 | 62 |
| net-t7.07 | -9.48 | 1149 | [1049, 1240] | 1092 | 1205 |
| net-t4.34 | -5.48 | 1557 | [1470, 1637] | 1518 | 1596 |
| easy | -1.24 | 1989 | [1936, 2046] | 2016 | 1962 |
| net-t2.66 | +1.02 | 2220 | [2088, 2352] | 2142 | 2298 |
| medium | +1.85 | 2305 | [2252, 2360] | 2355 | 2254 |
| hard | +3.03 | 2425 | [2346, 2518] | 2518 | 2331 |
| expert | +5.85 | 2712 | [2661, 2767] | 2743 | 2682 |
| net-t1 | +7.41 | 2871 | [2815, 2925] | 2860 | 2883 |
| net | +8.67 | 3000 | [3000, 3000] | 3025 | 2975 |
| puct2000 | +9.13 | 3047 | [2981, 3111] | 3066 | 3029 |

Variance cut over all seats: score SD 27.58 -> margin SD 22.18, cut 35.3 %; duplicate leave-one-out baseline cut 32.9 % (margin SD 22.60).

Self-play noise: one sampled game's score varies with SD 16.02 around the deal's expectation, so the cut at other sample counts is m=1: 3.6 %, m=4: 29.0 %, m=16: 35.3 %, m=64: 36.9 %; limit (exact expectation) 37.4 %.

Same-rung tables (Task 42's deals): variance cut and mean margin by seat (should be 0):

| band | deals | cut | seat 0 | seat 1 | seat 2 | seat 3 |
|---|---:|---:|---:|---:|---:|---:|
| net-t4.34 | 300 | 11.9 % | -0.78 +- 1.18 | +0.30 +- 1.34 | +0.91 +- 1.37 | -0.43 +- 1.40 |
| easy | 300 | 7.5 % | -2.65 +- 1.34 | +2.56 +- 1.26 | +0.96 +- 1.27 | -0.86 +- 1.19 |
| medium | 300 | 44.1 % | -1.36 +- 1.06 | -0.12 +- 1.09 | +0.90 +- 1.04 | +0.58 +- 1.11 |
| hard | 300 | 48.3 % | -0.71 +- 1.34 | -2.51 +- 1.13 | +2.14 +- 1.23 | +1.08 +- 1.10 |
| net-t1 | 300 | 66.0 % | -1.62 +- 0.94 | +0.97 +- 1.00 | +1.02 +- 0.97 | -0.37 +- 0.97 |
| expert | 300 | 50.7 % | +0.22 +- 1.14 | -1.11 +- 1.29 | +0.04 +- 1.17 | +0.85 +- 1.12 |
| net | 300 | 73.7 % | -0.07 +- 0.89 | -0.94 +- 0.82 | -0.42 +- 0.90 | +1.42 +- 0.83 |

Pairings: observed mean margin of a (per game, two seats) vs the additive model:

| a | b | games | observed | +- SE | predicted |
|---|---|---:|---:|---:|---:|
| easy | expert | 2000 | -3.65 | 0.26 | -3.55 |
| easy | hard | 800 | -1.30 | 0.41 | -2.14 |
| easy | medium | 2000 | -1.51 | 0.29 | -1.55 |
| easy | net | 2000 | -4.67 | 0.24 | -4.96 |
| easy | net-t1 | 2000 | -4.50 | 0.25 | -4.32 |
| easy | net-t2.66 | 800 | -1.32 | 0.42 | -1.13 |
| easy | net-t4.34 | 800 | +1.75 | 0.44 | +2.12 |
| easy | net-t7.07 | 800 | +4.07 | 0.45 | +4.12 |
| easy | random | 2000 | +9.61 | 0.32 | +9.75 |
| expert | hard | 800 | +0.94 | 0.39 | +1.41 |
| expert | net | 2000 | -1.73 | 0.23 | -1.41 |
| expert | net-t1 | 2000 | -0.95 | 0.21 | -0.78 |
| expert | puct2000 | 2000 | -1.60 | 0.22 | -1.64 |
| expert | random | 2000 | +13.83 | 0.32 | +13.29 |
| hard | medium | 800 | +0.27 | 0.43 | +0.59 |
| hard | net-t1 | 800 | -1.50 | 0.36 | -2.19 |
| medium | net | 2000 | -3.59 | 0.24 | -3.41 |
| medium | net-t1 | 2000 | -2.73 | 0.22 | -2.78 |
| medium | net-t4.34 | 800 | +3.41 | 0.43 | +3.66 |
| medium | random | 2000 | +11.44 | 0.32 | +11.29 |
| net | net-t1 | 2000 | +0.50 | 0.20 | +0.63 |
| net | puct2000 | 2000 | -0.27 | 0.19 | -0.23 |
| net | random | 2000 | +14.66 | 0.33 | +14.70 |
| net-t1 | net-t4.34 | 800 | +6.17 | 0.44 | +6.44 |
| net-t1 | random | 2000 | +14.03 | 0.32 | +14.07 |
| net-t2.66 | net-t7.07 | 800 | +5.06 | 0.47 | +5.25 |
| net-t4.34 | net-t7.07 | 800 | +1.83 | 0.41 | +2.00 |
| net-t4.34 | random | 800 | +6.91 | 0.47 | +7.63 |
| net-t7.07 | random | 800 | +5.22 | 0.54 | +5.63 |

# 0008 Elo comes from decisions, not points won, on a chess-like 0 to 3000+ scale

Date: 2026-11-28. Replaces the orchestrator's call in hk-games D152 ("keep the app's per-deal Elo
rule"), which went against Brian's wish.

Decision: a rating (bots and players) moves on the quality of the decisions made, measured against
the expert (the network plus the search budget Task 39 picks), so the luck of the cards does not
move it; points won do not count. The scale is chess-like: random play near 0, the expert
(superhuman) at 3000 or above. Task 40's per-deal ladder stays only as an ordering check of the
bots. How decision loss maps onto the scale, and how a player's rating updates from his decisions,
is worked out in Task 43 (with Tasks 41 and 42).

(Brian, 2026-11-28: "spread is small as you've suggested, I want it to be something like chess,
spanning 0 to 3000+ for superhuman level (our expert)"; then: "again, I've mentioned that elo
should be luck removal. The change in elo should be based on decision, not point won")

# 0009: review and hints search deeper in the background; bot moves stay fast

Date: 2026-11-28

Review and hints run in the background (review while or right after the deal is played, a hint from the start of the player's turn, or while others move), so they are not held to the 500/600 ms phone budget. They use PUCT at 2,000 simulations, or at the point where Task 51 shows the gain flattening if that point is higher and a phone can finish it in the background. Bot moves keep the in-game budget (puct200, about 0.13 s a move on a phone, estimated). Hint, review and grade still share one reference (hk-games D139), now this deeper setting.

Facts at the time (Task 39 confirm, job 413, against the raw hour-24 network, 2,000 deals): puct200 +0.11 [-0.06, +0.28] points a deal, phone about 0.13 s a move; puct2000 +0.34 [+0.10, +0.57], phone about 1.3 s a move (phone = 12x native, assumed). Background review at puct2000 is about 20 s a deal (about 16 real decisions, estimated).

Brian (2026-11-28): "since the analysis is buffered, maybe can use deeper depth than 200 on phone", then to the orchestrator's proposal (puct2000 or Task 51's flattening point for review and hints in the background, puct200 for bot moves): "yes".

# Web UI and study mode

## Running it

```sh
conda activate poker
make serve            # builds web/ and serves UI + API on http://127.0.0.1:8000
```

During UI development, run the API and the Vite dev server side by side (the dev
server forwards `/api` to port 8000 and reloads on save):

```sh
uvicorn poker.server.app:app --reload     # terminal 1
make web-dev                              # terminal 2 → http://127.0.0.1:5173
```

Add `?botDelay=0` to the URL to make bots move instantly (default 700 ms per move).

## Playing

* You sit at the bottom; play passes to your right (Bot 1 → Bot 2 → Bot 3).
* Click cards to select them. Cards that cannot be part of any legal play with your
  current selection are dimmed, and **Play** is enabled only for a legal selection.
* Keys: **Enter** play · **P** pass · **Esc** clear.
* **New game** chooses the opponents and every rule variant from
  [rules/big2.md](rules/big2.md).

| Opponents | Bot |
|---|---|
| Easy | Greedy rule-based bot (sheds its weakest cards first, 10% random) |
| Medium | Solver, 300 simulations per move |
| Hard | Solver, 2,000 simulations per move |
| Expert | Solver, 10,000 simulations per move on 4 threads |
| RL network | The distilled network (`models/big2-distilled.pt`), ~1 ms per move |
| RL network + search | The network plus rollout search (~0.2 s per move) |

The RL options appear only when the model file exists and the default rules are
chosen (the network was trained on them). With them available, study mode also
shows the network's own pick under the solver's top 3 — a quick way to see where
fast intuition and search disagree.

From the Phase 1 results, Medium through Expert are close in strength, because extra
simulations add little ([study-results/phase1-solver.md](study-results/phase1-solver.md)).
RL network + search is about as strong as the solver bots; the RL network alone is slightly
weaker ([study-results/phase3-rl.md](study-results/phase3-rl.md)).

## Study mode

The panel on the right has three settings:

* **Off** — no solver output.
* **Hints** — on your turn, the solver's top 3 moves with win rate and expected
  score. Click a move to select its cards.
* **Quiz** — the solver works out its answer silently. After you move, it shows
  whether you matched its top move and, if not, the difference in expected score.

**Optimize: Score / Win** chooses what the ranking maximizes. *Score* (default) is
the expected final score under the table's scoring; *Win* is the chance of going
out first. They often disagree late in a deal, when cutting your card count
(and penalty) matters more than winning the trick. Phase 1 measured that playing
for *Win* costs about 7.6 points per deal.

### How to read the numbers

* **Win rate** — share of simulated games in which you go out first.
* **Exp. score** — average final score (pairwise settlement).
* **±** — 95% interval from simulation noise only. It does not cover the solver's
  assumptions: opponents' hidden cards are sampled uniformly, and opponents are
  simulated by the greedy rollout bot. See [solver.md §3](solver.md#3-reading-the-numbers).

### Post-game review

After a deal, **Review my game** re-examines each of your decisions that had more
than one legal option. It uses PIMC with 300 sampled deals per decision, which
gives every legal move the same number of simulations. That makes "how much did my
move cost?" a fair comparison: the live hints use ISMCTS, which spends most of its
simulations on the moves it likes, so a move you played but it rejected may only
have a few simulations behind it.

Because the two use different estimators, the live verdict and the review can
disagree on close decisions. Moves costing more than 1 point of expected score are
flagged.

## Game logs

Every finished game is saved as JSON in `runs/games/` (set `POKER_GAME_LOG_DIR` to
change it): rules, seed, initial hands and every action, so a game can be replayed
exactly. The same data is available at `GET /api/games/{id}/log`.

## API

The UI uses a small JSON API; the endpoint table is in the docstring of
`python/poker/server/app.py`, and FastAPI serves interactive docs at `/docs`.
Bots move one step per `POST /api/games/{id}/step`, so the client controls pacing
and can animate each move. Games are kept in memory (the 200 most recent).

## Tests

| Test | Command |
|---|---|
| API (full game, illegal moves, analysis, review, logs) | `pytest python/tests/test_server.py` |
| UI helpers (card order, legal-selection matching) | `make test-web` |
| Browser end-to-end: full deal with hints, results, review; quiz verdict | `make test-e2e` |

# Web UI and study mode

## Running it

```sh
conda activate big2
make serve            # builds web/ and serves UI + API on http://127.0.0.1:8000
```

During UI development, run the API and the Vite dev server side by side (the dev
server forwards `/api` to port 8000 and reloads on save):

```sh
uvicorn big2.server.app:app --reload     # terminal 1
make web-dev                              # terminal 2 → http://127.0.0.1:5173
```

Add `?botDelay=0` to the URL to make bots move instantly (default 700 ms per move).

## Playing

* You sit at the bottom; play passes to your right (Bot 1 → Bot 2 → Bot 3).
* Click cards to select them. Cards that cannot be part of any legal play with your
  current selection are dimmed, and **Play** is enabled only for a legal selection.
* When passing is your only legal move, the UI passes for you after a short pause.
* Keys (also in the buttons' tooltips): **Enter** play · **P** pass · **Esc** clear (or close a dialog; **Results** reopens the deal summary).
* **New game** chooses the opponents and every rule variant from
  [rules/big2.md](rules/big2.md).
* After a deal, **Continue** deals the next one with the same opponents, rules and
  network; **New game** (top right) changes them.

| Opponents | Bot |
|---|---|
| Easy | Greedy rule-based bot (sheds its weakest cards first, 10% random) |
| Medium | Solver, 300 simulations per move |
| Hard | Solver, 2,000 simulations per move |
| Expert | Solver, 10,000 simulations per move on 4 threads |
| RL network | The chosen network (see below), ~1 ms per move |
| RL network + search | The network plus rollout search (~0.2 s per move) |

**Network** (in the New game dialog) picks the trained network for the RL
opponents and for the study-mode hint. Each network was trained under one set of
rules and is offered only for games under those rules: the Phase 3 networks
with last-card duty off, the newer one with the current defaults (duty on). The
default is the best network for the chosen rules. If no network was trained for
them, the RL opponents are unavailable and the hint compares the solver with the
greedy bot instead.

From the Phase 1 results, Medium through Expert are close in strength, because extra
simulations add little ([study-results/phase1-solver.md](study-results/phase1-solver.md)).
The RL network (distilled or DMC, +10 h) and RL network + search are about as strong as the
solver bots ([study-results/long-training.md](study-results/long-training.md)).

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

### Solver and network side by side

**Hints** shows two columns: the solver's top 3 moves and the network's top 3
(or the greedy bot's move). The same move has the same colour
in both, so you can see where each ranks the other's choices. Below them:

* **Agree** — both pick the same move.
* **Disagree** — where the solver ranks the network's first choice, and how much
  worse it rates it than its own best (points per game, or win-rate points when
  optimizing for *Win*), with the solver's ± for that move.

The network's numbers are its own predictions from one forward pass: no
simulation, no interval. A PPO network gives each move a probability instead,
and the greedy bot only its move. The greedy bot is how the solver imagines its
opponents play, so comparing with it shows where the solver expects to exploit them.
In **Quiz** mode the verdict also says what the network would have played.

**Compare with** in the panel switches the hint to another network that fits the
game's rules, to the **Greedy bot** (any rules), or to **No comparison** (the
solver alone). Choosing a network only
stages it; it is loaded on the server when you press **Load** (or **Use**, if it
is already in memory). Nothing but the default network is loaded until you ask.
The networks offered, and the rules each was trained under, are listed in
`python/big2/server/models.py`; set `POKER_RL_CHECKPOINT` to add any other
checkpoint (for the default rules) as the default.

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
`python/big2/server/app.py`, and FastAPI serves interactive docs at `/docs`.
Bots move one step per `POST /api/games/{id}/step`, so the client controls pacing
and can animate each move. Games are kept in memory (the 200 most recent).

## Tests

| Test | Command |
|---|---|
| API (full game, illegal moves, analysis, review, logs) | `pytest python/tests/test_server.py` |
| UI helpers (card order, legal-selection matching) | `make test-web` |
| Browser end-to-end: full deal with hints, results, review; quiz verdict | `make test-e2e` |

"""The ``poker`` command-line tool.

Examples::

    poker big2 play                                    # you vs three ISMCTS bots
    poker big2 play --hints --bots greedy              # easier bots, solver hints every turn
    poker big2 analyze --seat 0 --hand "3d 4d 4h 5h 6d 7d 7s 8d 8s Jd Jh Kc 2s"
    poker big2 analyze --seat 1 --hand "..." --history "0:3d; 1:pass; 2:5c"
    poker big2 tournament ismcts:iterations=1000 greedy greedy greedy --deals 200 --workers 6

Run ``poker big2 <command> --help`` for all options.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from poker._core import analyze as core_analyze
from poker.agents import make_agent
from poker.eval import run_tournament
from poker.games.big2 import Big2State, Rules, cards, pretty, state_from_observation

if TYPE_CHECKING:
    from poker._core import AnalysisResult

app = typer.Typer(help="Card-game solver and RL lab.", no_args_is_help=True)
big2 = typer.Typer(help="Big Two (see docs/rules/big2.md).", no_args_is_help=True)
app.add_typer(big2, name="big2")
console = Console()

# Shared rule options -------------------------------------------------------

Straights = Annotated[str, typer.Option(help="with_low_twos | no_twos")]
FlushOrder = Annotated[str, typer.Option(help="rank_first | suit_first")]
PassRule = Annotated[str, typer.Option(help="may_reenter | locked_out")]
Scoring = Annotated[str, typer.Option(help="tiered | original | win_loss")]
LastCardDuty = Annotated[
    bool, typer.Option(help="Force highest single when the next player has 1 card.")
]


def _rules(
    straights: str, flush_order: str, pass_rule: str, scoring: str, last_card_duty: bool
) -> Rules:
    return Rules(straights, flush_order, pass_rule, scoring, last_card_duty)


def _moves_table(result: AnalysisResult, top: int, title: str) -> Table:
    t = Table(title=title)
    t.add_column("#", justify="right")
    t.add_column("Move")
    t.add_column("Win rate", justify="right")
    t.add_column("Exp. score", justify="right")
    t.add_column("Simulations", justify="right")
    for i, m in enumerate(result["moves"][:top], 1):
        t.add_row(
            str(i),
            pretty(m["cards"]),
            f"{m['win_rate']:.1%} ± {m['win_ci95']:.1%}",
            f"{m['mean_score']:+.1f} ± {m['score_ci95']:.1f}",
            f"{m['visits']:,}",
        )
    return t


def _parse_history(text: str) -> list[tuple[int, int]]:
    """Parses ``"0:3d 4d; 1:pass; 2:5c 5h"``."""
    out = []
    for item in filter(None, (x.strip() for x in text.replace("|", ";").split(";"))):
        player, sep, action = item.partition(":")
        if not sep or not player.strip().isdigit():
            raise typer.BadParameter(
                f"history item {item!r} should look like '2:5c 5h' or '1:pass'"
            )
        out.append((int(player), cards(action)))
    return out


# analyze ---------------------------------------------------------------------


@big2.command()
def analyze(
    seat: Annotated[int, typer.Option(help="Your seat (0-3).")],
    hand: Annotated[str, typer.Option(help='Your current hand, e.g. "3d 4c 5h".')],
    history: Annotated[str, typer.Option(help='Actions so far: "0:3d; 1:pass; 2:5c 5h".')] = "",
    iterations: Annotated[int, typer.Option(help="Simulations.")] = 20_000,
    algorithm: Annotated[str, typer.Option(help="ismcts | pimc")] = "ismcts",
    objective: Annotated[str, typer.Option(help="score | win")] = "score",
    threads: Annotated[int, typer.Option(help="Parallel search trees.")] = 4,
    top: Annotated[int, typer.Option(help="Moves to show.")] = 3,
    seed: int = 0,
    straights: Straights = "with_low_twos",
    flush_order: FlushOrder = "rank_first",
    pass_rule: PassRule = "may_reenter",
    scoring: Scoring = "tiered",
    last_card_duty: LastCardDuty = False,
) -> None:
    """Recommend moves for a position, seen from one seat."""
    rules = _rules(straights, flush_order, pass_rule, scoring, last_card_duty)
    try:
        state = state_from_observation(seat, cards(hand), _parse_history(history), rules, seed=seed)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    if state.current_player != seat:
        raise typer.BadParameter(f"it is player {state.current_player}'s turn, not seat {seat}'s")
    result = core_analyze(
        state,
        algorithm=algorithm,
        iterations=iterations,
        threads=threads,
        seed=seed,
        objective=objective,
    )
    console.print(f"Seat {seat} to act. Cards left: {state.hand_sizes}")
    if state.current_play:
        p, c, kind = state.current_play
        console.print(f"To beat: {pretty(c)} ({kind}) by seat {p}")
    else:
        console.print("You lead.")
    title = (
        f"{algorithm.upper()} · {result['simulations']:,} simulations · {result['seconds']:.2f}s"
    )
    console.print(_moves_table(result, top, title))


# play ------------------------------------------------------------------------


def _show_table(state: Big2State, you: int, names: list[str]) -> None:
    sizes = state.hand_sizes
    row = "   ".join(
        f"[bold]{'You' if p == you else f'P{p}'}[/]"
        + ("" if p == you else f" ({names[p]})")
        + f": {sizes[p]} cards"
        for p in range(4)
    )
    console.print(row)
    if state.current_play:
        p, c, kind = state.current_play
        who = "you" if p == you else f"P{p}"
        console.print(f"On the table: [bold]{pretty(c)}[/] ({kind.replace('_', ' ')}) by {who}")
    else:
        console.print("[italic]New trick — you lead.[/]" if state.current_player == you else "")


@big2.command()
def play(
    seat: Annotated[int, typer.Option(help="Your seat (0-3).")] = 0,
    bots: Annotated[
        str, typer.Option(help="Agent spec for the three bots.")
    ] = "ismcts:iterations=2000",
    hints: Annotated[bool, typer.Option(help="Show the solver's top 3 every turn.")] = False,
    hint_iterations: int = 10_000,
    seed: Annotated[int | None, typer.Option(help="Deal seed (random if omitted).")] = None,
    straights: Straights = "with_low_twos",
    flush_order: FlushOrder = "rank_first",
    pass_rule: PassRule = "may_reenter",
    scoring: Scoring = "tiered",
    last_card_duty: LastCardDuty = False,
) -> None:
    """Play a deal against three bots in the terminal.

    On your turn type cards (e.g. "3d 3c"), "pass", "?" for a hint, or "quit".
    """
    rules = _rules(straights, flush_order, pass_rule, scoring, last_card_duty)
    state = rules.new_deal(seed)
    agents = {p: make_agent(bots, seed=(seed or 0) + p) for p in range(4) if p != seat}
    names = [bots if p != seat else "you" for p in range(4)]
    console.rule("Big Two")
    while not state.is_terminal:
        p = state.current_player
        assert p is not None
        if p != seat:
            action = agents[p].act(state)
            console.print(f"P{p}: {pretty(action)}")
            state.apply(action)
            continue
        console.print()
        _show_table(state, seat, names)
        console.print(f"Your hand: [bold cyan]{pretty(state.hand(seat))}[/]")
        if hints:
            r = core_analyze(state, iterations=hint_iterations, threads=4)
            console.print(_moves_table(r, 3, "Solver hint"))
        while True:
            text = console.input("[bold]Your move[/] (cards / pass / ? / quit): ").strip()
            if text.lower() in ("quit", "q", "exit"):
                raise typer.Exit()
            if text == "?":
                r = core_analyze(state, iterations=hint_iterations, threads=4)
                console.print(_moves_table(r, 3, "Solver hint"))
                continue
            try:
                state.apply(cards(text))
                break
            except ValueError as e:
                console.print(f"[red]{e}[/]")
    console.rule("Result")
    winner = state.winner
    console.print(f"{'You' if winner == seat else f'P{winner}'} went out first.")
    scores = state.scores or [0.0] * 4
    for p in range(4):
        label = "You" if p == seat else f"P{p}"
        console.print(f"  {label}: {state.hand_sizes[p]} cards left, score {scores[p]:+.0f}")


# tournament -----------------------------------------------------------------


@big2.command()
def tournament(
    lineup: Annotated[
        list[str],
        typer.Argument(help="Four agent specs, e.g. ismcts:iterations=1000 greedy greedy greedy"),
    ],
    deals: Annotated[int, typer.Option(help="Deals (each played 4× with seats rotated).")] = 100,
    workers: Annotated[int, typer.Option(help="Parallel processes.")] = 1,
    seed: int = 0,
    out: Annotated[str | None, typer.Option(help="Save full results as JSON.")] = None,
    straights: Straights = "with_low_twos",
    flush_order: FlushOrder = "rank_first",
    pass_rule: PassRule = "may_reenter",
    scoring: Scoring = "tiered",
    last_card_duty: LastCardDuty = False,
) -> None:
    """Duplicate tournament between four agents."""
    if len(lineup) != 4:
        raise typer.BadParameter("give exactly four agent specs")
    rules = _rules(straights, flush_order, pass_rule, scoring, last_card_duty)
    with console.status(f"Playing {deals} deals × 4 rotations..."):
        result = run_tournament(lineup, deals, seed=seed, workers=workers, rules=rules)
    t = Table(title=f"{deals} deals · {result.seconds:.0f}s")
    t.add_column("Agent", overflow="fold")
    t.add_column("Games", justify="right")
    t.add_column("Mean score (95% CI)", justify="right")
    t.add_column("Win rate (95% CI)", justify="right")
    t.add_column("s / move", justify="right")
    for s in sorted(result.summaries, key=lambda s: -s.mean_score):
        t.add_row(
            s.agent,
            str(s.games),
            f"{s.mean_score:+.2f} ± {s.score_ci95:.2f}",
            f"{s.win_rate:.1%} ± {s.win_ci95:.1%}",
            f"{s.seconds_per_move:.4f}",
        )
    console.print(t)
    if out:
        result.save(out)
        console.print(f"Saved to {out}")


# serve ------------------------------------------------------------------------


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: Annotated[bool, typer.Option(help="Restart on code changes (development).")] = False,
) -> None:
    """Start the web UI and API (build the UI first with `make web`)."""
    import uvicorn

    from poker.server.app import WEB_DIST

    if not WEB_DIST.is_dir():
        console.print(
            f"[yellow]No built UI at {WEB_DIST}; serving the API only. Run `make web` first.[/]"
        )
    console.print(f"Open http://{host}:{port}")
    uvicorn.run("poker.server.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()

"""Training-first Big Two strategy lab."""

import random
from typing import Any, cast

import pandas as pd
import streamlit as st

from big2_rl.game import BigTwoGame, BigTwoState
from big2_rl.rules import (
    PASS_ACTION,
    SCORE_ORIGINAL,
    SCORE_TIERED,
    SCORING_LABELS,
    card_name,
    encode_card_mask,
)
from big2_rl.training import (
    TrainingRun,
    checkpoint_bytes,
    default_checkpoint_path,
    evaluate_self_play,
    load_checkpoint,
    load_checkpoint_data,
    new_training_run,
    save_checkpoint,
    train_batch,
)

st.set_page_config(page_title="Big Two Strategy Lab", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f3f6f2; }
    h1, h2, h3 { color: #17382b; font-family: Georgia, serif; }
    [data-testid="stMetricValue"] { color: #16734d; }
    .playing-card {
        display: inline-flex; flex-direction: column; justify-content: space-between;
        width: 42px; height: 60px; margin: 3px; padding: 5px 6px;
        background: #fff; border: 1px solid #bac8bd; border-radius: 5px;
        box-shadow: 0 2px 4px #15281a20; font-weight: 700; color: #1b2820;
    }
    .playing-card.red { color: #b52c34; }
    .playing-card .suit { align-self: flex-end; font-size: 20px; line-height: 1; }
    .lede { color: #496256; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _initialize_session() -> None:
    defaults: dict[str, Any] = {
        "runs": {},
        "training_history": {},
        "game_state": None,
        "game_scoring_mode": None,
        "human_seat": 0,
        "game_rng": random.Random(2026),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_run(scoring_mode: str) -> TrainingRun:
    run: TrainingRun | None = st.session_state.runs.get(scoring_mode)
    if run is None:
        checkpoint = default_checkpoint_path(scoring_mode)
        if checkpoint.exists():
            try:
                run = load_checkpoint(checkpoint, scoring_mode)
            except (OSError, ValueError, KeyError) as error:
                st.warning(f"Could not load saved checkpoint: {error}")
                run = new_training_run(scoring_mode, seed=2026)
        else:
            run = new_training_run(scoring_mode, seed=2026)
        st.session_state.runs[scoring_mode] = run
    return cast(TrainingRun, run)


def _deal_game(scoring_mode: str) -> BigTwoState:
    state = cast(BigTwoState, BigTwoGame(scoring_mode=scoring_mode).new_initial_state())
    rng = st.session_state.game_rng
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        card = rng.choices(
            [action for action, _ in outcomes],
            weights=[probability for _, probability in outcomes],
            k=1,
        )[0]
        state.apply_action(card)
    return state


def _play_bot_turns(state: BigTwoState, human_seat: int, run: TrainingRun) -> None:
    policy = run.solver.average_policy()
    rng = st.session_state.game_rng
    decision_count = 0
    while not state.is_terminal() and state.current_player() != human_seat:
        player = state.current_player()
        probabilities = policy.action_probabilities(state, player)
        actions = list(probabilities)
        action = rng.choices(
            actions, weights=[probabilities[item] for item in actions], k=1
        )[0]
        state.apply_action(action)
        decision_count += 1
        if decision_count >= 256:
            raise RuntimeError("bot turn limit exceeded")


def _card_hand(cards: tuple[int, ...]) -> None:
    fragments = []
    for card in cards:
        label = card_name(card)
        rank, suit = label[:-1], label[-1]
        color = "red" if suit in ("♦", "♥") else "black"
        fragments.append(
            f'<span class="playing-card {color}"><span>{rank}</span>'
            f'<span class="suit">{suit}</span></span>'
        )
    st.markdown(
        '<div class="big2-hand">{}</div>'.format("".join(fragments)),
        unsafe_allow_html=True,
    )


def _render_strategy_hints(state: BigTwoState, run: TrainingRun) -> None:
    player = state.current_player()
    strategy = run.solver.average_policy().action_probabilities(state, player)
    ranked_actions = sorted(strategy.items(), key=lambda item: item[1], reverse=True)
    total_information_sets = len(run.solver._infostates)
    st.markdown("#### Approximate strategy hint")
    st.caption(
        "Sampled self-play strategy, not an exact answer or guaranteed equilibrium. "
        f"{total_information_sets} information sets currently stored."
    )
    if total_information_sets == 0:
        st.info(
            "No training data yet. This information set currently uses "
            "a uniform fallback strategy."
        )
    for action, probability in ranked_actions[:3]:
        label = state.action_to_string(player, action)
        st.write(f"**{label}** · {probability:.1%}")


def _render_play(scoring_mode: str) -> None:
    st.subheader("Play Big Two")
    st.markdown(
        '<p class="lede">Play one seat against three agents using the current '
        "average self-play strategy; inspect its most likely plays on your turn.</p>",
        unsafe_allow_html=True,
    )

    state = cast(BigTwoState | None, st.session_state.game_state)
    if state is None:
        controls = st.columns([1, 1, 3])
        human_seat = controls[0].selectbox("Your seat", tuple(range(4)), index=0)
        controls[1].markdown(" ")
        if controls[1].button("Deal a hand", type="primary", key="deal_big2"):
            st.session_state.human_seat = human_seat
            st.session_state.game_state = _deal_game(scoring_mode)
            st.session_state.game_scoring_mode = scoring_mode
            st.rerun()
        st.caption("Bots automatically play until it is your turn.")
        return

    game_mode = cast(str, st.session_state.game_scoring_mode)
    if game_mode != scoring_mode:
        st.info("This hand is using the scoring rule selected when it was dealt.")
    run = _get_run(game_mode)
    human_seat = st.session_state.human_seat
    _play_bot_turns(state, human_seat, run)

    score_label = SCORING_LABELS[game_mode]
    st.caption(score_label)
    table = st.columns(4)
    for player, column in enumerate(table):
        column.metric(
            f"Seat {player}" + (" · You" if player == human_seat else ""),
            state.cards_remaining()[player],
            help="Cards remaining in the private hand.",
        )

    if state.is_terminal():
        winner = state.winner
        st.success(f"Seat {winner} went out first.")
        returns = state.returns()
        st.markdown("**Pairwise score settlement**")
        scores = st.columns(4)
        for player, score_column in enumerate(scores):
            score_column.metric(f"Seat {player} net", f"{returns[player]:+.0f}")
        if st.button("Deal another hand", type="primary", key="next_big2"):
            st.session_state.game_state = _deal_game(scoring_mode)
            st.session_state.game_scoring_mode = scoring_mode
            st.rerun()
        return

    player = state.current_player()
    if player != human_seat:
        st.error("A bot turn did not advance to the human seat.")
        return

    st.markdown("#### Your cards")
    _card_hand(state.player_hand(player))
    current_play = state.current_play()
    if current_play is None:
        st.write("You are leading this trick.")
    else:
        leader = state.trick_leader()
        if leader is None:
            raise RuntimeError("active play has no leader")
        action_label = state.action_to_string(
            leader, _play_to_action(state, current_play, leader)
        )
        st.write(f"Current play: **{action_label}** · led by seat {leader}")

    _render_strategy_hints(state, run)
    legal_actions = state.legal_actions()
    playable_actions = [action for action in legal_actions if action != PASS_ACTION]
    action_labels = {
        action: state.action_to_string(player, action) for action in playable_actions
    }
    controls = st.columns([3, 1, 1])
    if playable_actions:
        chosen_action = controls[0].selectbox(
            "Choose cards",
            playable_actions,
            format_func=lambda action: action_labels[action],
            key=f"big2-play-{len(state.public_history())}",
        )
        if controls[1].button("Play", type="primary", key="play_big2"):
            state.apply_action(chosen_action)
            st.rerun()
    else:
        controls[0].caption("No legal combination beats the current play.")
    if PASS_ACTION in legal_actions and controls[2].button("Pass", key="pass_big2"):
        state.apply_action(PASS_ACTION)
        st.rerun()

    with st.expander("Public trick history"):
        for event in state.public_history():
            st.write(event)


def _play_to_action(state: BigTwoState, play, leader: int) -> int:
    return encode_card_mask(play.cards, state.original_hand(leader))


def _render_training(scoring_mode: str) -> None:
    st.subheader("Train approximate strategies")
    st.markdown(
        '<p class="lede">Outcome-sampling MCCFR trains average strategies from '
        "sampled four-player deals. Checkpoints let you save and continue later.</p>",
        unsafe_allow_html=True,
    )

    run = _get_run(scoring_mode)
    checkpoint = default_checkpoint_path(scoring_mode)
    metrics = st.columns(3)
    metrics[0].metric("MCCFR iterations", f"{run.iterations:,}")
    metrics[1].metric("Information sets", f"{len(run.solver._infostates):,}")
    metrics[2].metric(
        "Scoring mode",
        "Custom tiers" if scoring_mode == SCORE_TIERED else "Earlier tiers",
    )
    st.caption(
        "One MCCFR iteration samples one traversal for each of the four players."
    )

    batch_size = st.select_slider(
        "Iterations in next batch", (1, 2, 5, 10, 25, 50, 100), value=5
    )
    train_columns = st.columns([1, 1, 1, 3])
    if train_columns[0].button("Train batch", type="primary", key="train_big2"):
        train_batch(run, batch_size)
        evaluation = evaluate_self_play(run, episodes=12, seed=2026)
        history = st.session_state.training_history.setdefault(scoring_mode, [])
        history.append(
            {
                "iterations": run.iterations,
                "information_sets": len(run.solver._infostates),
                "seat_0_win_rate": evaluation["win_rates"][0],
                "mean_absolute_return": evaluation["mean_absolute_return"],
            }
        )
        st.rerun()

    if train_columns[1].button("Save checkpoint", key="save_big2"):
        save_checkpoint(run, checkpoint)
        st.success(f"Saved {run.iterations:,} iterations to `{checkpoint}`")
    if checkpoint.exists():
        if train_columns[2].button("Load saved checkpoint", key="load_big2"):
            st.session_state.runs[scoring_mode] = load_checkpoint(
                checkpoint, scoring_mode
            )
            st.rerun()
    else:
        train_columns[2].caption("No local checkpoint yet.")

    uploaded = st.file_uploader(
        "Resume from a checkpoint file", type=("json",), key="big2_checkpoint_upload"
    )
    if uploaded is not None and st.button(
        "Load uploaded checkpoint", key="load_upload_big2"
    ):
        st.session_state.runs[scoring_mode] = load_checkpoint_data(
            uploaded.getvalue(), scoring_mode
        )
        st.rerun()
    st.download_button(
        "Download current strategy",
        data=checkpoint_bytes(run),
        file_name=f"{scoring_mode}.json",
        mime="application/json",
        key="download_big2",
    )

    history = st.session_state.training_history.get(scoring_mode, [])
    if len(history) >= 2:
        frame = pd.DataFrame(history).set_index("iterations")
        chart_columns = st.columns(2)
        chart_columns[0].line_chart(frame[["information_sets"]], height=240)
        chart_columns[0].caption("Learned information-set coverage")
        chart_columns[1].line_chart(
            frame[["seat_0_win_rate", "mean_absolute_return"]], height=240
        )
        chart_columns[1].caption("Sampled self-play diagnostics, not equilibrium proof")
    elif history:
        st.info("Train another batch to compare multiple self-play snapshots.")
    else:
        st.info("Train a batch to start the progress traces.")

    st.caption(f"Local checkpoint path: `{checkpoint}`")


def _render_rules() -> None:
    st.subheader("Rules and model limits")
    st.markdown(
        """
        Four players receive 13 cards each. Rank order is 3 through Ace, then 2;
        suit order is diamonds < clubs < hearts < spades. The player holding 3♦ opens
        and must include it in the opening play. Passes are allowed; after three
        consecutive passes, the last player to play leads the next trick.

        Singles, pairs, triples, and five-card combinations are legal. Five-card
        categories rank straight < flush < full house < four of a kind < straight
        flush. For this version, straights use consecutive ranks from 3 through Ace;
        twos do not appear in straights. Aces are high. A combination must match the
        number of cards in the previous play.

        The default score multiplies cards left by the selected tier: 1–7 ×1, 8–9 ×2,
        10–12 ×3, 13 ×4. The alternate uses 1–9 ×1, 10–12 ×2, 13 ×3. Every pair of
        players settles the difference: the higher-penalty player pays the lower-penalty
        player. These transfers sum to zero.

        Big Two is a four-player hidden-information game. This app uses sampled MCCFR
        self-play, and its hints are approximate average strategies—not a guaranteed
        Nash equilibrium. The self-play charts show empirical behavior, not a proven
        equilibrium or exploitability bound.
        """
    )


def main() -> None:
    _initialize_session()
    st.title("Big Two Strategy Lab")
    with st.sidebar:
        st.subheader("Scoring")
        scoring_mode = st.selectbox(
            "Penalty schedule",
            (SCORE_TIERED, SCORE_ORIGINAL),
            format_func=lambda value: SCORING_LABELS[value],
        )
        st.caption("Each pair settles the difference in penalty points.")

    train_tab, play_tab, rules_tab = st.tabs(("Train", "Play", "Rules"))
    with train_tab:
        _render_training(scoring_mode)
    with play_tab:
        _render_play(scoring_mode)
    with rules_tab:
        _render_rules()


if __name__ == "__main__":
    main()

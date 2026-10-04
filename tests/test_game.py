import random

from big2_rl.game import BigTwoGame, BigTwoState
from big2_rl.rules import (
    FIRST_CARD,
    PASS_ACTION,
    SCORE_ORIGINAL,
    SCORE_TIERED,
    card_name,
)


def deal_random_game(scoring_mode=SCORE_TIERED):
    game = BigTwoGame(scoring_mode=scoring_mode)
    state = game.new_initial_state()
    rng = random.Random(11)
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        card = rng.choices(
            [action for action, _ in outcomes],
            weights=[probability for _, probability in outcomes],
            k=1,
        )[0]
        state.apply_action(card)
    return game, state


def test_four_player_deal_and_opening_must_contain_three_of_diamonds():
    game, state = deal_random_game()
    assert game.num_players() == 4
    assert len(state._dealt) == 52
    assert sorted(state.cards_remaining()) == [13, 13, 13, 13]
    legal_actions = state.legal_actions()
    assert legal_actions
    assert all(FIRST_CARD in state.decode_action(action) for action in legal_actions)


def test_hidden_hands_do_not_appear_in_another_players_information_state():
    _, state = deal_random_game()
    first_player, second_player = 0, 1
    second_hand = set(state.player_hand(second_player))
    first_information = state.information_state_string(first_player)
    assert all(card_name(card) not in first_information for card in second_hand)


def test_three_passes_return_lead_to_last_player():
    _, state = deal_random_game()
    first_player = state.current_player()
    opening = next(
        action
        for action in state.legal_actions()
        if _action_has_three_diamonds(state, action)
    )
    state.apply_action(opening)
    lead_player = first_player
    for _ in range(3):
        assert PASS_ACTION in state.legal_actions()
        state.apply_action(PASS_ACTION)
    assert state.current_player() == lead_player
    assert state.current_play() is None
    assert state.trick_leader() == lead_player
    assert PASS_ACTION not in state.legal_actions()


def test_game_accepts_both_scoring_modes_and_rejects_unknown_mode():
    assert BigTwoGame(scoring_mode=SCORE_TIERED).scoring_mode == SCORE_TIERED
    assert BigTwoGame(scoring_mode=SCORE_ORIGINAL).scoring_mode == SCORE_ORIGINAL
    try:
        BigTwoGame(scoring_mode="unrecognized")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown scoring mode should be rejected")


def _action_has_three_diamonds(state: BigTwoState, action: int) -> bool:
    return FIRST_CARD in state.decode_action(action)

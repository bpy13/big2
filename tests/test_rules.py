from big2_rl.rules import (
    FIRST_CARD,
    FULL_HOUSE,
    PAIR,
    SCORE_ORIGINAL,
    SCORE_TIERED,
    SINGLE,
    STRAIGHT_FLUSH,
    beats,
    card_name,
    classify_play,
    legal_plays,
    pairwise_settlement,
    penalty_points,
)


def test_card_order_and_three_of_diamonds_id():
    assert FIRST_CARD == 0
    assert card_name(0) == "3♦"
    assert card_name(51) == "2♠"


def test_single_pair_and_five_card_classification():
    single = classify_play((0,))
    pair = classify_play((0, 1))
    full_house = classify_play((0, 1, 2, 4, 5))
    straight_flush = classify_play((0, 4, 8, 12, 16))
    assert single is not None and single.category == SINGLE
    assert pair is not None and pair.category == PAIR
    assert full_house is not None and full_house.category == FULL_HOUSE
    assert straight_flush is not None and straight_flush.category == STRAIGHT_FLUSH


def test_five_card_play_must_be_one_of_supported_poker_shapes():
    assert classify_play((0, 5, 10, 15, 21)) is None


def test_combinations_must_match_size_and_beat_the_lead():
    lead = classify_play((0,))
    assert lead is not None
    hand = (4, 8, 12)
    plays = legal_plays(hand, lead)
    assert plays
    assert all(len(play.cards) == 1 and beats(play, lead) for play in plays)


def test_custom_and_earlier_penalty_tiers():
    assert [penalty_points(count, SCORE_TIERED) for count in (7, 8, 9, 10, 12, 13)] == [
        7,
        16,
        18,
        30,
        36,
        52,
    ]
    assert [penalty_points(count, SCORE_ORIGINAL) for count in (9, 10, 12, 13)] == [
        9,
        20,
        24,
        39,
    ]


def test_pairwise_settlement_balances_every_transfer():
    payouts = pairwise_settlement((52, 20, 7, 0))
    assert payouts == (-129, -1, 51, 79)
    assert sum(payouts) == 0

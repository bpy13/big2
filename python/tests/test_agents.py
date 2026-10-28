"""Agent construction and fair-play tests."""

import pytest
from big2.agents import (
    GreedyAgent,
    PatwaGreedy,
    PatwaSmart,
    RandomAgent,
    SearchAgent,
    make_agent,
)
from big2.agents.patwa import five_card_combos
from big2.eval import play_game
from big2.games.big2 import PASS, Big2State, Rules, cards


def test_make_agent_parses_specs() -> None:
    assert isinstance(make_agent("random"), RandomAgent)
    g = make_agent("greedy:epsilon=0.2")
    assert isinstance(g, GreedyAgent) and g.epsilon == 0.2
    s = make_agent("ismcts:iterations=50,exploration=0.5,objective=win")
    assert isinstance(s, SearchAgent)
    assert s.iterations == 50
    assert s.options == {"exploration": 0.5, "objective": "win"}
    with pytest.raises(ValueError):
        make_agent("alphago")


@pytest.mark.parametrize("spec", ["random", "greedy", "ismcts:iterations=50", "pimc:iterations=5"])
def test_agents_finish_a_game(spec: str) -> None:
    agents = [make_agent(spec, seed=i) for i in range(4)]
    final = play_game(Rules().new_deal(seed=7), agents)
    assert final.is_terminal
    assert final.scores is not None and sum(final.scores) == 0


def test_search_agent_ignores_hidden_cards() -> None:
    """Two states that differ only in cards hidden from the player to act must get
    the same decision and statistics from a seeded search agent."""
    s1 = Rules().new_deal(seed=11)
    p = s1.current_player
    assert p is not None
    s2 = s1.determinize(p, seed=99)
    assert s1.hands != s2.hands and s1.hand(p) == s2.hand(p)
    a1 = SearchAgent(iterations=500, seed=1).analyze(s1)
    a2 = SearchAgent(iterations=500, seed=1).analyze(s2)
    assert a1["moves"] == a2["moves"]


def test_make_agent_builds_patwa_agents() -> None:
    assert isinstance(make_agent("patwa-greedy"), PatwaGreedy)
    s = make_agent("patwa-smart:two=5,endgame=3")
    assert isinstance(s, PatwaSmart) and s.two == 5 and s.endgame == 3


@pytest.mark.parametrize("spec", ["patwa-greedy", "patwa-smart"])
def test_patwa_agents_finish_games(spec: str) -> None:
    for seed in range(5):
        agents = [make_agent(spec) for _ in range(4)]
        final = play_game(Rules().new_deal(seed=seed), agents)
        assert final.is_terminal


def _position(hand: str, to_beat: str | None) -> Big2State:
    """Seat 0 holds ``hand`` and acts; seat 3 made ``to_beat`` (or seat 0 leads).

    Seats 1–3 get three spare cards each, so the last-card duty never applies.
    """
    mine = cards(hand)
    table = cards(to_beat) if to_beat else 0
    free = [1 << i for i in range(52) if not (mine | table) >> i & 1]
    others = [free[0] | free[1] | free[2], free[3] | free[4] | free[5], free[6] | free[7] | free[8]]
    play = (3, table) if to_beat else None
    return Big2State.from_position([mine, *others], 0, play, **Rules().kwargs())


def test_patwa_greedy_plays_lowest_action() -> None:
    hand = "4d 4c 7h 9c 9d Qs Qh 2s"
    # Leading: the lowest single, whatever the hand's structure.
    assert PatwaGreedy().act(_position(hand, None)) == cards("4d")
    # Following a pair: the lowest pair that beats it.
    assert PatwaGreedy().act(_position(hand, "6d 6c")) == cards("9d 9c")


def test_patwa_smart_and_greedy_disagree_on_structure() -> None:
    # Following a 6: greedy breaks the pair of 7s with 7d; smart sheds its orphan 9h.
    s = _position("4d 4c 7c 7d 9h Qs Qh Kd", "6d")
    assert PatwaGreedy().act(s) == cards("7d")
    assert PatwaSmart().act(s) == cards("9h")


def test_patwa_smart_keeps_its_two_and_takes_a_win() -> None:
    # Only the 2 beats the king: greedy plays it, smart keeps it and passes.
    s = _position("4d 4c 7h 9c 9d Qs Qh 2s", "Kd")
    assert PatwaGreedy().act(s) == cards("2s")
    assert PatwaSmart().act(s) == PASS
    # With its last card, smart goes out.
    assert PatwaSmart().act(_position("2s", "Kd")) == cards("2s")


def test_patwa_smart_avoids_breaking_a_five_card_hand() -> None:
    # 5–9 straight plus a pair of queens. Following a 4, the lone K is an orphan;
    # smart plays it rather than a straight card or a queen.
    s = _position("5d 6c 7h 8s 9d Qs Qh Kd", "4d")
    assert five_card_combos(s.hand(0), s.rules()) == [cards("5d 6c 7h 8s 9d")]
    assert PatwaSmart().act(s) == cards("Kd")


def test_search_agent_accepts_large_seeds() -> None:
    # Regression: the web server uses 48-bit deal seeds; derived seeds must fit a u64.
    agent = make_agent("ismcts:iterations=20", seed=2**48 + 3)
    state = Rules().new_deal(seed=2**48)
    assert agent.act(state) in state.legal_actions()

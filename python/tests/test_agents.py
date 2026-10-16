"""Agent construction and fair-play tests."""

import pytest
from poker.agents import GreedyAgent, RandomAgent, SearchAgent, make_agent
from poker.eval import play_game
from poker.games.big2 import Rules


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

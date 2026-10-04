"""Four-player imperfect-information Big Two game for OpenSpiel."""

import pyspiel

from .rules import (
    ACTION_COUNT,
    CARD_COUNT,
    FIRST_CARD,
    PASS_ACTION,
    SCORE_TIERED,
    SCORING_LABELS,
    Play,
    beats,
    card_name,
    classify_play,
    decode_card_mask,
    encode_card_mask,
    legal_plays,
    pairwise_settlement,
    penalty_points,
)

NUM_PLAYERS = 4
MAX_GAME_LENGTH = 512
RULESET_VERSION = 1
_GAME_TYPE = pyspiel.GameType(
    short_name="python_big2",
    long_name="Python Big Two",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=NUM_PLAYERS,
    min_num_players=NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=False,
    provides_observation_tensor=False,
)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=ACTION_COUNT,
    max_chance_outcomes=CARD_COUNT,
    num_players=NUM_PLAYERS,
    min_utility=-156.0,
    max_utility=156.0,
    utility_sum=0.0,
    max_game_length=MAX_GAME_LENGTH,
)


class BigTwoGame(pyspiel.Game):
    """One four-player deal, with one of the supported penalty schedules."""

    def __init__(self, params=None, scoring_mode: str = SCORE_TIERED):
        if params:
            raise ValueError("Big Two is configured with scoring_mode, not game params")
        scoring = scoring_mode
        if scoring not in SCORING_LABELS:
            raise ValueError(f"unknown scoring mode: {scoring}")
        super().__init__(_GAME_TYPE, _GAME_INFO, {})
        self.scoring_mode = scoring

    def new_initial_state(self):
        return BigTwoState(self)


class BigTwoState(pyspiel.State):
    """A deal state; private hands stay hidden until the deal is scored."""

    def __init__(self, game: BigTwoGame):
        super().__init__(game)
        self._big2_game = game
        self._dealt: list[int] = []
        self._hands: list[set[int]] = [set() for _ in range(NUM_PLAYERS)]
        self._original_hands: list[tuple[int, ...]] = []
        self._public_history: list[str] = []
        self._current = pyspiel.PlayerId.CHANCE
        self._leader: int | None = None
        self._last_player: int | None = None
        self._last_play: Play | None = None
        self._passes = 0
        self._first_play = True
        self._game_over = False
        self._winner: int | None = None

    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        return self._current

    def chance_outcomes(self):
        remaining = [card for card in range(CARD_COUNT) if card not in self._dealt]
        probability = 1.0 / len(remaining)
        return [(card, probability) for card in remaining]

    def _legal_actions(self, player):
        if player != self._current or player < 0:
            return []
        if self._first_play:
            current_play = None
            must_include = FIRST_CARD
        else:
            current_play = self._last_play
            must_include = None

        plays = legal_plays(self._hands[player], current_play, must_include)
        actions = [
            encode_card_mask(play.cards, self._original_hands[player]) for play in plays
        ]
        if current_play is not None:
            actions.append(PASS_ACTION)
        return actions

    def _apply_action(self, action):
        if self.is_chance_node():
            if action in self._dealt:
                raise ValueError("a card cannot be dealt twice")
            player = len(self._dealt) % NUM_PLAYERS
            self._dealt.append(action)
            self._hands[player].add(action)
            if len(self._dealt) == CARD_COUNT:
                self._original_hands = [tuple(sorted(hand)) for hand in self._hands]
                self._leader = next(
                    player
                    for player, hand in enumerate(self._hands)
                    if FIRST_CARD in hand
                )
                self._current = self._leader
            return

        player = self._current
        if action == PASS_ACTION:
            if self._last_play is None:
                raise ValueError("a player cannot pass while leading")
            self._public_history.append(f"p{player}:pass")
            self._passes += 1
            if self._passes == NUM_PLAYERS - 1:
                self._current = self._last_player
                self._last_player = None
                self._last_play = None
                self._passes = 0
            else:
                self._current = (player + 1) % NUM_PLAYERS
            return

        cards = decode_card_mask(action, self._original_hands[player])
        if not cards or any(card not in self._hands[player] for card in cards):
            raise ValueError("play mask contains a card not held by the player")
        play = classify_play(cards)
        if play is None or (
            self._last_play is not None and not beats(play, self._last_play)
        ):
            raise ValueError("play is not a legal response")
        if self._first_play and FIRST_CARD not in cards:
            raise ValueError("the opening play must include the 3 of diamonds")

        self._hands[player].difference_update(cards)
        self._public_history.append(
            f"p{player}:" + ",".join(card_name(card) for card in cards)
        )
        self._last_play = play
        self._last_player = player
        self._passes = 0
        self._first_play = False
        if not self._hands[player]:
            self._game_over = True
            self._winner = player
        else:
            self._current = (player + 1) % NUM_PLAYERS

    def is_terminal(self):
        return self._game_over

    def returns(self):
        if not self._game_over:
            return [0.0] * NUM_PLAYERS
        penalties = tuple(
            penalty_points(len(hand), self._big2_game.scoring_mode)
            for hand in self._hands
        )
        return [float(score) for score in pairwise_settlement(penalties)]

    def information_state_string(self, player):
        private_hand = ",".join(card_name(card) for card in self._hands[player])
        played_counts = "/".join(
            str(len(self._original_hands[index]) - len(self._hands[index]))
            for index in range(NUM_PLAYERS)
        )
        public_history = ";".join(self._public_history)
        return (
            f"seat={player}|hand={private_hand}|played={played_counts}|"
            f"history={public_history}"
        )

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            return f"Deal:{card_name(action)}"
        if action == PASS_ACTION:
            return "Pass"
        cards = decode_card_mask(action, self._original_hands[player])
        return " ".join(card_name(card) for card in cards)

    def __str__(self):
        history = " ".join(self._public_history)
        return f"dealt={len(self._dealt)} current={self._current} history={history}"

    def player_hand(self, player: int) -> tuple[int, ...]:
        return tuple(sorted(self._hands[player]))

    def decode_action(self, action: int, player: int | None = None) -> tuple[int, ...]:
        player = self._current if player is None else player
        return decode_card_mask(action, self._original_hands[player])

    def original_hand(self, player: int) -> tuple[int, ...]:
        return self._original_hands[player]

    def current_play(self):
        return self._last_play

    def trick_leader(self) -> int | None:
        if self._last_play is not None:
            return self._last_player
        return self._current if self._current >= 0 else self._leader

    def cards_remaining(self) -> tuple[int, ...]:
        return tuple(len(hand) for hand in self._hands)

    def public_history(self) -> tuple[str, ...]:
        return tuple(self._public_history)

    @property
    def winner(self) -> int | None:
        return self._winner


pyspiel.register_game(_GAME_TYPE, BigTwoGame)

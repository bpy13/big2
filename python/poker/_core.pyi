"""Type stubs for the Rust extension module ``poker._core`` (crates/pybind).

Card sets are ``int`` bit masks (index = rank*4 + suit, ranks 3..2, suits ♦♣♥♠);
an action is the mask of cards played, ``0`` = pass.
"""

from typing import Any, TypedDict

__version__: str

class MoveStats(TypedDict):
    cards: int
    visits: int
    mean_score: float
    score_ci95: float
    win_rate: float
    win_ci95: float

class AnalysisResult(TypedDict):
    player: int
    simulations: int
    seconds: float
    moves: list[MoveStats]

def parse_cards(text: str) -> int: ...
def format_cards(cards: int) -> str: ...
def classify_cards(
    cards: int, *, straights: str | None = None, flush_order: str | None = None
) -> tuple[str, int] | None: ...

class Big2State:
    @staticmethod
    def new_deal(
        seed: int | None = None,
        *,
        straights: str | None = None,
        flush_order: str | None = None,
        pass_rule: str | None = None,
        scoring: str | None = None,
        last_card_duty: bool | None = None,
    ) -> Big2State: ...
    @staticmethod
    def from_hands(
        hands: list[int] | tuple[int, int, int, int],
        *,
        straights: str | None = None,
        flush_order: str | None = None,
        pass_rule: str | None = None,
        scoring: str | None = None,
        last_card_duty: bool | None = None,
    ) -> Big2State: ...
    @staticmethod
    def from_position(
        hands: list[int] | tuple[int, int, int, int],
        current: int,
        current_play: tuple[int, int] | None = None,
        *,
        straights: str | None = None,
        flush_order: str | None = None,
        pass_rule: str | None = None,
        scoring: str | None = None,
        last_card_duty: bool | None = None,
    ) -> Big2State: ...
    def rules(self) -> dict[str, Any]: ...
    @property
    def current_player(self) -> int | None: ...
    @property
    def hands(self) -> list[int]: ...
    def hand(self, player: int) -> int: ...
    @property
    def hand_sizes(self) -> list[int]: ...
    @property
    def played(self) -> int: ...
    @property
    def current_play(self) -> tuple[int, int, str] | None: ...
    @property
    def is_opening(self) -> bool: ...
    @property
    def locked_out(self) -> list[bool]: ...
    @property
    def history(self) -> list[tuple[int, int]]: ...
    @property
    def winner(self) -> int | None: ...
    @property
    def is_terminal(self) -> bool: ...
    @property
    def scores(self) -> list[float] | None: ...
    @property
    def penalties(self) -> list[int]: ...
    def legal_actions(self) -> list[int]: ...
    def apply(self, cards: int) -> None: ...
    def child(self, cards: int) -> Big2State: ...
    def determinize(self, observer: int, seed: int | None = None) -> Big2State: ...
    def greedy_action(self, epsilon: float = 0.0, seed: int | None = None) -> int: ...
    def random_action(self, seed: int | None = None) -> int: ...
    def copy(self) -> Big2State: ...

def analyze(
    state: Big2State,
    *,
    algorithm: str = "ismcts",
    iterations: int = 10_000,
    time_limit: float | None = None,
    exploration: float | None = None,
    threads: int = 1,
    seed: int = 0,
    objective: str = "score",
    endgame_threshold: int = 0,
    endgame_max_nodes: int = 20_000,
    rollout_epsilon: float = 0.1,
) -> AnalysisResult: ...

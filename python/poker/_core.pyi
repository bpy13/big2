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
    belief: str | None = None,
    belief_candidates: int = 256,
    belief_epsilon: float = 0.2,
    belief_min_ess: float = 0.25,
    value: str | None = None,
    value_rollout_weight: float = 0.0,
) -> AnalysisResult: ...
def value_estimate(state: Big2State, path: str) -> tuple[list[float], list[float]]: ...

# --- Phase 3: RL environment (crates/pybind/src/vecenv.rs) -----------------

class Big2VecEnv:
    """Many games in lockstep. Features are 0/1 bytes (crates/big2/src/features.rs)."""

    def __init__(
        self,
        n: int,
        seed: int = 0,
        mode: str = "selfplay",
        auto_reset: bool = True,
        bot_epsilon: float = 0.0,
        *,
        straights: str | None = None,
        flush_order: str | None = None,
        pass_rule: str | None = None,
        scoring: str | None = None,
        last_card_duty: bool | None = None,
    ) -> None: ...
    @staticmethod
    def from_states(states: list[Big2State]) -> Big2VecEnv: ...
    def scores(self, i: int) -> list[float] | None: ...
    def winner(self, i: int) -> int | None: ...
    @property
    def n(self) -> int: ...
    @property
    def num_done(self) -> int: ...
    @staticmethod
    def feature_sizes() -> tuple[int, int]: ...
    def reset(self) -> None: ...
    def observe(self) -> tuple[bytes, bytes, list[int], list[int], list[int]]: ...
    def full_features(self) -> bytes: ...
    def legal_cards(self, i: int) -> list[int]: ...
    def step(self, choices: list[int]) -> list[tuple[int, list[float], int, int | None]]: ...
    def state(self, i: int) -> Big2State: ...

def encode(state: Big2State) -> tuple[bytes, bytes, list[int]]: ...
def encode_full(state: Big2State) -> bytes: ...

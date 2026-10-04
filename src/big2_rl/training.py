"""Resumable outcome-sampling MCCFR training for Big Two."""

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from open_spiel.python.algorithms import outcome_sampling_mccfr

from .game import RULESET_VERSION, BigTwoGame, BigTwoState
from .rules import SCORE_TIERED, SCORING_LABELS

CHECKPOINT_VERSION = 1
_TRAINING_LOCK = threading.Lock()


class SelfPlayMetrics(TypedDict):
    episodes: int
    win_rates: tuple[float, ...]
    mean_returns: tuple[float, ...]
    mean_absolute_return: float


class BigTwoSolver(outcome_sampling_mccfr.OutcomeSamplingSolver):
    """MCCFR solver with explicit export/import of its information-set tables."""

    def export_information_sets(self) -> dict[str, dict[str, list[float]]]:
        return {
            key: {
                "regrets": values[0].astype(float).tolist(),
                "average_strategy": values[1].astype(float).tolist(),
            }
            for key, values in self._infostates.items()
        }

    def import_information_sets(
        self, information_sets: dict[str, dict[str, list[float]]]
    ) -> None:
        restored = {}
        for key, values in information_sets.items():
            regrets = np.asarray(values["regrets"], dtype=np.float64)
            average = np.asarray(values["average_strategy"], dtype=np.float64)
            if regrets.ndim != 1 or average.shape != regrets.shape:
                raise ValueError(
                    f"invalid action-vector shape for information set {key!r}"
                )
            if not np.isfinite(regrets).all() or not np.isfinite(average).all():
                raise ValueError(
                    f"non-finite solver values for information set {key!r}"
                )
            restored[key] = [regrets, average]
        self._infostates = restored


@dataclass
class TrainingRun:
    scoring_mode: str
    solver: BigTwoSolver
    iterations: int
    rng_state: Any


def new_training_run(scoring_mode: str = SCORE_TIERED, seed: int = 0) -> TrainingRun:
    if scoring_mode not in SCORING_LABELS:
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    rng = np.random.RandomState(seed)
    solver = BigTwoSolver(BigTwoGame(scoring_mode=scoring_mode))
    return TrainingRun(scoring_mode, solver, 0, rng.get_state())


def train_batch(run: TrainingRun, iterations: int) -> int:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    with _TRAINING_LOCK:
        process_rng_state = np.random.get_state()
        np.random.set_state(run.rng_state)
        try:
            for _ in range(iterations):
                run.solver.iteration()
        finally:
            run.rng_state = np.random.get_state()
            np.random.set_state(process_rng_state)
    run.iterations += iterations
    return run.iterations


def _encode_rng_state(state: Any) -> dict[str, Any]:
    return {
        "algorithm": state[0],
        "keys": state[1].astype(np.uint32).tolist(),
        "position": int(state[2]),
        "has_gaussian": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _decode_rng_state(payload: dict[str, Any]) -> Any:
    return (
        payload["algorithm"],
        np.asarray(payload["keys"], dtype=np.uint32),
        int(payload["position"]),
        int(payload["has_gaussian"]),
        float(payload["cached_gaussian"]),
    )


def _open_spiel_version() -> str:
    try:
        return version("open-spiel")
    except PackageNotFoundError:
        return "unknown"


def _checkpoint_payload(run: TrainingRun) -> dict:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "ruleset_version": RULESET_VERSION,
        "open_spiel_version": _open_spiel_version(),
        "scoring_mode": run.scoring_mode,
        "iterations": run.iterations,
        "rng_state": _encode_rng_state(run.rng_state),
        "information_sets": run.solver.export_information_sets(),
    }


def checkpoint_bytes(run: TrainingRun) -> bytes:
    return json.dumps(_checkpoint_payload(run), separators=(",", ":")).encode("utf-8")


def save_checkpoint(run: TrainingRun, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(run)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(payload, temporary_file, separators=(",", ":"))
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_checkpoint_data(data: bytes, scoring_mode: str | None = None) -> TrainingRun:
    payload = json.loads(data.decode("utf-8"))
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    if payload.get("ruleset_version") != RULESET_VERSION:
        raise ValueError("checkpoint uses a different Big Two ruleset version")
    saved_mode = payload.get("scoring_mode")
    if saved_mode not in SCORING_LABELS:
        raise ValueError("checkpoint has an unknown scoring mode")
    if scoring_mode is not None and saved_mode != scoring_mode:
        raise ValueError("checkpoint scoring mode does not match the selected mode")

    solver = BigTwoSolver(BigTwoGame(scoring_mode=saved_mode))
    solver.import_information_sets(payload["information_sets"])
    return TrainingRun(
        scoring_mode=saved_mode,
        solver=solver,
        iterations=int(payload["iterations"]),
        rng_state=_decode_rng_state(payload["rng_state"]),
    )


def load_checkpoint(path: str | Path, scoring_mode: str | None = None) -> TrainingRun:
    return load_checkpoint_data(Path(path).read_bytes(), scoring_mode)


def default_checkpoint_path(scoring_mode: str) -> Path:
    if scoring_mode not in SCORING_LABELS:
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    repository_root = Path(__file__).resolve().parents[2]
    model_directory = Path(os.environ.get("BIG2_MODEL_DIR", repository_root / "models"))
    return model_directory / f"{scoring_mode}.json"


def evaluate_self_play(
    run: TrainingRun, episodes: int = 24, seed: int = 2026
) -> SelfPlayMetrics:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    policy = run.solver.average_policy()
    evaluation_rng = np.random.RandomState(seed)
    wins = np.zeros(4, dtype=np.float64)
    returns = np.zeros(4, dtype=np.float64)

    for _ in range(episodes):
        state = BigTwoGame(scoring_mode=run.scoring_mode).new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, chance_probabilities = zip(*state.chance_outcomes())
                action = int(evaluation_rng.choice(outcomes, p=chance_probabilities))
            else:
                player = state.current_player()
                action_probabilities = policy.action_probabilities(state, player)
                actions = list(action_probabilities)
                action_probabilities_vector = np.asarray(
                    [action_probabilities[action] for action in actions],
                    dtype=np.float64,
                )
                action_probabilities_vector /= action_probabilities_vector.sum()
                action = int(
                    evaluation_rng.choice(actions, p=action_probabilities_vector)
                )
            state.apply_action(action)

        result = np.asarray(state.returns(), dtype=np.float64)
        returns += result
        if isinstance(state, BigTwoState) and state.winner is not None:
            wins[state.winner] += 1

    return {
        "episodes": episodes,
        "win_rates": tuple((wins / episodes).tolist()),
        "mean_returns": tuple((returns / episodes).tolist()),
        "mean_absolute_return": float(np.abs(returns / episodes).mean()),
    }

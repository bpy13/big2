"""Tests for the RL environment, network, training loop and agents."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from poker._core import Big2State, Big2VecEnv, encode  # noqa: E402
from poker.agents import make_agent  # noqa: E402
from poker.eval import play_game  # noqa: E402
from poker.games.big2 import Rules  # noqa: E402
from poker.rl.device import actor_device, resolve_device  # noqa: E402
from poker.rl.dmc import TrainConfig, choose, train  # noqa: E402
from poker.rl.model import ACTION_LEN, STATE_LEN, QNet  # noqa: E402
from poker.rl.search import rollout_search  # noqa: E402

# Every device this machine can train on; the trainers are tested on each.
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if torch.backends.mps.is_available():
    DEVICES.append("mps")


def random_rollout(
    env: Big2VecEnv, rng: np.random.Generator
) -> list[tuple[int, list[float], int, int | None]]:
    finished = []
    while env.num_done < env.n:
        _, _, counts, ids, _ = env.observe()
        if not ids:
            break
        choices = (rng.random(len(counts)) * np.asarray(counts)).astype(int)
        finished += env.step(choices.tolist())
    return finished


def test_vecenv_observation_shapes() -> None:
    env = Big2VecEnv(8, seed=1)
    st, ac, counts, ids, players = env.observe()
    assert len(st) == len(ids) * STATE_LEN
    assert len(ac) == sum(counts) * ACTION_LEN
    assert len(ids) == len(players) == 8
    # The first state of each game is an opening: 13 cards in hand, leading.
    x = np.frombuffer(st, dtype=np.uint8).reshape(len(ids), STATE_LEN)
    assert (x[:, :52].sum(axis=1) == 13).all()


def test_selfplay_games_finish_zero_sum() -> None:
    env = Big2VecEnv(16, seed=2, auto_reset=False)
    finished = random_rollout(env, np.random.default_rng(0))
    assert len(finished) == 16
    for _, scores, winner, seat in finished:
        assert sum(scores) == 0
        assert 0 <= winner < 4
        assert seat is None


def test_vs_greedy_groups_share_a_deal_with_rotated_learner() -> None:
    env = Big2VecEnv(8, seed=3, mode="vs_greedy", auto_reset=False)

    def initial_hands(s: Big2State) -> list[int]:
        hands = list(s.hands)
        for p, c in s.history:
            hands[p] |= c
        return hands

    for g in range(2):
        deals = [initial_hands(env.state(4 * g + j)) for j in range(4)]
        assert all(d == deals[0] for d in deals)
    # Each observed game waits on its learner seat, i.e. seat i % 4.
    _, _, _, ids, players = env.observe()
    assert all(p == i % 4 for i, p in zip(ids, players, strict=True))
    finished = random_rollout(env, np.random.default_rng(1))
    assert sorted(f[0] for f in finished) == list(range(8))
    assert all(f[3] == f[0] % 4 for f in finished)


def test_encode_matches_vecenv() -> None:
    s = Rules().new_deal(seed=5)
    for _ in range(9):
        s.apply(s.greedy_action())
    st, ac, legal = encode(s)
    env = Big2VecEnv.from_states([s])
    st2, ac2, counts, _, _ = env.observe()
    assert st == st2 and ac == ac2 and counts == [len(legal)]
    assert env.legal_cards(0) == legal


def test_choose_takes_argmax_or_explores() -> None:
    net = QNet()
    env = Big2VecEnv(4, seed=6)
    st, ac, counts_l, ids, _ = env.observe()
    counts = np.asarray(counts_l)
    states = np.frombuffer(st, dtype=np.uint8).reshape(len(ids), STATE_LEN)
    actions = np.frombuffer(ac, dtype=np.uint8).reshape(-1, ACTION_LEN)
    rng = np.random.default_rng(0)
    greedy = choose(net, states, actions, counts, 0.0, rng)
    with torch.no_grad():
        index = torch.from_numpy(np.repeat(np.arange(len(counts)), counts))
        q, _ = net(
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            index,
        )
    start = 0
    for k, n in enumerate(counts):
        assert greedy[k] == int(q[start : start + n].argmax())
        start += n
    explored = choose(net, states, actions, counts, 1.0, rng)
    assert ((explored >= 0) & (explored < counts)).all()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("dmc")
    cfg = TrainConfig(
        run_dir=str(run), envs=32, games_per_iter=64, max_iters=2, eval_every=2, eval_deals=8
    )
    train(cfg, log=lambda r: None)
    return run


def test_training_writes_logs_and_checkpoints(checkpoint: Path) -> None:
    lines = (checkpoint / "log.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert (checkpoint / "latest.pt").exists() and (checkpoint / "best.pt").exists()
    ckpt = torch.load(checkpoint / "latest.pt", weights_only=False)
    assert ckpt["info"]["iter"] == 2 and "optimizer" in ckpt


def test_rl_agents_play_full_games(checkpoint: Path) -> None:
    ck = checkpoint / "latest.pt"
    for spec in [f"dmc:checkpoint={ck}", f"dmcsearch:checkpoint={ck},determinizations=4,top=3"]:
        agents = [make_agent(spec, seed=i) for i in range(4)]
        final = play_game(Rules().new_deal(seed=9), agents)
        assert final.is_terminal and final.scores is not None


def test_rollout_search_reports_candidates(checkpoint: Path) -> None:
    from poker.rl.agent import load_model

    s = Rules().new_deal(seed=4)
    moves = rollout_search(load_model(checkpoint / "latest.pt"), s, determinizations=8, top=4)
    assert 1 <= len(moves) <= 4
    assert all(m["visits"] == 8 for m in moves)
    assert set(map(int, s.legal_actions())) >= {m["cards"] for m in moves}


def test_resolve_device() -> None:
    assert resolve_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            resolve_device("cuda")
    assert actor_device("cuda") == "cuda" and actor_device("mps") == "cpu"


@pytest.mark.parametrize("device", [d for d in DEVICES if d != "cpu"])
def test_dmc_trains_on_device(tmp_path: Path, device: str) -> None:
    cfg = TrainConfig(
        run_dir=str(tmp_path),
        envs=16,
        games_per_iter=16,
        max_iters=1,
        eval_every=1,
        eval_deals=4,
        device=device,
    )
    train(cfg, log=lambda r: None)
    ckpt = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert ckpt["train_config"]["device"] == device
    assert all(v.device.type == "cpu" for v in ckpt["model"].values())


def test_rl_specs_need_a_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        make_agent("dmc")


@pytest.mark.parametrize("device", DEVICES)
def test_distillation_generates_and_trains(tmp_path: Path, device: str) -> None:
    from poker.rl import distill

    data = tmp_path / "data"
    n = distill.generate(str(data), workers=1, minutes=0.02, iterations=100, shard_games=1)
    assert n > 0 and list(data.glob("*.npz"))
    shard = np.load(next(data.glob("*.npz")))
    assert shard["states"].shape[1] == STATE_LEN and shard["actions"].shape[1] == ACTION_LEN
    assert (shard["visits"] >= 20).all() and (np.diff(shard["owner"]) >= 0).all()
    cfg = distill.DistillConfig(
        data_dir=str(data),
        run_dir=str(tmp_path / "run"),
        max_minutes=0.05,
        eval_every_minutes=0.02,
        eval_deals=4,
        batch_size=16,
        device=device,
    )
    run = distill.train(cfg, log=lambda r: None)
    assert (run / "latest.pt").exists()
    agent = make_agent(f"dmc:checkpoint={run / 'latest.pt'}")
    assert agent.act(Rules().new_deal(seed=1)) in Rules().new_deal(seed=1).legal_actions()


@pytest.mark.parametrize("device", DEVICES)
def test_ppo_trains_and_its_checkpoint_plays(tmp_path: Path, device: str) -> None:
    from poker.rl.ppo import PPOConfig, segment_log_softmax
    from poker.rl.ppo import train as ppo_train

    # Log-softmax within segments sums to probability 1 per segment.
    logits = torch.tensor([0.0, 1.0, 2.0, 5.0, -1.0])
    seg = torch.tensor([0, 0, 0, 1, 1])
    p = segment_log_softmax(logits, seg, 2).exp()
    assert torch.allclose(torch.zeros(2).index_add(0, seg, p), torch.ones(2))

    cfg = PPOConfig(
        run_dir=str(tmp_path),
        envs=16,
        games_per_iter=16,
        max_minutes=0.05,
        eval_every=1,
        eval_deals=4,
        device=device,
    )
    ppo_train(cfg, log=lambda r: None)
    agent = make_agent(f"dmc:checkpoint={tmp_path / 'latest.pt'}")
    s = Rules().new_deal(seed=2)
    assert agent.act(s) in s.legal_actions()

"""Tests for the RL environment, network, training loop and agents."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from big2._core import Big2State, Big2VecEnv, encode  # noqa: E402
from big2.agents import make_agent  # noqa: E402
from big2.eval import play_game  # noqa: E402
from big2.games.big2 import Rules  # noqa: E402
from big2.rl.device import actor_device, resolve_device  # noqa: E402
from big2.rl.dmc import TrainConfig, choose, train  # noqa: E402
from big2.rl.model import ACTION_LEN, STATE_LEN, QNet  # noqa: E402
from big2.rl.search import rollout_search  # noqa: E402

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
    from big2.rl.agent import load_model

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
    from big2.rl import distill

    data = tmp_path / "data"
    n = distill.generate(str(data), workers=1, minutes=0.02, iterations=100, shard_games=1)
    assert n > 0 and list(data.glob("*.npz"))
    shard = np.load(next(data.glob("*.npz")))
    assert shard["states"].shape[1] == STATE_LEN and shard["actions"].shape[1] == ACTION_LEN
    assert (shard["visits"] >= 0).all() and (np.diff(shard["owner"]) >= 0).all()
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
    steps = torch.load(run / "latest.pt", weights_only=False)["info"]["iter"]
    # More data from a second generator run lands beside the first, then training resumes.
    first = set(data.glob("*.npz"))
    time.sleep(1.1)  # new run prefix
    distill.generate(str(data), workers=1, minutes=0.01, iterations=100, shard_games=1)
    assert set(data.glob("*.npz")) > first
    resumed = replace(cfg, run_dir=str(tmp_path / "run2"), resume=str(run / "latest.pt"))
    distill.train(resumed, log=lambda r: None)
    assert torch.load(tmp_path / "run2" / "latest.pt", weights_only=False)["info"]["iter"] > steps
    agent = make_agent(f"dmc:checkpoint={run / 'latest.pt'}")
    assert agent.act(Rules().new_deal(seed=1)) in Rules().new_deal(seed=1).legal_actions()


def test_distill_gen_round0_fields(tmp_path: Path) -> None:
    """Task 11: a 2-game run stores every candidate with its search statistics and
    the deal's outcome per decision (shard format 2), plus progress and results."""
    import json

    from big2._core import candidates
    from big2.rl import distill

    data = tmp_path / "data"
    n = distill.generate(
        str(data), workers=1, minutes=0, iterations=100, shard_games=2, rollout="greedy"
    )
    shards = list(data.glob("*.npz"))
    assert len(shards) == 1
    z = np.load(shards[0])
    assert int(z["format"]) == distill.SHARD_FORMAT == 2
    d, rows = len(z["states"]), len(z["owner"])
    assert d == n > 20  # two whole deals
    assert z["states"].shape == (d, STATE_LEN)
    for key, shape in [
        ("player", (d,)),
        ("final_cards", (d, 4)),
        ("final_scores", (d, 4)),
        ("next_leader", (d,)),
        ("chosen", (d,)),
    ]:
        assert z[key].shape == shape, key
    assert z["actions"].shape == (rows, ACTION_LEN)
    for key in ("cards", "keeps", "breaks", "visits", "score", "win"):
        assert z[key].shape == (rows,), key
    owner = z["owner"]
    assert (np.diff(owner) >= 0).all() and set(owner.tolist()) == set(range(d))
    counts = np.bincount(owner, minlength=d)
    assert (counts >= 2).all()  # only searched decisions are stored
    assert (z["chosen"] >= 0).all() and (z["chosen"] < counts).all()
    lo = np.concatenate([[0], np.cumsum(counts)[:-1]])
    assert (z["visits"][lo + z["chosen"]] > 0).all()  # the played move was searched
    assert np.bincount(owner, weights=z["visits"]).min() >= 90  # ~100 simulations each
    # Planner order: moves that keep a decomposition come before the others.
    keeps = z["keeps"]
    for i in range(d):
        k = keeps[lo[i] : lo[i] + counts[i]]
        assert not ((k[:-1] < 0) & (k[1:] >= 0)).any()
    assert ((keeps >= 0) | (z["breaks"] > 0) | (z["cards"] == 0)).all()
    # Outcome relative to the actor: exactly one seat ends with no cards, scores sum to 0.
    fc = z["final_cards"]
    assert ((fc == 0).sum(axis=1) == 1).all() and (fc <= 13).all()
    assert np.allclose(z["final_scores"].sum(axis=1), 0, atol=1e-4)
    winner_abs = (z["player"] + np.argmax(fc == 0, axis=1)) % 4
    assert len(set(winner_abs.tolist())) <= 2  # one winner per deal
    nl = z["next_leader"]
    # -1 appears only when a searched decision sits in the deal's last trick; a deal
    # whose last trick is one forced lead (the winner's only legal move) has none.
    assert ((nl >= -1) & (nl <= 3)).all() and (nl >= 0).any()
    # The binding lists every legal action once.
    s = Rules().new_deal(seed=0)
    assert sorted(c for c, _, _ in candidates(s)) == sorted(s.legal_actions())
    progress = json.loads((tmp_path / "progress.json").read_text())
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["status"] == "done" and results["decisions"] == d
    assert progress["candidate_rows"] == rows and results["config"]["rollout"] == "greedy"
    # A target already reached returns at once (the resume path of the long job).
    assert distill.generate(str(data), workers=1, iterations=100, target_decisions=d) == 0


def test_qnet_v2_shapes_and_size() -> None:
    """Task 12: per candidate 4 × 14 cards-left logits and one prior logit; ~0.8–1.0 M
    parameters with the v1 towers (state 512–256, move 128)."""
    from big2.rl.model import QNetV2, V2Scorer

    net = QNetV2()
    params = sum(p.numel() for p in net.parameters())
    assert 800_000 <= params <= 1_000_000, params
    st = torch.rand(3, STATE_LEN).round()
    ac = torch.rand(7, ACTION_LEN).round()
    index = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    dist, prior = net(st, ac, index)
    assert dist.shape == (7, 4, 14) and prior.shape == (7,)
    for mode in ("prior", "score"):
        score, zero = V2Scorer(net, mode)(st, ac, index)
        assert score.shape == (7,) and not zero.any()
    # Dropout adds no parameters and is off in eval mode (the Rust reader has none).
    from big2.rl.model import NetConfigV2

    drop = QNetV2(NetConfigV2(dropout=0.2))
    assert sum(p.numel() for p in drop.parameters()) == params
    drop.eval()
    assert torch.equal(drop(st, ac, index)[1], drop(st, ac, index)[1])


def test_pvnet_agent_plays_a_v2_checkpoint_under_each_scoring(tmp_path: Path) -> None:
    """Task 15 (c): ``pvnet:checkpoint=…`` plays a v2 network greedily, scoring moves
    by the game's own scoring (``tiered`` or ``win_loss``), or by the prior."""
    from dataclasses import asdict

    from big2.rl.model import NetConfigV2, QNetV2, V2Scorer

    cfg = NetConfigV2(state_hidden=32, state_out=16, action_out=16, head_hidden=32)
    net = QNetV2(cfg)
    ckpt = tmp_path / "v2.pt"
    torch.save({"arch": "qnet_v2", "net_config": asdict(cfg), "model": net.state_dict()}, ckpt)
    net.eval()
    for scoring in ("tiered", "win_loss"):
        agent = make_agent(f"pvnet:checkpoint={ckpt}")
        s = Rules(scoring=scoring).new_deal(seed=3)
        # The agent's move is the argmax of the actor's expected score under this scoring.
        s_enc, a_enc, legal = encode(s)
        st = torch.from_numpy(np.frombuffer(s_enc, np.uint8).reshape(1, -1).astype(np.float32))
        ac = torch.from_numpy(
            np.frombuffer(a_enc, np.uint8).reshape(-1, ACTION_LEN).astype(np.float32)
        )
        with torch.no_grad():
            score, _ = V2Scorer(net, "score", scoring)(
                st, ac, torch.zeros(len(legal), dtype=torch.long)
            )
        assert agent.act(s) == legal[int(score.argmax())]
        final = play_game(s, [agent, make_agent("greedy"), agent, make_agent("greedy")])
        assert final.is_terminal
    assert make_agent(f"pvnet:checkpoint={ckpt},mode=prior").act(Rules().new_deal(1)) in (
        Rules().new_deal(1).legal_actions()
    )
    with pytest.raises(ValueError):
        make_agent(f"pvnet:checkpoint={ckpt},mode=bad")
    with pytest.raises(ValueError):
        make_agent("pvnet")


def test_v2_payoff_reproduces_the_rules_settlement() -> None:
    """Task 12: the tiered payoff is rules §7.1's penalty table; one-hot distributions
    of a finished deal give exactly the engine's scores under every scoring."""
    from big2.rl.model import payoff, seat_scores

    assert payoff("tiered").tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 16, 18, 30, 33, 36, 52]
    assert payoff("original").tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 22, 24, 39]
    assert payoff("win_loss").tolist() == [0] + [1] * 13
    with pytest.raises(ValueError):
        payoff("sum_of_others")
    rng = np.random.default_rng(5)
    for scoring in ("tiered", "original", "win_loss"):
        for seed in range(20):
            s = Rules(scoring=scoring).new_deal(seed=seed)
            while not s.is_terminal:
                legal = s.legal_actions()
                s.apply(legal[int(rng.integers(len(legal)))])
            onehot = torch.nn.functional.one_hot(torch.tensor(s.hand_sizes), 14).float()
            got = seat_scores(onehot, scoring).tolist()
            assert got == pytest.approx(list(s.scores), abs=1e-5), (scoring, s.hand_sizes)


def test_v2_segment_log_softmax_matches_per_group() -> None:
    from big2.rl.model import segment_log_softmax

    logits = torch.tensor([1.0, 2.0, -3.0, 0.5, 40.0, 41.0])
    index = torch.tensor([0, 0, 0, 1, 2, 2])
    got = segment_log_softmax(logits, index, 3)
    want = torch.cat(
        [logits[:3].log_softmax(0), logits[3:4].log_softmax(0), logits[4:].log_softmax(0)]
    )
    assert torch.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_distill_train_v2_writes_results_and_resumes(tmp_path: Path, device: str) -> None:
    """Task 12: ``train_v2`` on format-2 shards writes per-head validation losses,
    ``best.pt``, ``latest.pt``, progress and results, and resumes from ``latest.pt``."""
    import json
    import shutil

    from big2.rl import distill

    data = tmp_path / "data"
    distill.generate(str(data), workers=1, minutes=0, iterations=50, shard_games=2)
    (shard,) = data.glob("*.npz")
    shutil.copy(shard, data / "z_copy.npz")  # two shards: one held out, one for training
    cfg = distill.DistillV2Config(
        data_dir=str(data),
        run_dir=str(tmp_path / "run"),
        max_minutes=0.05,
        eval_every_minutes=0.02,
        eval_deals=2,
        batch_size=16,
        val_every=2,
        warmup_steps=5,
        device=device,
        net=distill.NetConfigV2(state_hidden=32, state_out=16, action_out=16, head_hidden=32),
    )
    run = distill.train_v2(cfg, log=lambda r: None)
    results = json.loads((run / "results.json").read_text())
    assert results["status"] == "done"
    heads = results["val_loss_per_head"]
    assert set(heads) == {"actor", "seats", "prior", "score"}
    assert all(v is not None and v > 0 for v in heads.values()), heads
    assert results["val_decisions"] > 0 and results["decisions"] == 2 * results["val_decisions"]
    assert (run / "best.pt").exists() and (run / "latest.pt").exists()
    net, ckpt = distill.load_qnet_v2(run / "best.pt")
    assert ckpt["arch"] == "qnet_v2" and net.cfg.head_hidden == 32
    steps = torch.load(run / "latest.pt", weights_only=False)["info"]["steps"]
    resumed = replace(cfg, max_minutes=0.1, resume=str(run / "latest.pt"))
    distill.train_v2(resumed, log=lambda r: None)
    info = torch.load(run / "latest.pt", weights_only=False)["info"]
    assert info["steps"] > steps and len(info["history"]) >= 2
    with pytest.raises(ValueError, match="v2"):
        distill.load_qnet_v2(_v1_checkpoint(tmp_path / "v1.pt"))


def test_distill_v2_trains_on_several_folders(tmp_path: Path) -> None:
    """Task 17 round 2: ``extra_data_dirs`` adds earlier rounds' shards to training, and
    each folder holds out the same shards as when it is loaded alone (so round 1's
    held-out shards stay unseen in round 2)."""
    import json
    import shutil

    from big2.rl import distill

    seed = tmp_path / "seed"
    distill.generate(str(seed), workers=1, minutes=0, iterations=50, shard_games=1)
    (shard,) = seed.glob("*.npz")
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    for folder, n in ((r1, 3), (r2, 2)):
        folder.mkdir()
        for k in range(n):
            shutil.copy(shard, folder / f"s{k}.npz")
    with np.load(shard) as z:
        per_shard = len(z["states"])

    both = distill._DataV2([str(r2), str(r1)], 2, log=lambda m: None)
    alone = [distill._DataV2(str(f), 2, log=lambda m: None) for f in (r2, r1)]
    assert len(both.states) == 5 * per_shard
    assert both.val.tolist() == alone[0].val.tolist() + alone[1].val.tolist()
    # Held out: r2's s0, r1's s0 and s2 (every 2nd shard of each folder, from 0).
    assert int(both.val.sum()) == 3 * per_shard

    cfg = distill.DistillV2Config(
        data_dir=str(r2),
        extra_data_dirs=[str(r1)],
        run_dir=str(tmp_path / "run"),
        max_minutes=0.02,
        eval_every_minutes=0.01,
        eval_deals=2,
        batch_size=16,
        val_every=2,
        warmup_steps=5,
        device="cpu",
        net=distill.NetConfigV2(state_hidden=32, state_out=16, action_out=16, head_hidden=32),
    )
    run = distill.train_v2(cfg, log=lambda r: None)
    results = json.loads((run / "results.json").read_text())
    assert results["decisions"] == 5 * per_shard
    assert results["val_decisions"] == 3 * per_shard


def _v1_checkpoint(path: Path) -> Path:
    torch.save({"model": QNet().state_dict(), "net_config": QNet().config_dict()}, path)
    return path


@pytest.mark.parametrize("device", DEVICES)
def test_ppo_trains_and_its_checkpoint_plays(tmp_path: Path, device: str) -> None:
    from big2.rl.ppo import PPOConfig, segment_log_softmax
    from big2.rl.ppo import train as ppo_train

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
    first = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert "optimizer" in first and first["info"]["best_eval"] >= first["info"]["eval_score"]
    ppo_train(
        replace(cfg, run_dir=str(tmp_path / "more")),
        log=lambda r: None,
        resume=str(tmp_path / "latest.pt"),
    )
    more = torch.load(tmp_path / "more" / "latest.pt", weights_only=False)
    assert more["info"]["iter"] > first["info"]["iter"]
    assert more["info"]["games"] > first["info"]["games"]
    agent = make_agent(f"dmc:checkpoint={tmp_path / 'latest.pt'}")
    s = Rules().new_deal(seed=2)
    assert agent.act(s) in s.legal_actions()


def test_value_network_trains_exports_and_guides_search(tmp_path: Path) -> None:
    from big2._core import analyze, value_estimate
    from big2.rl.value import ValueConfig
    from big2.rl.value import train as value_train

    out = tmp_path / "v.pknn"
    cfg = ValueConfig(
        policy=str(Path(__file__).parents[2] / "models" / "big2-distilled.pt"),
        out=str(out),
        run_dir=str(tmp_path / "run"),
        envs=16,
        games_per_round=16,
        hidden=16,
        max_minutes=0.05,
    )
    if not Path(cfg.policy).exists():
        pytest.skip("models/big2-distilled.pt not present")
    value_train(cfg, log=lambda r: None)
    assert out.exists()
    # Training plays the policy on the GPU when there is one, but the cached copy that
    # load_model hands to the server and the agents must stay on the CPU.
    from big2.rl.agent import load_model

    assert all(p.device.type == "cpu" for p in load_model(cfg.policy).parameters())
    s = Rules().new_deal(seed=8)
    scores, wins = value_estimate(s, str(out))
    assert abs(sum(scores)) < 1e-3 and abs(sum(wins) - 1) < 1e-5
    for weight in (0.0, 0.5):
        r = analyze(s, iterations=200, value=str(out), value_rollout_weight=weight)
        assert sum(m["visits"] for m in r["moves"]) == 200
    with pytest.raises(ValueError, match="pknn"):
        analyze(s, iterations=10, value=str(tmp_path / "missing.pknn"))


def test_belief_search_runs_and_reports_ess() -> None:
    from big2._core import analyze

    s = Rules().new_deal(seed=6)
    for _ in range(14):
        s.apply(s.greedy_action())
    r = analyze(s, iterations=300, belief="greedy", belief_candidates=64)
    assert r["belief_ess"] is not None and 0 < r["belief_ess"] <= 64
    assert analyze(s, iterations=50)["belief_ess"] is None
    with pytest.raises(ValueError, match="belief"):
        analyze(s, iterations=10, belief="psychic")
    agent = make_agent("ismcts:iterations=100,belief=greedy,belief_candidates=32", seed=1)
    assert agent.act(s) in s.legal_actions()


def test_rules_belief_through_analyze() -> None:
    """Task 18: ``belief="rules"`` (the rules' hard constraints) reaches ISMCTS through
    ``analyze``, the agent spec and ``sample_deals``, and reports its rejection share."""
    from big2._core import analyze, sample_deals

    s = Rules().new_deal(seed=6)
    for _ in range(30):
        s.apply(s.greedy_action())
    me = s.current_player
    assert me is not None
    r = analyze(s, iterations=300, belief="rules", seed=2)
    assert r["belief_rejection_share"] is not None
    assert 0.0 <= r["belief_rejection_share"] < 1.0
    assert r["belief_fallbacks"] == 0
    assert sum(m["visits"] for m in r["moves"]) == 300
    plain = analyze(s, iterations=50)
    assert plain["belief_rejection_share"] is None and plain["belief_fallbacks"] is None
    deals = sample_deals(s, 20, belief="rules", seed=1)
    assert all(d[me] == s.hand(me) for d in deals)
    assert all([bin(h).count("1") for h in d] == s.hand_sizes for d in deals)
    with pytest.raises(ValueError, match="rules"):
        analyze(s, algorithm="pimc", iterations=10, belief="rules")
    agent = make_agent("ismcts:iterations=100,belief=rules", seed=1)
    assert agent.act(s) in s.legal_actions()


def test_infoset_endgame_through_analyze() -> None:
    """Task 20: ``infoset_endgame`` solves every deal at the root through ``analyze``
    and the agent spec, and reports ``exact`` and ``forced``."""
    from big2._core import Big2State, analyze, parse_cards

    hands = [parse_cards(h) for h in ("2s 5c 5d", "3d", "9h", "Kc")]
    s = Big2State.from_position(hands, 0)
    r = analyze(s, iterations=300, infoset_endgame=12)
    assert r["exact"] is True
    assert r["forced"] == (0, 5)  # seat 0 goes out in 5 moves whatever the others hold
    assert r["simulations"] == 6  # 3 hidden cards, one each: 3! deals
    assert r["moves"][0]["win_rate"] == 1.0
    plain = analyze(s, iterations=300)
    assert plain["exact"] is None and plain["simulations"] == 300
    # 13 hidden cards: above the threshold, the search runs as usual.
    t = Rules().new_deal(seed=4)
    for _ in range(8):
        t.apply(t.greedy_action())
    big = analyze(t, iterations=100, infoset_endgame=12)
    assert big["exact"] is None and big["simulations"] == 100
    with pytest.raises(ValueError, match="infoset_endgame"):
        analyze(s, algorithm="pimc", iterations=10, infoset_endgame=12)
    agent = make_agent("ismcts:iterations=100,infoset_endgame=12,belief=rules", seed=1)
    assert agent.act(s) in s.legal_actions()


def test_distill_data_skips_files_renamed_while_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generator renames ``*.tmp.npz`` to ``*.npz`` while the trainer lists the
    directory; a file that vanishes between listing and stat is retried later."""
    from big2.rl.distill import _Data

    owner = np.array([0, 0, 1], dtype=np.int64)
    np.savez(tmp_path / "a.npz", states=np.zeros((2, 3)), owner=owner)
    np.savez(tmp_path / "b.npz", states=np.zeros((2, 3)), owner=owner)
    real_stat = Path.stat

    def flaky_stat(self: Path, **kw: Any) -> Any:
        if self.name == "b.npz":
            raise FileNotFoundError(self)
        return real_stat(self, **kw)

    data = _Data()
    with monkeypatch.context() as m:
        m.setattr(Path, "stat", flaky_stat)
        assert data.refresh(str(tmp_path), 10**6) == 1
    assert data.refresh(str(tmp_path), 10**6) == 1  # picked up next time
    assert data.decisions() == 4

"""The Rust Q-network (crates/big2/src/qnet.rs) against PyTorch, through the exported file.

The Rust side runs as the ``qnet_dump`` example (built with cargo here), which plays
seeded games, scores every decision and writes the features and scores; PyTorch
then scores the same features. No change to the Python extension is needed.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from big2.rl.export import (  # noqa: E402
    dequantize_groups,
    export_pknn,
    export_qnet_pknn,
    load_pvnet,
    load_qnet,
    quantize_groups,
)
from big2.rl.model import (  # noqa: E402
    ACTION_LEN,
    SCORE_SCALE,
    STATE_LEN,
    NetConfig,
    NetConfigV2,
    QNet,
    QNetV2,
)

ROOT = Path(__file__).parents[2]
DUTY_MODEL = ROOT / "models" / "big2-distilled-duty.pt"
V2_MODEL = ROOT / "runs" / "net_r0" / "best.pt"


@pytest.fixture(scope="module")
def qnet_dump() -> Path:
    """Builds the ``qnet_dump`` example (release) and returns its executable."""
    if shutil.which("cargo") is None:
        pytest.skip("cargo not on PATH")
    r = subprocess.run(
        ["cargo", "build", "--release", "-p", "big2-big2", "--example", "qnet_dump"]
        + ["--message-format=json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in r.stdout.splitlines():
        msg = json.loads(line)
        if msg.get("reason") == "compiler-artifact" and msg["target"]["name"] == "qnet_dump":
            return Path(msg["executable"])
    raise AssertionError("cargo built no qnet_dump executable")


def rust_scores(
    exe: Path, pknn: Path, out: Path, positions: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Runs the dump; returns per position ``(state, actions, scores, win_logits)``."""
    subprocess.run([str(exe), str(pknn), str(out), str(positions), str(seed)], check=True)
    b = out.read_bytes()
    assert b[:4] == b"QDMP"
    state_len, action_len, count = struct.unpack_from("<III", b, 4)
    assert (state_len, action_len, count) == (STATE_LEN, ACTION_LEN, positions)
    pos, rows = 16, []
    for _ in range(count):
        (n,) = struct.unpack_from("<I", b, pos)
        pos += 4
        st = np.frombuffer(b, np.uint8, state_len, pos)
        pos += state_len
        ac = np.frombuffer(b, np.uint8, n * action_len, pos).reshape(n, action_len)
        pos += n * action_len
        sc = np.frombuffer(b, "<f4", n, pos)
        pos += 4 * n
        wl = np.frombuffer(b, "<f4", n, pos)
        pos += 4 * n
        rows.append((st, ac, sc, wl))
    assert pos == len(b)
    return rows


@torch.no_grad()
def compare(net: QNet, rows: list) -> dict[str, float]:
    """Move agreement (same best move) and RMS score difference in points."""
    states = torch.from_numpy(np.stack([r[0] for r in rows]).astype(np.float32))
    actions = torch.from_numpy(np.concatenate([r[1] for r in rows]).astype(np.float32))
    index = torch.from_numpy(np.repeat(np.arange(len(rows)), [len(r[1]) for r in rows]))
    score, win = net(states, actions, index)
    score = score.numpy() * SCORE_SCALE
    win = win.numpy()
    agree, sq, wsq, lo = 0, 0.0, 0.0, 0
    for _, ac, rs, rw in rows:
        hi = lo + len(ac)
        agree += int(np.argmax(score[lo:hi]) == np.argmax(rs))
        sq += float(((score[lo:hi] - rs) ** 2).sum())
        wsq += float(((win[lo:hi] - rw) ** 2).sum())
        lo = hi
    return {
        "agreement": agree / len(rows),
        "rms_points": (sq / lo) ** 0.5,
        "rms_win_logit": (wsq / lo) ** 0.5,
    }


def small_checkpoint(path: Path) -> Path:
    torch.manual_seed(0)
    cfg = NetConfig(state_hidden=32, state_out=16, action_out=16, head_hidden=32)
    net = QNet(cfg)
    torch.save({"net_config": asdict(cfg), "model": net.state_dict()}, path)
    return path


def test_quantize_groups_is_symmetric_per_group() -> None:
    w = np.array(
        [[1.0, -0.5, 0.0, 0.25], [0.0, 0.0, 0.0, 0.0], [-2.54, 0.02, 1.27, -0.01]],
        dtype=np.float32,
    )
    scales, q = quantize_groups(w, 4)  # one group per row
    assert scales.dtype == np.float16 and scales.shape == (3, 1)
    np.testing.assert_allclose(scales[:, 0], [1 / 127, 1.0, 2.54 / 127], rtol=1e-3)
    # 1.27 is half the row's peak: 63.5 steps, so 63 or 64 after the scale's f16 rounding.
    assert q.dtype == np.int8 and q[0, 0] == 127 and q[2, 0] == -127 and q[2, 2] in (63, 64)
    back = dequantize_groups(scales, q, 4)
    assert np.abs(back - w).max() <= float(scales.max()) / 2 + 1e-6
    scales, q = quantize_groups(w, 3)  # two groups per row, the second of one weight
    assert scales.shape == (3, 2) and q[0, 3] == 127 and q[2, 3] == -127
    assert np.abs(dequantize_groups(scales, q, 3) - w).max() < 0.011


def test_rust_matches_pytorch_on_a_small_network(qnet_dump: Path, tmp_path: Path) -> None:
    ckpt = small_checkpoint(tmp_path / "small.pt")
    net, _ = load_qnet(ckpt)
    exact = export_qnet_pknn(ckpt, tmp_path / "small-f32.pknn", precision="f32")
    m = compare(net, rust_scores(qnet_dump, exact, tmp_path / "f32.bin", 200, 1))
    assert m["agreement"] == 1.0 and m["rms_points"] < 1e-3, m
    mixed = export_qnet_pknn(ckpt, tmp_path / "small-mixed.pknn")
    assert mixed.stat().st_size < exact.stat().st_size / 2.5
    m = compare(net, rust_scores(qnet_dump, mixed, tmp_path / "mixed.bin", 200, 1))
    assert m["agreement"] >= 0.95, m


def test_ppo_checkpoints_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "ppo.pt"
    torch.save({"arch": "ppo", "net_config": {}, "model": {}}, path)
    with pytest.raises(ValueError, match="PPO"):
        export_qnet_pknn(path, tmp_path / "x.pknn")


@pytest.fixture(scope="module")
def pvnet_dump() -> Path:
    """Builds the ``pvnet_dump`` example (release) and returns its executable."""
    if shutil.which("cargo") is None:
        pytest.skip("cargo not on PATH")
    r = subprocess.run(
        ["cargo", "build", "--release", "-p", "big2-big2", "--example", "pvnet_dump"]
        + ["--message-format=json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in r.stdout.splitlines():
        msg = json.loads(line)
        if msg.get("reason") == "compiler-artifact" and msg["target"]["name"] == "pvnet_dump":
            return Path(msg["executable"])
    raise AssertionError("cargo built no pvnet_dump executable")


def rust_pvnet(
    exe: Path, pknn: Path, out: Path, positions: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Runs the v2 dump; returns per position ``(state, actions, prior, dist [n, 4, 14])``."""
    subprocess.run([str(exe), str(pknn), str(out), str(positions), str(seed)], check=True)
    b = out.read_bytes()
    assert b[:4] == b"PDMP"
    state_len, action_len, count = struct.unpack_from("<III", b, 4)
    assert (state_len, action_len, count) == (STATE_LEN, ACTION_LEN, positions)
    pos, rows = 16, []
    for _ in range(count):
        (n,) = struct.unpack_from("<I", b, pos)
        pos += 4
        st = np.frombuffer(b, np.uint8, state_len, pos)
        pos += state_len
        ac = np.frombuffer(b, np.uint8, n * action_len, pos).reshape(n, action_len)
        pos += n * action_len
        prior = np.frombuffer(b, "<f4", n, pos)
        pos += 4 * n
        dist = np.frombuffer(b, "<f4", n * 56, pos).reshape(n, 4, 14)
        pos += 4 * n * 56
        rows.append((st, ac, prior, dist))
    assert pos == len(b)
    return rows


@torch.no_grad()
def compare_v2(net: QNetV2, rows: list) -> dict[str, float]:
    """Prior argmax agreement, and the L1 distance of each candidate's per-seat
    distributions (mean, 99th percentile and max over every distribution)."""
    states = torch.from_numpy(np.stack([r[0] for r in rows]).astype(np.float32))
    actions = torch.from_numpy(np.concatenate([r[1] for r in rows]).astype(np.float32))
    index = torch.from_numpy(np.repeat(np.arange(len(rows)), [len(r[1]) for r in rows]))
    dist_logits, prior_logit = net(states, actions, index)
    dist = dist_logits.softmax(-1).numpy()
    prior_logit = prior_logit.numpy()
    agree, lo, l1 = 0, 0, []
    for _, ac, rp, rd in rows:
        hi = lo + len(ac)
        agree += int(np.argmax(prior_logit[lo:hi]) == np.argmax(rp))
        l1.append(np.abs(dist[lo:hi] - rd).sum(axis=-1).ravel())
        lo = hi
    d = np.concatenate(l1)
    return {
        "prior_agreement": agree / len(rows),
        "dist_l1_mean": float(d.mean()),
        "dist_l1_p99": float(np.quantile(d, 0.99)),
        "dist_l1_max": float(d.max()),
    }


def small_v2_checkpoint(path: Path) -> Path:
    torch.manual_seed(0)
    cfg = NetConfigV2(state_hidden=32, state_out=16, action_out=16, head_hidden=32)
    net = QNetV2(cfg)
    torch.save({"arch": "qnet_v2", "net_config": asdict(cfg), "model": net.state_dict()}, path)
    return path


def test_v2_rust_matches_pytorch_on_a_small_network(pvnet_dump: Path, tmp_path: Path) -> None:
    ckpt = small_v2_checkpoint(tmp_path / "small-v2.pt")
    net, _ = load_pvnet(ckpt)
    exact = export_pknn(ckpt, tmp_path / "small-v2-f32.pknn", precision="f32")
    m = compare_v2(net, rust_pvnet(pvnet_dump, exact, tmp_path / "f32.bin", 200, 1))
    assert m["prior_agreement"] == 1.0 and m["dist_l1_max"] < 1e-4, m
    mixed = export_pknn(ckpt, tmp_path / "small-v2-mixed.pknn")
    assert mixed.stat().st_size < exact.stat().st_size / 2.5
    m = compare_v2(net, rust_pvnet(pvnet_dump, mixed, tmp_path / "mixed.bin", 200, 1))
    assert m["prior_agreement"] >= 0.95 and m["dist_l1_mean"] < 0.02, m
    with pytest.raises(ValueError, match="v2"):
        load_qnet(ckpt)


def test_v2_trained_network_in_rust_matches_pytorch(pvnet_dump: Path, tmp_path: Path) -> None:
    """Plan Task 13: prior argmax agrees on >= 99 % and the distributions are within
    0.01 L1 (mean over every candidate and seat) on 1,000 positions."""
    if not V2_MODEL.exists():
        pytest.skip("runs/net_r0/best.pt not present (Task 12's training run)")
    out = export_pknn(V2_MODEL, tmp_path / "pvnet.pknn")
    assert out.stat().st_size < 1.2e6
    net, _ = load_pvnet(V2_MODEL)
    m = compare_v2(net, rust_pvnet(pvnet_dump, out, tmp_path / "pvnet.bin", 1000, 0))
    print(f"\nRust (mixed int8/f16) vs PyTorch, network v2, 1,000 positions: {m}")
    assert m["prior_agreement"] >= 0.99, m
    assert m["dist_l1_mean"] < 0.01, m


def test_distilled_duty_network_in_rust_matches_pytorch(qnet_dump: Path, tmp_path: Path) -> None:
    """Plan Task 7 / A6: >= 99 % same move and < 0.05 points RMS on 1,000 positions."""
    if not DUTY_MODEL.exists():
        pytest.skip("models/big2-distilled-duty.pt not present")
    out = export_qnet_pknn(DUTY_MODEL, tmp_path / "duty.pknn")
    assert out.stat().st_size < 1.2e6
    net, _ = load_qnet(DUTY_MODEL)
    m = compare(net, rust_scores(qnet_dump, out, tmp_path / "duty.bin", 1000, 0))
    print(f"\nRust (mixed int8/f16) vs PyTorch on 1,000 positions: {m}")
    assert m["agreement"] >= 0.99, m
    assert m["rms_points"] < 0.05, m


def test_puct_leaf_network_weight_through_analyze(tmp_path: Path) -> None:
    """Task 15 fallback: ``leaf_network_weight`` reaches PUCT through ``analyze`` and the
    ``puct:`` agent spec; 1 (the default) gives the network-only search exactly."""
    from big2._core import Big2State, analyze
    from big2.agents import make_agent

    pknn = export_pknn(
        small_v2_checkpoint(tmp_path / "small-v2.pt"), tmp_path / "v2.pknn", precision="f32"
    )
    state = Big2State.new_deal(3)
    kw = {"algorithm": "puct", "iterations": 200, "seed": 5, "net": str(pknn)}
    plain = analyze(state, **kw)
    assert analyze(state, **kw, leaf_network_weight=1.0)["moves"] == plain["moves"]
    mixed = analyze(state, **kw, leaf_network_weight=0.5, rollout_moves=0)
    assert mixed["simulations"] == 200
    assert sum(m["visits"] for m in mixed["moves"]) == 200
    assert mixed["moves"] != plain["moves"]
    short = analyze(state, **kw, leaf_network_weight=0.0, rollout_moves=4, rollout="structured")
    assert short["simulations"] == 200
    with pytest.raises(ValueError, match="between 0 and 1"):
        analyze(state, **kw, leaf_network_weight=1.5)
    with pytest.raises(ValueError, match="only supported with puct"):
        analyze(state, iterations=10, leaf_network_weight=0.5)
    agent = make_agent(f"puct:net={pknn},simulations=50,leaf_network_weight=0.5,c_puct=2.5")
    legal = state.legal_actions()
    assert agent.act(state) in legal
    # Task 18: the rules mask as PUCT's sampler.
    masked = analyze(state, **kw, belief="rules")
    assert masked["simulations"] == 200 and masked["belief_fallbacks"] == 0
    assert masked["belief_rejection_share"] is not None
    agent = make_agent(f"puct:net={pknn},simulations=50,belief=rules")
    assert agent.act(state) in legal

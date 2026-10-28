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
    export_qnet_pknn,
    load_qnet,
    quantize_groups,
)
from big2.rl.model import ACTION_LEN, SCORE_SCALE, STATE_LEN, NetConfig, QNet  # noqa: E402

ROOT = Path(__file__).parents[2]
DUTY_MODEL = ROOT / "models" / "big2-distilled-duty.pt"


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

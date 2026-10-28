"""Export a trained Q-network to the ``.pknn`` version 2 file the Rust engine reads.

The Rust side is ``crates/big2/src/qnet.rs`` (``QNet::from_bytes``); the format is
documented in ``crates/nn/src/lib.rs``. One file holds three towers of the
:class:`big2.rl.model.QNet`: ``state``, ``action`` and ``head`` (the ReLUs that end
the two towers are applied by the Rust code, not stored).

**Precision ("mixed", the default).** The state tower's first layer (two thirds of
the weights) is stored as int8 with one f16 scale per 32 weights of a row; every
other layer as f16; biases as f32. The default network then takes about 1.16 MB
(f32: 3.4 MB). Rounding the first layer shifts each hidden unit's average input,
so the exporter plays a few self-play games with the network and adds that shift
back into the layer's bias ("bias correction"), measured on those positions.
Per-row int8 everywhere (0.85 MB) was tried first: its scores were 0.75 points RMS
from PyTorch's and it picked a different move in 7 % of positions.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from big2.rl.model import ACTION_LEN, SCORE_SCALE, STATE_LEN, NetConfig, QNet

QNET_KIND = "big2-qnet"
QNET_FEATURES = "big2-state-v1"
"""Must match ``QNET_KIND`` / ``QNET_FEATURES`` in ``crates/big2/src/qnet.rs``."""

F32, INT8, F16 = 0, 1, 2
"""Layer precision codes of the file format."""
INT8_GROUP = 32
"""Weights per int8 scale in the mixed format."""


def load_qnet(checkpoint: str | Path) -> tuple[QNet, dict[str, Any]]:
    """Loads a Q-network checkpoint (DMC or distilled) on the CPU, without caching it.

    Raises ``ValueError`` for a PPO checkpoint, whose first output is a logit, not
    a score.
    """
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if ckpt.get("arch") == "ppo":
        raise ValueError("a PPO policy is not a Q-network; export a DMC or distilled checkpoint")
    net = QNet(NetConfig(**ckpt["net_config"]))
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, ckpt


def quantize_groups(w: np.ndarray, group: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 in runs of ``group`` weights along each row.

    ``scale = max|w| / 127`` per run, rounded to f16 (1 if that is 0), then
    ``q = round(w / scale)``. Returns ``(scales [n_out, ceil(n_in / group)] float16,
    q [n_out, n_in] int8)``. The same rule as ``big2_nn::quantize_groups`` in Rust.
    """
    w = w.astype(np.float32)
    n_out, n_in = w.shape
    per_row = -(-n_in // group)
    scales = np.empty((n_out, per_row), dtype=np.float16)
    q = np.empty((n_out, n_in), dtype=np.int8)
    for j in range(per_row):
        blk = w[:, j * group : (j + 1) * group]
        s = (np.abs(blk).max(axis=1) / np.float32(127.0)).astype(np.float16)
        s[s == 0] = 1.0
        s32 = s.astype(np.float32)
        scales[:, j] = s
        q[:, j * group : (j + 1) * group] = np.clip(np.rint(blk / s32[:, None]), -127, 127)
    return scales, q


def dequantize_groups(scales: np.ndarray, q: np.ndarray, group: int) -> np.ndarray:
    """The weights a reader gets back from :func:`quantize_groups`."""
    s = np.repeat(scales.astype(np.float32), group, axis=1)[:, : q.shape[1]]
    return (q.astype(np.float32) * s).astype(np.float32)


def stored_weight(w: np.ndarray, precision: int) -> np.ndarray:
    """The f32 weights the Rust reader will see for ``w`` stored at ``precision``."""
    if precision == F32:
        return w.astype(np.float32)
    if precision == F16:
        return w.astype(np.float16).astype(np.float32)
    return dequantize_groups(*quantize_groups(w, INT8_GROUP), INT8_GROUP)


def _linears(seq: nn.Sequential) -> list[nn.Linear]:
    return [m for m in seq if isinstance(m, nn.Linear)]


Layer = tuple[nn.Linear, int, np.ndarray | None]
"""One layer to write: the linear layer, its precision code, and a bias to store
instead of its own (or ``None``)."""


def write_pknn_v2(
    path: str | Path, towers: list[tuple[str, list[Layer]]], meta: dict[str, Any]
) -> None:
    """Writes named towers as a ``.pknn`` version 2 file.

    Each layer is ``(linear, precision, bias)``; ``bias`` replaces the layer's own
    bias when given (bias correction).
    """
    text = "\n".join(f"{k}={v}" for k, v in meta.items()).encode()
    out = bytearray(b"PKNN" + struct.pack("<II", 2, len(text)) + text)
    out += struct.pack("<I", len(towers))
    for name, layers in towers:
        raw = name.encode()
        out += struct.pack("<I", len(raw)) + raw + struct.pack("<I", len(layers))
        for lin, precision, bias in layers:
            w = lin.weight.detach().cpu().numpy().astype(np.float32)
            b = lin.bias.detach().cpu().numpy() if bias is None else bias
            out += struct.pack("<IIB", lin.in_features, lin.out_features, precision)
            if precision == INT8:
                scales, q = quantize_groups(w, INT8_GROUP)
                out += struct.pack("<I", INT8_GROUP)
                out += scales.astype("<f2").tobytes(order="C") + q.tobytes(order="C")
            elif precision == F16:
                out += w.astype("<f2").tobytes(order="C")
            else:
                out += w.astype("<f4").tobytes(order="C")
            out += np.asarray(b, dtype="<f4").tobytes()
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_bytes(bytes(out))
    tmp.replace(path)


@torch.no_grad()
def calibration_states(net: QNet, positions: int = 2000, seed: int = 12345) -> np.ndarray:
    """State features of ``positions`` decisions from self-play with ``net`` (5 % random
    moves), under the default rules. Used to measure the bias correction."""
    from big2._core import Big2VecEnv
    from big2.rl.dmc import choose

    env = Big2VecEnv(64, seed=seed)
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    while sum(len(r) for r in rows) < positions:
        st, ac, counts_l, ids, _ = env.observe()
        states = np.frombuffer(st, dtype=np.uint8).reshape(len(ids), STATE_LEN)
        actions = np.frombuffer(ac, dtype=np.uint8).reshape(-1, ACTION_LEN)
        counts = np.asarray(counts_l, dtype=np.int64)
        rows.append(states.copy())
        env.step(choose(net, states, actions, counts, 0.05, rng).tolist())
    return np.concatenate(rows)[:positions]


def export_qnet_pknn(checkpoint: str | Path, out: str | Path, precision: str = "mixed") -> Path:
    """Exports a Q-network checkpoint to ``out`` (``.pknn`` version 2); returns ``out``.

    ``precision`` is ``"mixed"`` (see the module docs) or ``"f32"`` (exact, about 3×
    larger). The metadata records the kind, the feature encoding, the score scale,
    the precision, the source checkpoint's file name and the rules it was trained
    under.
    """
    if precision not in ("mixed", "f32"):
        raise ValueError(f"precision must be 'mixed' or 'f32', not {precision!r}")
    net, ckpt = load_qnet(checkpoint)
    rules = (ckpt.get("train_config") or {}).get("rules") or {}
    meta: dict[str, Any] = {
        "kind": QNET_KIND,
        "features": QNET_FEATURES,
        "score_scale": SCORE_SCALE,
        "precision": precision,
        "source": Path(checkpoint).name,
        # "default" means the trainer ran with its default rules (no overrides).
        "rules": ";".join(f"{k}:{v}" for k, v in sorted(rules.items())) or "default",
    }
    mixed = precision == "mixed"
    towers: list[tuple[str, list[Layer]]] = [
        (name, [(lin, F16 if mixed else F32, None) for lin in _linears(seq)])
        for name, seq in (
            ("state", net.state_tower),
            ("action", net.action_tower),
            ("head", net.head),
        )
    ]
    if mixed:
        first = towers[0][1][0][0]
        w = first.weight.detach().numpy().astype(np.float32)
        mean_x = calibration_states(net).astype(np.float32).mean(axis=0)
        # Keep each hidden unit's average input unchanged after rounding the weights.
        bias = first.bias.detach().numpy() - (stored_weight(w, INT8) - w) @ mean_x
        towers[0][1][0] = (first, INT8, bias)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_pknn_v2(out, towers, meta)
    return out

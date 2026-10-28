"""Play with an exported Q-network (``.pknn`` version 2), in NumPy.

The shipped engine runs the same file through ``big2_big2::qnet::QNetPolicy`` in
Rust. That policy is not exposed through the Python extension, so this module reads
the file itself and repeats the Rust forward pass and sampling rule:

* the weights are the ones the Rust reader sees (int8 groups and f16 rounded the
  same way), so the scores match Rust's to float precision, not PyTorch's;
* ``temperature`` 0 plays the best predicted score; otherwise a move is drawn with
  probability proportional to ``exp(score / temperature)``, ``temperature`` in points.

No PyTorch is needed. The file format is documented in ``crates/nn/src/lib.rs`` and
written by :mod:`big2.rl.export`.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from big2._core import Big2State, Big2VecEnv, encode

STATE_LEN, ACTION_LEN = Big2VecEnv.feature_sizes()
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QNET = ROOT / "models" / "big2-qnet-duty.pknn"
"""The distilled duty-on network exported by Task 7 (``big2 rl export``)."""

_F32, _INT8, _F16 = 0, 1, 2

Layer = tuple[np.ndarray, np.ndarray]
"""``(weight [n_out, n_in] float32, bias [n_out] float32)``."""


def read_pknn_v2(path: str | Path) -> tuple[dict[str, str], dict[str, list[Layer]]]:
    """Reads a ``.pknn`` version 2 file: ``(metadata, {tower name: layers})``.

    Weights come back as the f32 values the Rust reader computes (int8 times its
    group scale, or f16 widened).
    """
    b = Path(path).read_bytes()
    if b[:4] != b"PKNN":
        raise ValueError(f"{path}: not a .pknn file")
    version, meta_len = struct.unpack_from("<II", b, 4)
    if version != 2:
        raise ValueError(f"{path}: .pknn version {version}, expected 2")
    pos = 12
    meta = dict(
        line.split("=", 1) for line in b[pos : pos + meta_len].decode().splitlines() if line
    )
    pos += meta_len
    (n_towers,) = struct.unpack_from("<I", b, pos)
    pos += 4
    towers: dict[str, list[Layer]] = {}
    for _ in range(n_towers):
        (name_len,) = struct.unpack_from("<I", b, pos)
        pos += 4
        name = b[pos : pos + name_len].decode()
        pos += name_len
        (n_layers,) = struct.unpack_from("<I", b, pos)
        pos += 4
        layers: list[Layer] = []
        for _ in range(n_layers):
            n_in, n_out, precision = struct.unpack_from("<IIB", b, pos)
            pos += 9
            if precision == _INT8:
                (group,) = struct.unpack_from("<I", b, pos)
                pos += 4
                per_row = -(-n_in // group)
                scales = np.frombuffer(b, "<f2", n_out * per_row, pos).reshape(n_out, per_row)
                pos += 2 * n_out * per_row
                q = np.frombuffer(b, np.int8, n_out * n_in, pos).reshape(n_out, n_in)
                pos += n_out * n_in
                s = np.repeat(scales.astype(np.float32), group, axis=1)[:, :n_in]
                w = q.astype(np.float32) * s
            elif precision == _F16:
                w = np.frombuffer(b, "<f2", n_out * n_in, pos).reshape(n_out, n_in)
                w = w.astype(np.float32)
                pos += 2 * n_out * n_in
            elif precision == _F32:
                w = np.frombuffer(b, "<f4", n_out * n_in, pos).reshape(n_out, n_in).copy()
                pos += 4 * n_out * n_in
            else:
                raise ValueError(f"{path}: unknown precision code {precision}")
            bias = np.frombuffer(b, "<f4", n_out, pos).copy()
            pos += 4 * n_out
            layers.append((np.ascontiguousarray(w.T), bias))  # stored transposed for x @ w
        towers[name] = layers
    if pos != len(b):
        raise ValueError(f"{path}: {len(b) - pos} bytes left after the last tower")
    return meta, towers


def _mlp(x: np.ndarray, layers: list[Layer], relu_last: bool) -> np.ndarray:
    for i, (w, bias) in enumerate(layers):
        x = x @ w + bias
        if i < len(layers) - 1 or relu_last:
            np.maximum(x, 0.0, out=x)
    return x


class NumpyQNet:
    """The Q-network of a ``.pknn`` v2 file (towers ``state``, ``action``, ``head``)."""

    def __init__(self, path: str | Path) -> None:
        self.meta, towers = read_pknn_v2(path)
        if self.meta.get("kind") != "big2-qnet":
            raise ValueError(f"{path}: kind {self.meta.get('kind')!r}, expected 'big2-qnet'")
        self.state = towers["state"]
        self.action = towers["action"]
        self.head = towers["head"]
        self.score_scale = float(self.meta.get("score_scale", 50.0))

    def score_moves(self, state: Big2State) -> tuple[list[int], np.ndarray]:
        """``(legal moves, predicted final score of each in points)`` for the player to act."""
        s, a, legal = encode(state)
        st = np.frombuffer(s, np.uint8).astype(np.float32)[None, :]
        ac = np.frombuffer(a, np.uint8).reshape(-1, ACTION_LEN).astype(np.float32)
        hand_left = st[:, :52] - ac[:, :52]  # as QNet.forward in big2.rl.model
        h = _mlp(st, self.state, relu_last=True)
        m = _mlp(np.concatenate([ac, hand_left], axis=1), self.action, relu_last=True)
        x = np.concatenate([np.repeat(h, len(legal), axis=0), m], axis=1)
        out = _mlp(x, self.head, relu_last=False)
        return legal, out[:, 0] * self.score_scale


_NETS: dict[str, NumpyQNet] = {}


def load_numpy_qnet(path: str | Path) -> NumpyQNet:
    """Loads (and caches per process) a ``.pknn`` Q-network."""
    key = str(Path(path).resolve())
    if key not in _NETS:
        _NETS[key] = NumpyQNet(key)
    return _NETS[key]


@dataclass
class QNetAgent:
    """Plays with an exported Q-network, as ``QNetPolicy { temperature }`` in Rust."""

    path: str = str(DEFAULT_QNET)
    temperature: float = 1.0
    seed: int | None = None
    name: str = "qnet"
    net: NumpyQNet = field(init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        p = Path(self.path)
        if not p.is_absolute() and not p.exists():
            p = ROOT / p  # paths in specs are relative to the repo root
        self.net = load_numpy_qnet(p)
        self._rng = random.Random(self.seed)

    def act(self, state: Big2State) -> int:
        legal, scores = self.net.score_moves(state)
        if len(legal) == 1:
            return int(legal[0])
        best = float(scores.max())
        if self.temperature <= 0.0:
            return int(legal[int(np.argmax(scores))])
        weights = np.exp((scores.astype(np.float64) - best) / self.temperature)
        x = self._rng.random() * float(weights.sum())
        for move, w in zip(legal, weights, strict=True):
            if x < w:
                return int(move)
            x -= float(w)
        return int(legal[-1])

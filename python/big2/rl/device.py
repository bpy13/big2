"""Choosing where the networks run: CPU, an NVIDIA GPU (CUDA) or Apple's GPU (MPS).

Training has two kinds of network work:

* **Learning** (gradient steps on minibatches of thousands of samples) — runs on
  the chosen ``device``.
* **Acting** (choosing moves for the self-play games and evaluation games, one
  batched call per step of the vectorized environment) — runs on
  :func:`actor_device`: the GPU on CUDA; the CPU otherwise, because on Apple's
  MPS the per-call overhead outweighs the speed-up for these small batches.
"""

from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> str:
    """Returns a usable torch device name.

    ``"auto"`` picks ``"cuda"`` when an NVIDIA GPU is usable and ``"cpu"``
    otherwise (MPS is used only when asked for). ``"cuda"``, ``"cuda:N"``,
    ``"mps"`` and ``"cpu"`` are checked and returned as given.
    """
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            "device 'cuda' requested but PyTorch sees no CUDA GPU; install a CUDA build "
            "of PyTorch (see docs/rl.md, 'Hardware') or use --device cpu"
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device 'mps' requested but MPS is not available on this machine")
    if name != "cpu" and not name.startswith(("cuda", "mps")):
        raise ValueError(f"unknown device {name!r}: use auto, cpu, cuda or mps")
    return name


def actor_device(device: str) -> str:
    """The device for move selection during self-play and evaluation (see module docs)."""
    return device if device.startswith("cuda") else "cpu"

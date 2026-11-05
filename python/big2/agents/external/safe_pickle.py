"""Load Charlesworth's two pickle files without running any code from them.

``actionIndices.pkl`` is a plain NumPy pickle. ``modelParameters136500`` is a joblib
file: a pickle stream with each array's raw bytes written in between. We do not
install joblib. Instead :class:`_Wrapper` stands in for ``joblib``'s array wrapper
and reads those bytes itself. Everything else the stream names is refused: a pickle
that names ``os.system`` (or anything not on the list below) raises
:class:`UnsafePickle` before the call happens.
"""

from __future__ import annotations

import io
import pickle
from typing import Any, BinaryIO

import numpy as np

#: The only ``(module, name)`` pairs a stream may import. Both spellings of NumPy's
#: reconstruct function exist (NumPy 1 writes ``numpy.core``, NumPy 2 ``numpy._core``).
ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("builtins", "list"),
    ("builtins", "dict"),
    ("builtins", "tuple"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("collections", "OrderedDict"),
}
# NumPy 2 moved ``numpy.core`` to ``numpy._core``; the stream may use either name.
_RECONSTRUCT = (
    np._core.multiarray._reconstruct  # type: ignore[attr-defined]
    if hasattr(np, "_core")
    else np.core.multiarray._reconstruct
)
#: joblib's array wrapper, replaced by our own reader.
JOBLIB_WRAPPER = ("joblib.numpy_pickle", "NumpyArrayWrapper")


class UnsafePickle(pickle.UnpicklingError):
    """The stream asked for a class or function that is not on the allow list."""


class _Wrapper:
    """Reads one joblib array: the state names dtype and shape, the bytes follow."""

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.state = state

    def read(self, file: BinaryIO) -> np.ndarray:
        dtype = np.dtype(self.state["dtype"])
        if dtype.hasobject:
            raise UnsafePickle("object arrays are refused")
        shape = tuple(self.state["shape"])
        count = int(np.prod(shape)) if shape else 1
        if self.state.get("numpy_array_alignment_bytes") is not None:
            pad = file.read(1)[0]
            file.read(pad)
        raw = file.read(count * dtype.itemsize)
        if len(raw) != count * dtype.itemsize:
            raise pickle.UnpicklingError("array data is cut short")
        array = np.frombuffer(raw, dtype=dtype).copy()
        if self.state.get("order") == "F":
            return array.reshape(shape[::-1]).transpose()
        return array.reshape(shape)


class RestrictedUnpickler(pickle._Unpickler):  # pure-Python class: BUILD can be hooked
    """Unpickler whose ``find_class`` only allows :data:`ALLOWED`."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in ALLOWED:
            if module == "builtins":
                return getattr(__import__("builtins"), name)
            if module == "collections":
                import collections

                return collections.OrderedDict
            if name == "_reconstruct":
                return _RECONSTRUCT
            return getattr(np, name)
        if (module, name) == JOBLIB_WRAPPER:
            return _Wrapper
        raise UnsafePickle(f"refused to import {module}.{name}")

    def load_build(self) -> None:
        super().load_build()
        if isinstance(self.stack[-1], _Wrapper):
            self.stack[-1] = self.stack[-1].read(self._file_handle)

    dispatch = dict(pickle._Unpickler.dispatch)
    dispatch[pickle.BUILD[0]] = load_build  # type: ignore[assignment]

    def __init__(self, file: BinaryIO) -> None:
        super().__init__(file)
        self._file_handle = file


def load(path: str) -> Any:
    """Load a pickle (or joblib file) from ``path`` with the restricted rules."""
    with open(path, "rb") as f:
        return RestrictedUnpickler(f).load()


def loads(data: bytes) -> Any:
    """Load from bytes; used by the tests."""
    return RestrictedUnpickler(io.BytesIO(data)).load()

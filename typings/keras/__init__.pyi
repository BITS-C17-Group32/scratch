"""Minimal hand-written type stubs for the keras API this project uses.

Keras 3 ships no inline types and no stub package exists on PyPI
(`keras-stubs` returns 404, checked 2026-11), so per pyright's docs
these cover exactly the surface the code uses; the runtime classes are
the same objects as tf.keras's (probe-verified). Extend a file when new
API is used rather than loosening types.
"""

from collections.abc import Sequence

import numpy as np
from keras import callbacks as callbacks, layers as layers, utils as utils

class Model:
    """The keras functional model, typed for this project's usage."""

    def __init__(
        self,
        inputs: layers.KerasTensor | Sequence[layers.KerasTensor],
        outputs: layers.KerasTensor | Sequence[layers.KerasTensor],
    ) -> None: ...
    def compile(
        self,
        optimizer: str = ...,
        loss: str = ...,
        metrics: Sequence[str] = ...,
    ) -> None: ...
    def fit(
        self,
        x: Sequence[np.ndarray],
        y: np.ndarray,
        validation_split: float = ...,
        epochs: int = ...,
        batch_size: int = ...,
        shuffle: bool = ...,
        verbose: int = ...,
        callbacks: Sequence[callbacks.Callback] = ...,
    ) -> None: ...
    def predict(
        self,
        x: Sequence[np.ndarray],
        verbose: int = ...,
    ) -> np.ndarray: ...

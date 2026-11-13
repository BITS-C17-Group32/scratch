"""Minimal hand-written type stubs for keras.layers (see __init__.pyi)."""

from collections.abc import Sequence

class KerasTensor:
    """A symbolic tensor: the input/output currency between layers."""

def Input(
    shape: Sequence[int | None] | None = ...,
    name: str | None = ...,
) -> KerasTensor: ...

class Layer:
    def __call__(self, inputs: KerasTensor | Sequence[KerasTensor]) -> KerasTensor: ...

class Dense(Layer):
    def __init__(self, units: int) -> None: ...

class GRU(Layer):
    def __init__(self, units: int) -> None: ...

class LSTM(Layer):
    def __init__(self, units: int, return_sequences: bool = ...) -> None: ...

class Dropout(Layer):
    def __init__(self, rate: float) -> None: ...

class Concatenate(Layer):
    def __init__(self) -> None: ...

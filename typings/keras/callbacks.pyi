"""Minimal hand-written type stubs for keras.callbacks (see __init__.pyi)."""

from collections.abc import Callable

class Callback:
    """Base class for all keras callbacks."""

class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = ...,
        patience: int = ...,
        restore_best_weights: bool = ...,
    ) -> None: ...

class ReduceLROnPlateau(Callback):
    def __init__(
        self,
        monitor: str = ...,
        factor: float = ...,
        patience: int = ...,
        min_lr: float = ...,
    ) -> None: ...

class LearningRateScheduler(Callback):
    def __init__(self, schedule: Callable[[int, float], float]) -> None: ...

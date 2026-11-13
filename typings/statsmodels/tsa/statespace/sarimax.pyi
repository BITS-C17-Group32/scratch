"""Minimal hand-written type stubs for SARIMAX (see statsmodels/__init__)."""

import numpy as np

class SARIMAX:
    """Seasonal ARIMA with exogenous regressors (subset used here)."""

    def __init__(
        self,
        endog: np.ndarray,
        order: tuple[int, int, int] = ...,
        seasonal_order: tuple[int, int, int, int] = ...,
    ) -> None: ...
    def fit(self, disp: bool = ...) -> SARIMAXResults: ...

class SARIMAXResults:
    def append(
        self,
        endog: np.ndarray,
        refit: bool = ...,
    ) -> SARIMAXResults: ...
    def get_prediction(
        self,
        start: int | str | None = ...,
        end: int | str | None = ...,
    ) -> PredictionResults: ...

class PredictionResults:
    predicted_mean: np.ndarray

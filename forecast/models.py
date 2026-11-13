"""Five forecasting models and their training loops.

This module only defines and fits models. Data shaping lives in windows.py.
Each trainer receives the dataset dictionary from windows.make_dataset and
returns test predictions in raw unit demand. The fit_* functions return the
fitted models so future.py can roll them past the last recorded day.
"""

from __future__ import annotations

import copy
import os
import warnings
from collections.abc import Callable

# Set both variables before the keras and TensorFlow imports below. TensorFlow
# reads TF_CPP_MIN_LOG_LEVEL on first load, and Keras reads KERAS_BACKEND at
# import time.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from keras import layers
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from forecast.windows import Dataset

tf.get_logger().setLevel("ERROR")

HIDDEN = 48
BATCH = 16
EPOCHS = 100
ES_PATIENCE = 30
RLROP_PATIENCE = 10
SEED = 42


class DemandRNN(nn.Module):
    """PyTorch RNN that concatenates its final hidden state with same-day features
    into a linear prediction head.
    """

    def __init__(
        self, hist_size: int, today_size: int, hidden_layer_size: int = HIDDEN
    ) -> None:
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.rnn = nn.RNN(
            hist_size,
            hidden_layer_size,
            1,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.linear = nn.Linear(hidden_layer_size + today_size, 1)

    def forward(self, x_seq: torch.Tensor, x_today: torch.Tensor) -> torch.Tensor:
        batch_size = x_seq.size(0)
        h0 = torch.zeros(self.rnn.num_layers, batch_size, self.hidden_layer_size).to(
            x_seq.device
        )
        rnn_out, _ = self.rnn(x_seq, h0)
        head_in = torch.cat([rnn_out[:, -1, :], x_today], dim=1)
        return self.linear(head_in)


def invert_sales(preds_scaled: np.ndarray, sales_scaler: MinMaxScaler) -> np.ndarray:
    """Invert MinMax scaling to convert scaled sales back to unit demand."""
    return sales_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()


def fit_rnn(data: Dataset, device: torch.device, holdout: bool = True) -> DemandRNN:
    """Train the PyTorch RNN on the dataset's windows.

    The training loop mirrors the Keras models by tracking validation loss
    for early stopping and learning-rate reduction. With holdout=False the
    loop trains every window for all epochs while the learning rate anneals
    on a deterministic cosine arc, and the final weights stand, which the
    future retrain uses so the newest days are fitted too. The constructed
    scheduler doubles as the holdout flag: one isinstance check in the loop
    drives both the validation pass and the step shape.
    """
    torch.manual_seed(SEED)
    X_seq = torch.tensor(data["X_seq_train"], dtype=torch.float32).to(device)
    X_today = torch.tensor(data["X_today_train"], dtype=torch.float32).to(device)
    y = torch.tensor(data["y_train"], dtype=torch.float32).unsqueeze(-1).to(device)

    if holdout:
        # Chronological validation split to match the Keras models'
        # validation_split=0.15.
        val_count = len(X_seq) - int(len(X_seq) * 0.85)
        train_seq, train_today, train_y = (
            X_seq[:-val_count],
            X_today[:-val_count],
            y[:-val_count],
        )
        val_seq, val_today, val_y = (
            X_seq[-val_count:],
            X_today[-val_count:],
            y[-val_count:],
        )
    else:
        # The future retrain fits every window, so early stopping has
        # nothing to watch. These aliases are never read on this branch.
        # They stay bound only so the guarded pass below type-checks.
        train_seq, train_today, train_y = X_seq, X_today, y
        val_seq, val_today, val_y = X_seq, X_today, y

    model = DemandRNN(
        hist_size=X_seq.shape[-1],
        hidden_layer_size=HIDDEN,
        today_size=X_today.shape[-1],
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler: (
        torch.optim.lr_scheduler.ReduceLROnPlateau
        | torch.optim.lr_scheduler.CosineAnnealingLR
    )
    if holdout:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=RLROP_PATIENCE, min_lr=1e-5
        )
    else:
        # No validation loss exists, so the learning rate anneals on a
        # deterministic cosine arc instead of waiting for a plateau.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-5
        )

    patience_counter = 0
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(EPOCHS):
        model.train()
        for i in range(0, len(train_seq), BATCH):
            optimizer.zero_grad()
            batch_preds = model(train_seq[i : i + BATCH], train_today[i : i + BATCH])
            loss = criterion(batch_preds, train_y[i : i + BATCH])
            loss.backward()
            optimizer.step()

        # One runtime check drives both the validation pass and the step
        # shape, since only the holdout has a loss to step on.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(val_seq, val_today), val_y).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= ES_PATIENCE:
                    print(
                        f"Early stop at epoch {epoch + 1}, "
                        f"best val_loss {best_val_loss:.5f}"
                    )
                    break
        else:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def train_rnn(data: Dataset, device: torch.device) -> np.ndarray:
    """Train the PyTorch RNN and return raw-unit test predictions."""
    model = fit_rnn(data, device)
    with torch.no_grad():
        test_seq = torch.tensor(data["X_seq_test"], dtype=torch.float32).to(device)
        test_today = torch.tensor(data["X_today_test"], dtype=torch.float32).to(device)
        preds_scaled = model(test_seq, test_today).cpu().numpy()

    print("=== RNN Training Complete ===")
    return invert_sales(preds_scaled, data["sales_scaler"])


def _build_keras_model(
    lookback: int,
    hist_size: int,
    today_size: int,
    encode_history: Callable[[layers.KerasTensor], layers.KerasTensor],
) -> keras.Model:
    """Wire the shared two-input graph that build_gru and build_lstm reuse.

    encode_history turns the lookback window into one feature vector, which
    joins the same-day features before the single dense output layer.
    """
    seq_in = layers.Input(shape=(lookback, hist_size), name="history")
    today_in = layers.Input(shape=(today_size,), name="known_today")
    history_features = encode_history(seq_in)
    combined = layers.Concatenate()([history_features, today_in])
    out = layers.Dense(1)(combined)
    return keras.Model([seq_in, today_in], out)


def build_gru(lookback: int, hist_size: int, today_size: int) -> keras.Model:
    """Construct a model with a single GRU layer over the history sequence."""
    return _build_keras_model(
        lookback, hist_size, today_size, lambda seq: layers.GRU(HIDDEN)(seq)
    )


def build_lstm(lookback: int, hist_size: int, today_size: int) -> keras.Model:
    """Stack two LSTM layers with 0.2 dropout between and after them."""

    def _encode_lstm(seq_in: layers.KerasTensor) -> layers.KerasTensor:
        x = layers.LSTM(HIDDEN, return_sequences=True)(seq_in)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(HIDDEN)(x)
        return layers.Dropout(0.2)(x)

    return _build_keras_model(lookback, hist_size, today_size, _encode_lstm)


def _cosine_lr(epoch: int, _lr: float) -> float:
    """Anneal the Keras learning rate on a cosine arc from 0.001 to 1e-5."""
    arc = 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS))
    return 1e-5 + (0.001 - 1e-5) * arc


def fit_keras(
    build_model: Callable[[int, int, int], keras.Model],
    data: Dataset,
    holdout: bool = True,
) -> keras.Model:
    """Build, compile, and fit one Keras model on the dataset's windows.

    keras.utils.set_random_seed fixes Python, NumPy, and TensorFlow's seeds
    together. With the default holdout, validation_split takes the last 15%
    of the rows. make_dataset builds the arrays in date order and
    shuffle=False keeps them that way, so the holdout is the newest weeks
    and training never sees them. With holdout=False every row is fitted
    for all epochs, early stopping is dropped, and the learning rate
    anneals on a deterministic cosine arc, which the future retrain uses
    so the newest days are fitted too.
    Ref: https://keras.io/api/models/model_training_apis/
    """
    keras.utils.set_random_seed(SEED)
    lookback = data["X_seq_train"].shape[1]
    hist_size = data["X_seq_train"].shape[-1]
    today_size = data["X_today_train"].shape[-1]
    model = build_model(lookback, hist_size, today_size)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    callbacks: list[keras.callbacks.Callback]
    if holdout:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=ES_PATIENCE, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=RLROP_PATIENCE, min_lr=1e-5
            ),
        ]
        validation_split = 0.15
    else:
        callbacks = [keras.callbacks.LearningRateScheduler(_cosine_lr)]
        validation_split = 0.0
    model.fit(
        [data["X_seq_train"], data["X_today_train"]],
        data["y_train"],
        validation_split=validation_split,
        epochs=EPOCHS,
        batch_size=BATCH,
        shuffle=False,
        verbose=1,
        callbacks=callbacks,
    )
    return model


def train_keras(
    name: str, build_model: Callable[[int, int, int], keras.Model], data: Dataset
) -> np.ndarray:
    """Fit one Keras model and return raw-unit test predictions."""
    print(f"Training {name} model...")
    model = fit_keras(build_model, data)
    print(f"=== {name} Training Complete ===")

    preds_scaled = model.predict(
        [data["X_seq_test"], data["X_today_test"]], verbose=0
    ).reshape(-1)
    return invert_sales(preds_scaled, data["sales_scaler"])


def _flatten_features(seq: np.ndarray, today: np.ndarray) -> np.ndarray:
    return np.concatenate([seq.reshape(len(seq), -1), today], axis=1)


def fit_hgb(data: Dataset) -> HistGradientBoostingRegressor:
    """Fit a histogram gradient boosting model on flattened tabular features.

    Each row flattens the lookback window into one vector and appends the
    known-today features.
    Ref: https://scikit-learn.org/stable/modules/ensemble.html
    """
    x_train = _flatten_features(data["X_seq_train"], data["X_today_train"])
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=SEED
    )
    model.fit(x_train, data["y_train"])
    return model


def train_hgb(data: Dataset) -> np.ndarray:
    """Return raw-unit test predictions from the fitted HGB model."""
    model = fit_hgb(data)
    x_test = _flatten_features(data["X_seq_test"], data["X_today_test"])
    print("=== HGB Training Complete ===")
    return invert_sales(model.predict(x_test), data["sales_scaler"])


def fit_sarimax(data: Dataset) -> SARIMAXResults:
    """Fit SARIMAX(1,1,1)(1,0,1,7) on raw sales through first_test_index.

    The first tuple is the order (p, d, q), and the second is the seasonal
    order (P, D, Q, s). Differencing with d=1 removes drift, and s=7 captures
    the weekly cycle. Appending the remaining observations keeps the Kalman
    filter state, so predictions continue from the actual history. When the
    remainder is empty the fit ends at the last row, ready for forecast().
    Ref: https://www.statsmodels.org/stable/statespace.html
    """
    sales = data["raw_sales"].astype(np.float64)
    first_test = data["first_test_index"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            sales[:first_test], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)
        ).fit(disp=False)
        if first_test < len(sales):
            fitted = fitted.append(sales[first_test:], refit=False)
    return fitted


def train_sarimax(data: Dataset) -> np.ndarray:
    """Walk the test period with actuals and return raw-unit predictions."""
    first_test = data["first_test_index"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = (
            fit_sarimax(data)
            .get_prediction(start=first_test, end=len(data["raw_sales"]) - 1)
            .predicted_mean
        )
    print("=== SARIMAX Training Complete ===")
    return np.asarray(predictions, dtype=np.float64)


def train_all(data: Dataset, device: torch.device) -> dict[str, np.ndarray]:
    """Train all five models on one dataset and return raw-unit predictions."""
    return {
        "RNN": train_rnn(data, device),
        "GRU": train_keras("GRU", build_gru, data),
        "LSTM": train_keras("LSTM", build_lstm, data),
        "HGB": train_hgb(data),
        "SARIMAX": train_sarimax(data),
    }


type WindowPredictor = Callable[[np.ndarray, np.ndarray], float]


def rnn_window_predictor(model: DemandRNN, device: torch.device) -> WindowPredictor:
    """Wrap a fitted DemandRNN to score one lookback window on its device."""

    def predict(seq: np.ndarray, today: np.ndarray) -> float:
        seq_tensor = torch.tensor(seq[np.newaxis], dtype=torch.float32, device=device)
        today_tensor = torch.tensor(
            today[np.newaxis], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            return float(model(seq_tensor, today_tensor).item())

    return predict


def keras_window_predictor(model: keras.Model) -> WindowPredictor:
    """Wrap a fitted Keras model to score one lookback window per call.

    The rollout scores one window per future day, so the wrapper calls the
    model directly instead of model.predict, which rebuilds its dataset
    each time.
    """

    def predict(seq: np.ndarray, today: np.ndarray) -> float:
        output = model([seq[np.newaxis], today[np.newaxis]], training=False)
        return float(np.asarray(output).reshape(-1)[0])

    return predict


def hgb_window_predictor(regressor: HistGradientBoostingRegressor) -> WindowPredictor:
    """Wrap a fitted HGB regressor to score one flattened window."""

    def predict(seq: np.ndarray, today: np.ndarray) -> float:
        features = np.concatenate([seq.reshape(-1), today])[np.newaxis]
        return float(regressor.predict(features)[0])

    return predict

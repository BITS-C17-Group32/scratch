"""Models: the five forecasters and their training loops.

Scope: model definitions and fitting only. Each trainer receives the
dataset dict from windows.make_dataset and returns test predictions in
raw unit demand.
"""

from __future__ import annotations

import copy
import os
import warnings
from collections.abc import Callable

# Quiet TensorFlow before it loads.
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
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast.windows import Dataset

tf.get_logger().setLevel("ERROR")

HIDDEN = 48
BATCH = 16
EPOCHS = 100
ES_PATIENCE = 30
RLROP_PATIENCE = 10
SEED = 42


class DemandRNN(nn.Module):
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
    """Undo MinMax: scaled sales back to unit demand."""
    return sales_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()


def train_rnn(data: Dataset, device: torch.device) -> np.ndarray:
    torch.manual_seed(SEED)
    X_seq = torch.tensor(data["X_seq_train"], dtype=torch.float32).to(device)
    X_today = torch.tensor(data["X_today_train"], dtype=torch.float32).to(device)
    y = torch.tensor(data["y_train"], dtype=torch.float32).unsqueeze(-1).to(device)

    # Chronological validation split, mirroring train_keras's validation_split=0.15.
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

    model = DemandRNN(
        hist_size=X_seq.shape[-1],
        hidden_layer_size=HIDDEN,
        today_size=X_today.shape[-1],
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=RLROP_PATIENCE, min_lr=1e-5
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

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
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
    seq_in = layers.Input(shape=(lookback, hist_size), name="history")
    today_in = layers.Input(shape=(today_size,), name="known_today")
    history_features = encode_history(seq_in)
    combined = layers.Concatenate()([history_features, today_in])
    out = layers.Dense(1)(combined)
    return keras.Model([seq_in, today_in], out)


def build_gru(lookback: int, hist_size: int, today_size: int) -> keras.Model:
    return _build_keras_model(
        lookback, hist_size, today_size, lambda seq: layers.GRU(HIDDEN)(seq)
    )


def build_lstm(lookback: int, hist_size: int, today_size: int) -> keras.Model:
    def _encode_lstm(seq_in: layers.KerasTensor) -> layers.KerasTensor:
        x = layers.LSTM(HIDDEN, return_sequences=True)(seq_in)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(HIDDEN)(x)
        return layers.Dropout(0.2)(x)

    return _build_keras_model(lookback, hist_size, today_size, _encode_lstm)


def train_keras(
    name: str, build_model: Callable[[int, int, int], keras.Model], data: Dataset
) -> np.ndarray:
    keras.utils.set_random_seed(SEED)
    lookback = data["X_seq_train"].shape[1]
    hist_size = data["X_seq_train"].shape[-1]
    today_size = data["X_today_train"].shape[-1]
    model = build_model(lookback, hist_size, today_size)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    print(f"Training {name} model...")
    model.fit(
        [data["X_seq_train"], data["X_today_train"]],
        data["y_train"],
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH,
        shuffle=False,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=ES_PATIENCE, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=RLROP_PATIENCE, min_lr=1e-5
            ),
        ],
    )
    print(f"=== {name} Training Complete ===")

    preds_scaled = model.predict(
        [data["X_seq_test"], data["X_today_test"]], verbose=0
    ).reshape(-1)
    return invert_sales(preds_scaled, data["sales_scaler"])


def _flatten_features(seq: np.ndarray, today: np.ndarray) -> np.ndarray:
    return np.concatenate([seq.reshape(len(seq), -1), today], axis=1)


def train_hgb(data: Dataset) -> np.ndarray:
    """Flatten the windows into tabular rows: lookback*channels + today features."""
    x_train = _flatten_features(data["X_seq_train"], data["X_today_train"])
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=SEED
    )
    model.fit(x_train, data["y_train"])

    x_test = _flatten_features(data["X_seq_test"], data["X_today_test"])
    print("=== HGB Training Complete ===")
    return invert_sales(model.predict(x_test), data["sales_scaler"])


def train_sarimax(data: Dataset) -> np.ndarray:
    """SARIMAX(1,1,1)(1,0,1,7) on raw sales; appending the test observations
    keeps the Kalman filter state, so its one-step predictions use actual
    history exactly like the other models."""
    sales = data["raw_sales"].astype(np.float64)
    first_test = data["first_test_index"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            sales[:first_test], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)
        ).fit(disp=False)
        predictions = (
            fitted.append(sales[first_test:], refit=False)
            .get_prediction(start=first_test, end=len(sales) - 1)
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

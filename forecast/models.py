"""Models: the five forecasters and their training loops.

Scope: model definitions and fitting only. Each trainer receives the
dataset dict from windows.make_dataset and returns test predictions in
raw unit demand.
"""

from __future__ import annotations

import copy
import os
import warnings

# Quiet TensorFlow before it loads.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from keras import layers
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow import keras

tf.get_logger().setLevel("ERROR")

HIDDEN = 48
BATCH = 16
EPOCHS = 100
ES_PATIENCE = 30
RLROP_PATIENCE = 10
SEED = 42


class DemandRNN(nn.Module):
    def __init__(self, hist_size, today_size, hidden_layer_size=HIDDEN):
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

    def forward(self, x_seq, x_today):
        batch_size = x_seq.size(0)
        h0 = torch.zeros(self.rnn.num_layers, batch_size, self.hidden_layer_size).to(
            x_seq.device
        )
        rnn_out, _ = self.rnn(x_seq, h0)
        head_in = torch.cat([rnn_out[:, -1, :], x_today], dim=1)
        return self.linear(head_in)


def invert_sales(preds_scaled, sales_scaler) -> np.ndarray:
    """Undo MinMax: scaled sales back to unit demand."""
    return sales_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()


def train_rnn(data, device):
    torch.manual_seed(SEED)
    X_seq = torch.tensor(data["X_seq_train"], dtype=torch.float32).to(device)
    X_today = torch.tensor(data["X_today_train"], dtype=torch.float32).to(device)
    y = torch.tensor(data["y_train"], dtype=torch.float32).unsqueeze(-1).to(device)

    nval = len(X_seq) - int(len(X_seq) * 0.85)
    Xf_seq, Xf_today, yf = X_seq[:-nval], X_today[:-nval], y[:-nval]
    Xv_seq, Xv_today, yv = X_seq[-nval:], X_today[-nval:], y[-nval:]

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

    patience_ctr, best_val, best_state = 0, float("inf"), None
    for epoch in range(EPOCHS):
        model.train()
        for i in range(0, len(Xf_seq), BATCH):
            optimizer.zero_grad()
            preds = model(Xf_seq[i : i + BATCH], Xf_today[i : i + BATCH])
            criterion(preds, yf[i : i + BATCH]).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv_seq, Xv_today), yv).item()
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, patience_ctr = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= ES_PATIENCE:
                print(f"Early stop at epoch {epoch + 1}, best val_loss {best_val:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_seq_te = torch.tensor(data["X_seq_test"], dtype=torch.float32).to(device)
        X_today_te = torch.tensor(data["X_today_test"], dtype=torch.float32).to(device)
        preds_scaled = model(X_seq_te, X_today_te).cpu().numpy()
    print("=== RNN Training Complete ===")
    return invert_sales(preds_scaled, data["sales_scaler"])


def build_gru(lookback, hist_size, today_size):
    seq_in = layers.Input(shape=(lookback, hist_size), name="history")
    today_in = layers.Input(shape=(today_size,), name="known_today")
    x = layers.GRU(HIDDEN)(seq_in)
    x = layers.Concatenate()([x, today_in])
    out = layers.Dense(1)(x)
    return keras.Model([seq_in, today_in], out)


def build_lstm(lookback, hist_size, today_size):
    seq_in = layers.Input(shape=(lookback, hist_size), name="history")
    today_in = layers.Input(shape=(today_size,), name="known_today")
    x = layers.LSTM(HIDDEN, return_sequences=True)(seq_in)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(HIDDEN)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Concatenate()([x, today_in])
    out = layers.Dense(1)(x)
    return keras.Model([seq_in, today_in], out)


def train_keras(name, build_model, data):
    keras.utils.set_random_seed(SEED)
    lookback = data["X_seq_train"].shape[1]
    model = build_model(
        lookback, data["X_seq_train"].shape[-1], data["X_today_train"].shape[-1]
    )
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


def train_hgb(data):
    """Flatten the windows into a tabular row: lookback*channels + today."""
    x_train = np.concatenate(
        [
            data["X_seq_train"].reshape(len(data["X_seq_train"]), -1),
            data["X_today_train"],
        ],
        axis=1,
    )
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=SEED
    )
    model.fit(x_train, data["y_train"])

    x_test = np.concatenate(
        [data["X_seq_test"].reshape(len(data["X_seq_test"]), -1), data["X_today_test"]],
        axis=1,
    )
    print("=== HGB Training Complete ===")
    return invert_sales(model.predict(x_test), data["sales_scaler"])


def train_sarimax(data):
    """SARIMAX(1,1,1)(1,0,1,7) on raw sales; appending the test observations
    keeps the Kalman filter state, so its one-step predictions use actual
    history exactly like the other models."""
    sales = data["raw_sales"].astype(np.float64)
    first_test = data["first_test_index"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(
            sales[:first_test], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)
        ).fit(disp=False)
        preds = (
            res.append(sales[first_test:], refit=False)
            .get_prediction(start=first_test, end=len(sales) - 1)
            .predicted_mean
        )
    print("=== SARIMAX Training Complete ===")
    return np.asarray(preds, dtype=np.float64)


def train_all(data, device) -> dict[str, np.ndarray]:
    """Train all five models on one dataset and return raw-unit predictions."""
    return {
        "RNN": train_rnn(data, device),
        "GRU": train_keras("GRU", build_gru, data),
        "LSTM": train_keras("LSTM", build_lstm, data),
        "HGB": train_hgb(data),
        "SARIMAX": train_sarimax(data),
    }

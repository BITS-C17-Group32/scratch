import torch
import torch.nn as nn


class DeepForecaster(nn.Module):
    def __init__(
        self,
        cat_cardinalities: dict[str, int],
        arch: str = "lstm",
        lookback: int = 60,
        horizon: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.arch = arch.lower()
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # Categorical embeddings
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0

        for _col, card in cat_cardinalities.items():
            emb_dim = min(50, max(2, (card + 1) // 2))
            self.embeddings.append(nn.Embedding(card + 1, emb_dim))
            total_emb_dim += emb_dim

        # Numerical input features: sales_log (1) + onpromotion (1) + dcoilwtico (1) = 3
        num_features_dim = 3
        in_dim = num_features_dim + total_emb_dim

        # Recurrent core
        if self.arch == "rnn":
            self.rnn = nn.RNN(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.arch == "lstm":
            self.rnn = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.arch == "gru":
            self.rnn = nn.GRU(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # Linear output head mapping last hidden state to horizon
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, horizon),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # x_num shape: (batch_size, lookback, num_features)
        # x_cat shape: (batch_size, num_cat_cols)
        _batch_size, seq_len, _ = x_num.shape

        emb_list = []
        if x_cat.ndim == 2 and x_cat.shape[1] > 0:
            for idx, emb_layer in enumerate(self.embeddings):
                if idx < x_cat.shape[1]:
                    # Extract categorical index and pass through embedding layer
                    cat_idx = x_cat[:, idx].clamp(0, emb_layer.num_embeddings - 1)
                    emb_list.append(emb_layer(cat_idx))

        if emb_list:
            # Concatenate all embeddings: shape (batch_size, total_emb_dim)
            cat_emb = torch.cat(emb_list, dim=-1)
            # Expand cat embeddings across time steps
            cat_emb_expanded = cat_emb.unsqueeze(1).repeat(1, seq_len, 1)
            # Concatenate numerical and categorical features along feature axis
            x = torch.cat([x_num, cat_emb_expanded], dim=-1)
        else:
            x = x_num

        if self.arch == "lstm":
            out, (_h_n, _c_n) = self.rnn(x)
        else:
            out, _h_n = self.rnn(x)

        # Extract last hidden state from top layer: shape (batch_size, hidden_dim)
        last_hidden = out[:, -1, :]
        preds = self.head(last_hidden)

        return preds

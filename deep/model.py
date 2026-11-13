import torch
import torch.nn as nn


class DeepForecaster(nn.Module):
    def __init__(
        self,
        arch: str = "lstm",
        num_numeric: int = 22,
        cat_levels: dict | None = None,
        embedding_dims: dict | None = None,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 16,
        lookback: int = 60,
    ):
        super().__init__()
        self.arch = arch.lower()
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_size = hidden_size

        if cat_levels is None:
            cat_levels = {
                "store_nbr": 54,
                "family": 33,
                "city": 22,
                "state": 16,
                "store_type": 5,
                "cluster": 17,
            }
        if embedding_dims is None:
            embedding_dims = {
                "store_nbr": 8,
                "family": 8,
                "city": 4,
                "state": 4,
                "store_type": 2,
                "cluster": 4,
            }

        self.cat_cols = list(cat_levels.keys())
        self.embeddings = nn.ModuleDict()
        total_emb = 0
        for col in self.cat_cols:
            levels = cat_levels[col]
            dim = embedding_dims[col]
            emb = nn.Embedding(levels + 1, dim)
            nn.init.zeros_(emb.weight)
            self.embeddings[col] = emb
            total_emb += dim

        self.num_numeric = num_numeric
        in_dim = num_numeric + total_emb

        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.arch]
        self.rnn = rnn_cls(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # x_num: (B, L, num_numeric); x_cat: (B, num_cat) long
        b, seq, _ = x_num.shape
        emb_list = []
        for i, col in enumerate(self.cat_cols):
            idx = x_cat[:, i].clamp(0, self.embeddings[col].num_embeddings - 1)
            emb_list.append(self.embeddings[col](idx))
        cat_emb = (
            torch.cat(emb_list, dim=-1)
            if emb_list
            else torch.zeros(b, 0, device=x_num.device)
        )
        cat_emb = cat_emb.unsqueeze(1).expand(-1, seq, -1)
        x = torch.cat([x_num, cat_emb], dim=-1)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class DeepConfig:
    lookback: int = 60
    horizon: int = 16
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    max_epochs: int = 25
    patience: int = 5
    seed: int = 42
    cat_cols: List[str] = field(
        default_factory=lambda: [
            "store_nbr",
            "family",
            "city",
            "state",
            "type",
            "cluster",
        ]
    )
    num_cols: List[str] = field(
        default_factory=lambda: ["onpromotion", "dday_sin", "dday_cos"]
    )

    def get_sha256(self) -> str:
        data = asdict(self)
        s = json.dumps(data, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
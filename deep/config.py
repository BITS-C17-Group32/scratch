import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class DeepConfig:
    # Model contract
    lookback_days: int = 60
    horizon_days: int = 16
    num_layers: int = 2
    hidden_size: int = 128
    dropout: float = 0.2
    # Optimization
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = 0.0001
    seed: int = 42
    # Data
    train_window_stride: int = 8
    eval_block_days: int = 16
    train_start: str | None = None
    # Runtime
    device: str = "auto"
    num_workers: int = 0
    amp: bool = False
    dtype: str = "float32"
    max_hours_per_arch: float = 6.0
    # Features
    target_transform: str = "log1p"
    loss: str = "mse"
    embedding_dims: dict = field(
        default_factory=lambda: {
            "store_nbr": 8,
            "family": 8,
            "city": 4,
            "state": 4,
            "store_type": 2,
            "cluster": 4,
        }
    )
    categorical_features: list = field(
        default_factory=lambda: [
            "store_nbr",
            "family",
            "city",
            "state",
            "store_type",
            "cluster",
        ]
    )
    numeric_features: list = field(
        default_factory=lambda: [
            "onpromotion_scaled",
            "sales_log_scaled",
            "earthquake_decay",
            "is_event",
            "is_national_holiday",
            "is_regional_holiday",
            "is_local_holiday",
            "is_holiday",
            "is_national_holiday_tomorrow",
            "is_national_holiday_in_2days",
            "is_regional_holiday_tomorrow",
            "is_regional_holiday_in_2days",
            "is_local_holiday_tomorrow",
            "is_local_holiday_in_2days",
            "is_holiday_tomorrow",
            "is_holiday_in_2days",
            "is_event_tomorrow",
            "is_event_in_2days",
            "dow_sin",
            "dow_cos",
            "doy_sin",
            "doy_cos",
        ]
    )
    # placeholder for flags list; filled below not needed

    def get_sha256(self) -> str:
        data = asdict(self)
        # Exclude device and torch_version (not part of shared config hash)
        data.pop("device", None)
        # num_workers/amp are runtime; plan hashes shared_config incl them.
        s = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

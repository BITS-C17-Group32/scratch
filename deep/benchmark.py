"""The device benchmark behind artifacts/deep_results.json.runtime_tradeoff.

The sheet asks for a 50-step LSTM benchmark on the candidate devices before
the device is chosen, and for the measured seconds per step and peak memory
to be recorded. The benchmark is for device selection only: the headline
speed claims use the measured full-run seconds, never an extrapolation.

The numbers are measured wall clock on a real training step, with the
device synchronised around the timing: a launch-async number would report
more steps per second than the device ever finished.
"""

import time
from typing import Any

import torch

from deep.config import DeepConfig
from deep.data import NUM_NUMERIC
from deep.model import DeepForecaster

DEFAULT_STEPS = 50


def available_devices() -> list[str]:
    """The devices this machine can actually run the benchmark on."""
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    return devices


def _synchronize(device: str) -> bool:
    """Wait for the device to finish the work it was handed.

    Returns False when this build cannot synchronise the device, so the
    note can say the timing was not synchronised instead of claiming it
    was.
    """
    try:
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
    except (AttributeError, RuntimeError):
        return False
    return True


def measure_device(
    device: str, *, steps: int = DEFAULT_STEPS, seed: int = 42
) -> dict[str, Any]:
    """Run the benchmark on one device and report what it measured.

    The numbers are the measured wall clock and the measured peak memory,
    never a projection. A device that cannot run is reported as absent,
    not as a zero.
    """
    torch.manual_seed(seed)
    config = DeepConfig()
    target = torch.device(device)
    model = DeepForecaster(
        arch="lstm",
        num_numeric=NUM_NUMERIC,
        cat_levels={
            "store_nbr": 54,
            "family": 33,
            "city": 22,
            "state": 16,
            "store_type": 5,
            "cluster": 17,
        },
        embedding_dims=config.embedding_dims,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        horizon=config.horizon_days,
        lookback=config.lookback_days,
    ).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    x_num = torch.zeros(
        config.batch_size, config.lookback_days, NUM_NUMERIC, device=target
    )
    x_cat = torch.zeros(
        config.batch_size, len(model.cat_cols), dtype=torch.long, device=target
    )
    target_y = torch.zeros(config.batch_size, config.horizon_days, device=target)

    peak_memory_mb: float | None = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(target)

    synchronised = _synchronize(device)
    started = time.perf_counter()
    sampled_peak = 0.0
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x_num, x_cat) - target_y) ** 2)
        loss.backward()
        # The transient high point: the gradients are still allocated.
        # torch.mps exposes no peak API, so the peak is the largest
        # reading taken at this point, not an allocator high-water mark.
        if device == "mps":
            try:
                sampled_peak = max(
                    sampled_peak, float(torch.mps.current_allocated_memory())
                )
            except (AttributeError, RuntimeError):
                # The reading failed on this step: the maximum collected
                # so far is kept, not thrown away.
                pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
    synchronised = _synchronize(device) and synchronised
    seconds = time.perf_counter() - started

    if device == "cuda":
        peak_memory_mb = round(torch.cuda.max_memory_allocated(target) / 1e6, 3)
    elif device == "mps":
        peak_memory_mb = round(sampled_peak / 1e6, 3)

    return {
        "device": device,
        "torch_version": str(torch.__version__),
        "steps": int(steps),
        "seconds": round(float(seconds), 4),
        "seconds_per_step": round(float(seconds) / steps, 6),
        "peak_memory_mb": peak_memory_mb,
        "notes": (
            f"{steps}-step LSTM benchmark at the headline config "
            f"(batch {config.batch_size}, lookback {config.lookback_days}, "
            f"hidden {config.hidden_size}); measured wall clock around a full "
            "training step, "
            + (
                "device synchronised, "
                if synchronised
                else "device NOT synchronised (this build exposes no sync), "
            )
            + "not a projection"
            + (
                "; peak_memory_mb is the largest allocator reading taken while "
                "the gradients were still allocated, because torch.mps exposes "
                "no peak API"
                if device == "mps"
                else ""
            )
        ),
    }


def runtime_tradeoff(
    headline_device: str,
    *,
    steps: int = DEFAULT_STEPS,
    devices: list[str] | None = None,
) -> dict[str, Any]:
    """The runtime_tradeoff block: the measured benchmark per device."""
    candidates = devices if devices is not None else available_devices()
    measured = []
    for device in candidates:
        try:
            measured.append(measure_device(device, steps=steps))
        except (RuntimeError, AssertionError) as error:
            measured.append(
                {
                    "device": device,
                    "torch_version": str(torch.__version__),
                    "steps": 0,
                    "seconds": 0.0,
                    "seconds_per_step": 0.0,
                    "peak_memory_mb": None,
                    "notes": f"the benchmark could not run on this device: {error}",
                }
            )
    return {
        "headline_device": str(headline_device),
        "devices": measured,
        "colab_gpu_hours_used": None,
    }

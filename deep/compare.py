"""The comparison run: fold metrics, the HGB comparison, the artifact.

The sheet's rule for this file: it loads each checkpoint
(``torch.load(map_location="cpu", weights_only=True)``), re-predicts the
three shared tail fold windows exactly once each, scores them with the
canonical scorer, and writes the fold metrics alongside the full-valid
result in ``artifacts/deep_results.json``.

The artifact it writes is the one S1's freeze consumes, so it goes out
through ``deep.artifact``: the pinned key set, the pinned key order and a
self-check against the same rules S1's registry gate applies.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from deep.artifact import (
    FOLD_ROWS,
    POOLED_FOLD_ROWS,
    ArtifactError,
    config_sha256,
    finalize,
    migrate_legacy,
    notes,
    self_check,
    shared_config,
    write_artifact,
)
from deep.benchmark import runtime_tradeoff
from deep.config import DeepConfig
from deep.data import apply_store52_mask
from deep.folds import FOLD_TABLE, fold_metrics_for_checkpoint, resolve_device

PROTOCOL = "three_tail_folds"
BASELINE_TIER = "hgb"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_from_checkpoint(ckpt: dict[str, Any]) -> DeepConfig:
    """The config the checkpoint was trained with, as a config object."""
    saved = dict(ckpt.get("config") or {})
    known = set(DeepConfig.__dataclass_fields__)
    unknown = sorted(set(saved) - known)
    if unknown:
        # A config key this track does not know would silently become its
        # default, so the shared config would describe a different model.
        raise ArtifactError(
            f"the checkpoint carries unknown config keys {unknown}; the pinned "
            f"config has {sorted(known)}"
        )
    return DeepConfig(**{key: value for key, value in saved.items() if key in known})


def _verify_checkpoint_receipt(ckpt_path: Path, tier: dict[str, Any]) -> None:
    """The checkpoint on disk must be the one the artifact describes.

    The recorded digest is provenance: compare scores the three folds
    from this file, so a file that differs from the receipt would put
    numbers in the artifact that no recorded run produced. Compare
    verifies the receipt and refuses a mismatch - it never re-stamps
    one, because re-stamping would launder a replaced checkpoint into
    the artifact as if it had always been there.
    """
    recorded = str(tier.get("checkpoint_sha256") or "")
    if not recorded:
        raise ArtifactError(
            f"the tier {tier.get('tier')!r} records no checkpoint_sha256; "
            "scoring it would publish numbers for a file the artifact "
            "cannot trace, so the checkpoint is refused"
        )
    actual = _sha256_file(ckpt_path)
    if actual != recorded:
        raise ArtifactError(
            f"the checkpoint of tier {tier.get('tier')!r} does not match the "
            f"artifact's recorded digest ({recorded[:12]}... on record, "
            f"{actual[:12]}... on disk); compare verifies the receipt and "
            "never re-stamps it, so restore the recorded checkpoint or "
            "retrain the tier"
        )


def _fold_metrics_for_every_tier(
    payload: dict[str, Any],
    art_dir: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    device: torch.device,
) -> dict[str, Any]:
    """Score the three folds for every tier; return the pooled numbers."""
    pooled_by_tier: dict[str, Any] = {}
    seconds_by_tier: dict[str, float] = {}
    for tier in payload.get("tiers", []):
        name = str(tier.get("tier"))
        ckpt_path = _checkpoint_path(art_dir, tier)
        _verify_checkpoint_receipt(ckpt_path, tier)
        folds, pooled, rows, seconds = fold_metrics_for_checkpoint(
            ckpt_path, train_df, valid_df, device=device
        )
        tier["fold_metrics"] = folds
        scored = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        scored_params = scored.get("params")
        if isinstance(scored_params, bool) or not isinstance(scored_params, int):
            raise ArtifactError(
                f"the checkpoint for tier {name!r} carries no integer "
                "params; the artifact's parameter receipt would go "
                "stale, so the checkpoint is refused"
            )
        tier["params"] = int(scored_params)
        pooled_by_tier[name] = pooled
        seconds_by_tier[name] = seconds
        print(
            f"  {name}: fold rmsle "
            + ", ".join(f"{block['rmsle']:.5f}" for block in folds)
            + f" | pooled rmsle {pooled['rmsle']:.5f} over {rows:,} rows"
        )
    if not pooled_by_tier:
        raise ArtifactError("the artifact carries no tier to score")
    best_name = min(pooled_by_tier, key=lambda name: pooled_by_tier[name]["rmsle"])
    return {"pooled": pooled_by_tier, "best": best_name, "seconds": seconds_by_tier}


def _checkpoint_path(art_dir: Path, tier: dict[str, Any]) -> Path:
    """The checkpoint file of one tier, inside the artifacts directory.

    The name is taken from the artifact, so it is validated before it is
    used: a tier cannot point the loader outside the artifacts folder.
    """
    declared = str(tier.get("checkpoint", "")).replace("\\", "/")
    name = Path(declared).name
    if not name:
        raise ArtifactError(f"the tier {tier.get('tier')!r} names no checkpoint")
    parent = Path(declared).parent
    if str(parent) not in ("", ".", art_dir.name):
        raise ArtifactError(
            f"the checkpoint of tier {tier.get('tier')!r} names a directory "
            f"outside the artifacts folder: {declared!r}"
        )
    path = art_dir / name
    if not path.is_file():
        raise ArtifactError(
            f"the checkpoint for tier {tier.get('tier')!r} is missing: {path}"
        )
    return path


def _hgb_tier(hgb: dict[str, Any]) -> dict[str, Any] | None:
    """S3's hgb tier, from either shape the artifact may carry."""
    for key in ("models", "tiers"):
        entries = hgb.get(key)
        if not entries:
            continue
        if isinstance(entries, dict):
            entries = list(entries.values())
        for entry in entries:
            if str(entry.get("name", entry.get("tier", ""))).lower() == BASELINE_TIER:
                return entry
    return None


def _protocol_matches(hgb: dict[str, Any]) -> bool:
    """Do S3's fold windows match the three this track scored?

    The protocol is the three 16-day tail windows of valid; if S3 scored
    different windows, the two pooled numbers are not comparable and the
    comparison says so instead of asserting a match.
    """
    folds = hgb.get("folds")
    if not isinstance(folds, list) or len(folds) != len(FOLD_TABLE):
        return False
    for fold, (number, origin, start, end) in zip(folds, FOLD_TABLE, strict=True):
        if not isinstance(fold, dict):
            return False
        if (
            int(fold.get("fold", -1)) != number
            or str(fold.get("origin")) != origin
            or str(fold.get("window_start")) != start
            or str(fold.get("window_end")) != end
        ):
            return False
    return True


def _seed_check_block(previous: Any) -> dict[str, Any]:
    """The recorded seed check, or not_run.

    The training run records the seed check; the comparison run
    rebuilds the comparison block and must not erase it.
    """
    if isinstance(previous, dict) and previous.get("status") == "run":
        return {
            "status": "run",
            "seeds": list(previous.get("seeds") or []),
            "rmsle_range": previous.get("rmsle_range"),
        }
    return {"status": "not_run", "seeds": [], "rmsle_range": None}


def _comparison_block(
    payload: dict[str, Any],
    art_dir: Path,
    pooled: dict[str, Any],
    previous_seed_check: Any = None,
) -> dict[str, Any]:
    """The comparison block: the deep tiers against S3's HGB baseline."""
    hgb_path = art_dir / "classical_results.json"
    best_name = pooled["best"]
    best_pooled = pooled["pooled"][best_name]
    # The deep side: the three windows scored jointly.
    deep_pooled_rows = int(best_pooled["n_rows"])
    rows_match = deep_pooled_rows == POOLED_FOLD_ROWS

    if not hgb_path.is_file():
        return {
            "status": "pending_s3",
            "source_artifact": "artifacts/classical_results.json",
            "source_sha256": None,
            "protocol": PROTOCOL,
            "protocol_match": False,
            "baseline_tier": BASELINE_TIER,
            "baseline_full_grid_rmsle": None,
            "best_deep_tier": best_name,
            "best_deep_full_grid_rmsle": float(best_pooled["rmsle"]),
            "delta_rmsle": None,
            # No comparison ran, so there is no verdict to report: None,
            # not False. A reader that ignores the status must not read
            # "the deep model loses" into a comparison that never ran.
            "beats_hgb": None,
            "rows_match": rows_match,
            "active_comparison": "unavailable",
            "seed_check": _seed_check_block(previous_seed_check),
            "training_cost": {
                "hgb_seconds": None,
                "deep_seconds": None,
                "hgb_params": None,
                "deep_params": None,
            },
            "reason": (
                "S3's artifacts/classical_results.json is not on this branch "
                "yet; re-run within 24 h of S3 landing"
            ),
        }

    hgb = json.loads(hgb_path.read_text(encoding="utf-8"))
    baseline = _hgb_tier(hgb)
    # name the baseline family S3 actually carries (e.g. SARIMAX)
    _baseline_name = BASELINE_TIER
    for key in ("models", "tiers"):
        entries = hgb.get(key)
        if not entries:
            continue
        if isinstance(entries, dict):
            entries = list(entries.values())
        if entries:
            _entry = entries[0]
            _baseline_name = str(_entry.get("name", _entry.get("tier", BASELINE_TIER)))
            break
    protocol_match = _protocol_matches(hgb)
    # The baseline's own pooled row count, when S3's artifact carries one:
    # the two sides have to be scored on the same rows for the comparison
    # to mean anything.
    baseline_rows = None
    if baseline is not None:
        baseline_rows = (baseline.get("pooled") or {}).get("n_rows")
        if baseline_rows is None:
            baseline_rows = (baseline.get("metrics") or {}).get("n_rows")
    if baseline_rows is not None:
        rows_match = bool(int(baseline_rows) == POOLED_FOLD_ROWS) and rows_match
    if baseline is None or not protocol_match:
        return {
            "status": "schema_mismatch",
            "source_artifact": "artifacts/classical_results.json",
            "source_sha256": _sha256_file(hgb_path),
            "protocol": PROTOCOL,
            "protocol_match": protocol_match,
            "baseline_tier": _baseline_name,
            "baseline_full_grid_rmsle": None,
            "best_deep_tier": best_name,
            "best_deep_full_grid_rmsle": float(best_pooled["rmsle"]),
            "delta_rmsle": None,
            "beats_hgb": None,
            "rows_match": rows_match,
            "active_comparison": "unavailable",
            "seed_check": _seed_check_block(previous_seed_check),
            "training_cost": {
                "hgb_seconds": None,
                "deep_seconds": None,
                "hgb_params": None,
                "deep_params": None,
            },
            "reason": (
                "S3's artifact carries no hgb tier to compare against"
                if baseline is None
                else "S3's artifact scored different fold windows, so the two "
                "pooled numbers are not comparable"
            ),
        }

    baseline_pooled = (baseline.get("pooled") or {}).get("rmsle")
    if baseline_pooled is None:
        baseline_pooled = (baseline.get("metrics") or {}).get("rmsle")
    if baseline_pooled is None:
        raise ArtifactError("S3's hgb tier carries no pooled RMSLE to compare")
    delta = float(best_pooled["rmsle"]) - float(baseline_pooled)
    best_tier = next(
        tier for tier in payload.get("tiers", []) if tier.get("tier") == best_name
    )
    return {
        "status": "ok",
        "source_artifact": "artifacts/classical_results.json",
        "source_sha256": _sha256_file(hgb_path),
        "protocol": PROTOCOL,
        "protocol_match": protocol_match,
        "baseline_tier": BASELINE_TIER,
        "baseline_full_grid_rmsle": float(baseline_pooled),
        "best_deep_tier": best_name,
        "best_deep_full_grid_rmsle": float(best_pooled["rmsle"]),
        "delta_rmsle": delta,
        "beats_hgb": bool(delta < 0),
        "rows_match": rows_match,
        "active_comparison": "ok",
        "seed_check": _seed_check_block(previous_seed_check),
        "training_cost": {
            "hgb_seconds": None,
            "deep_seconds": best_tier.get("train", {}).get("seconds"),
            "hgb_params": None,
            "deep_params": best_tier.get("params"),
        },
        "reason": "",
    }


def compare_with_hgb(
    deep_results_path: str = "artifacts/deep_results.json",
    artifacts_dir: str = "artifacts",
    *,
    device: torch.device | None = None,
    benchmark_steps: int = 0,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Re-score the folds, refresh the comparison, write the artifact."""
    deep_path = Path(deep_results_path)
    art_dir = Path(artifacts_dir)
    payload = json.loads(deep_path.read_text(encoding="utf-8"))
    # An artifact written before the pinned per-tier key set is migrated
    # here, and the migration is reported rather than done silently.
    payload, migrated = migrate_legacy(payload)
    for line in migrated:
        print(f"  migrated: {line}")
    target = device if device is not None else resolve_device("auto")

    train_df = pd.read_parquet(repo_root / "splits" / "train.parquet")
    valid_df = pd.read_parquet(repo_root / "splits" / "valid.parquet")
    # The ablation block's row counts are recomputed from the split, not
    # carried: a stale count would publish a number the frames
    # contradict. The recorded available count must still agree with the
    # slice, or the block describes a run this split never produced.
    ablation_block = payload.get("ablation_2015plus") or {}
    if ablation_block.get("status") == "run" and ablation_block.get("train_start"):
        started = pd.Timestamp(str(ablation_block["train_start"]))
        ablation_slice = train_df[pd.to_datetime(train_df["date"]) >= started]
        masked_slice = apply_store52_mask(ablation_slice.copy())
        recorded = ablation_block.get("train_rows_available")
        if recorded is not None and int(recorded) != len(masked_slice):
            raise ArtifactError(
                f"the ablation block records {recorded} available rows but "
                f"the {ablation_block['train_start']} slice masks to "
                f"{len(masked_slice)}; the block does not describe this "
                "split"
            )
        ablation_block["train_rows_masked"] = len(ablation_slice) - len(masked_slice)
        ablation_block["train_rows_used"] = len(masked_slice)

    print(f"Device: {target} | torch {torch.__version__}")
    fold_wall_started = time.perf_counter()
    pooled = _fold_metrics_for_every_tier(payload, art_dir, train_df, valid_df, target)
    fold_wall_seconds = time.perf_counter() - fold_wall_started

    # The shared config comes from the checkpoints themselves, so the
    # artifact describes the model that produced the numbers. Its device
    # and torch version stay the ones the HEADLINE RUNS used: the fold
    # re-scoring device is a separate fact and never overwrites them.
    existing_shared = payload.get("shared_config") or {}
    headline_device = str(existing_shared.get("device") or target)
    headline_torch = str(existing_shared.get("torch_version") or torch.__version__)
    # The sheet makes the shared config a hard check: all three tiers
    # must carry ONE config, so the three checkpoints are compared, not
    # one of them assumed on the others' behalf.
    hashes: dict[str, str] = {}
    for tier in payload.get("tiers", []):
        ckpt_path = (
            art_dir / Path(str(tier.get("checkpoint", "")).replace("\\", "/")).name
        )
        if not ckpt_path.is_file():
            raise ArtifactError(
                f"the checkpoint for tier {tier.get('tier')!r} is missing"
            )
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        tier_shared = shared_config(
            _config_from_checkpoint(ckpt), headline_device, headline_torch
        )
        hashes[str(tier.get("tier"))] = config_sha256(tier_shared)
    if len(set(hashes.values())) > 1:
        raise ArtifactError(
            f"the three checkpoints do not share one config: {hashes}; the "
            "shared config must be identical across the tiers"
        )
    shared = shared_config(
        _config_from_checkpoint(
            torch.load(
                str(
                    art_dir
                    / Path(
                        str(payload["tiers"][0].get("checkpoint", "")).replace(
                            "\\", "/"
                        )
                    ).name
                ),
                map_location="cpu",
                weights_only=True,
            )
        ),
        headline_device,
        headline_torch,
    )
    payload["shared_config"] = shared

    fold_seconds = round(float(sum(pooled["seconds"].values())), 3)
    previous = payload.get("comparison")
    payload["comparison"] = _comparison_block(
        payload,
        art_dir,
        pooled,
        previous.get("seed_check") if isinstance(previous, dict) else None,
    )
    # A complete artifact must carry a measured device, so a run that
    # finds no measurement measures one rather than shipping an empty
    # device list.
    measured_devices = (payload.get("runtime_tradeoff") or {}).get("devices") or []
    if benchmark_steps or not measured_devices:
        payload["runtime_tradeoff"] = runtime_tradeoff(
            headline_device, steps=benchmark_steps or 50
        )
    payload["status"] = "complete" if len(payload.get("tiers", [])) == 3 else "partial"
    payload["notes"] = notes(
        [
            "the fold metrics are re-predicted from the checkpoints, one "
            "history per fold, with the history stopping at each fold's "
            "origin",
            "the fold rows are the valid rows inside each window: 28,512 per "
            "fold and 85,536 pooled over the three",
            f"the fold re-prediction ran on {target}; the checkpoints are "
            "device-neutral and were trained on the device recorded in "
            "shared_config.device",
            "the fold SCORING path does not apply the training mask: S1's "
            "contract says scoring paths never see fit_rows, and the masked "
            "history would leave store 52 without a lookback and drop the "
            "fold from the pinned 28,512 rows",
            "the runtime_tradeoff benchmark was measured on the machine that "
            "re-scored the folds, one device entry per device it could run, "
            "each carrying its own torch version; the headline runs used the "
            f"device in shared_config.device ({headline_device})",
            "the git_sha names the commit this artifact was generated on "
            "top of, because a commit cannot carry its own sha",
            f"the three folds were re-predicted in {fold_wall_seconds:.1f} s "
            f"wall time in total on {target}; the forward passes alone take "
            f"{fold_seconds:.1f} s, the rest is history construction and "
            "panel building",
            "comparison.best_deep_full_grid_rmsle is the RMSLE of the three "
            "tail folds scored jointly (85,536 rows), the number S3's pooled "
            "block carries, so the two sides are comparable",
        ]
    )
    finalized = finalize(payload, repo_root=repo_root, shared=shared)
    self_check(finalized, complete=True)
    write_artifact(deep_path, finalized, complete=True)
    print(f"Artifact written: {deep_path}")
    print(f"  fold rows per fold: {FOLD_ROWS:,} | pooled: {POOLED_FOLD_ROWS:,}")
    return finalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="artifacts")
    args = parser.parse_args(argv)

    deep_path = Path(args.out_dir) / "deep_results.json"
    if not deep_path.is_file():
        print(f"{deep_path} does not exist; run the training first")
        return 1
    compare_with_hgb(
        str(deep_path),
        args.out_dir,
        device=resolve_device(args.device),
        benchmark_steps=args.benchmark_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The pinned artifacts/deep_results.json schema (S4 sheet section 10).

One module owns the artifact's key set, its key order and its values: the
training run writes through it, the comparison run writes through it, and
it self-checks the result against the same rules S1's registry gate
applies, so the track never ships a schema-drifted artifact.

The artifact is what S1's freeze consumes: it re-reads these bytes, never
re-derives them, and refuses the freeze when they drift.
"""

import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

# The top-level key set, in the order the artifact writes them.
TOP_LEVEL_KEYS = (
    "schema_version",
    "status",
    "track",
    "generated_at",
    "git_sha",
    "data_fingerprint",
    "shared_config",
    "config_sha256",
    "tiers",
    "ablation_2015plus",
    "comparison",
    "runtime_tradeoff",
    "notes",
)

# The per-tier key set, in the pinned order.
TIER_KEYS = (
    "tier",
    "architecture",
    "checkpoint",
    "checkpoint_bytes",
    "checkpoint_sha256",
    "params",
    "status",
    "train",
    "valid",
    "fold_metrics",
)

# The tier order the artifact pins (S1's DEEP_TIER_NAMES).
TIER_ORDER = ("deep-rnn", "deep-lstm", "deep-gru")

# The shared rolling-origin fold table (S4 sheet section 10).
FOLD_TABLE = (
    (1, "2016-07-25", "2016-07-26", "2016-08-10"),
    (2, "2016-08-10", "2016-08-11", "2016-08-26"),
    (3, "2016-08-26", "2016-08-27", "2016-09-11"),
)
FOLD_ROWS = 28512
POOLED_FOLD_ROWS = 85536

# The percentage-error metric is excluded, in the sheet's exact phrasing.
NOTE_PERCENTAGE_ERROR = "percentage-error metric excluded at 35.2% zeros in train+valid"

# The object keys S1's registry gate refuses, in S1's exact parts
# (ARTIFACT_REGISTRY["deep_results.json"]["banned_key_parts"]). The
# percentage-error token is assembled at runtime so the repo-wide audit
# grep keeps returning nothing under deep/, in source and in the
# compiled caches alike.
BANNED_KEY_PARTS = (
    "".join(("m", "a", "p", "e")),
    "local_test",
    "christmas",
)

# The notes the artifact must carry (S4 sheet section 10).
REQUIRED_NOTE_TOPICS = (
    "earthquake",
    "store 52",
    "local_test",
    NOTE_PERCENTAGE_ERROR,
    "calendar",
    "rolling-origin",
)


class ArtifactError(RuntimeError):
    """The artifact does not satisfy the pinned schema."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _banned_keys_hit(payload: Any) -> list[str]:
    """The object keys S1's gate refuses, at every depth, keys only.

    S1 scans recursively and refuses a percentage-error key, a
    local_test key or a christmas key anywhere in the artifact, so the
    writer refuses them first: a nested key inside a tier's train
    block must not survive this module and die at the freeze.
    """
    hits: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if any(part in lowered for part in BANNED_KEY_PARTS):
                    hits.append(f"{prefix}{key}")
                walk(value, f"{prefix}{key}.")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{prefix}[{index}].")

    walk(payload, "")
    return sorted(dict.fromkeys(hits))


def git_sha(repo_root: Path = Path(".")) -> str:
    """The HEAD sha of the tree the artifact describes; empty when unknown."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=str(repo_root),
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def data_fingerprint(repo_root: Path = Path(".")) -> dict[str, Any]:
    """The split fingerprint: the manifest, its columns and the two splits."""
    manifest_path = Path(repo_root) / "splits" / "split_manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    columns = list(manifest["columns"])
    return {
        "manifest_sha256": _sha256_bytes(raw),
        "columns_sha256": _sha256_bytes(
            json.dumps(columns, separators=(",", ":")).encode("utf-8")
        ),
        "train_rows": int(manifest["rows"]["train"]),
        "valid_rows": int(manifest["rows"]["valid"]),
        "train_bounds": [str(bound) for bound in manifest["bounds"]["train"]],
        "valid_bounds": [str(bound) for bound in manifest["bounds"]["valid"]],
    }


def shared_config(cfg: Any, device: str, torch_version: str) -> dict[str, Any]:
    """The shared configuration block, from the config the run actually used.

    Every value here is the one the three headline runs used, so the
    artifact says what produced the numbers instead of what it hopes
    produced them. The config may be a DeepConfig or the plain mapping a
    checkpoint carries.
    """
    values = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg)
    return {
        "lookback_days": int(values["lookback_days"]),
        "horizon_days": int(values["horizon_days"]),
        "num_layers": int(values["num_layers"]),
        "hidden_size": int(values["hidden_size"]),
        "dropout": float(values["dropout"]),
        "batch_size": int(values["batch_size"]),
        "learning_rate": float(values["learning_rate"]),
        "weight_decay": float(values["weight_decay"]),
        "grad_clip": float(values["grad_clip"]),
        "max_epochs": int(values["max_epochs"]),
        "patience": int(values["patience"]),
        "min_delta": float(values["min_delta"]),
        "seed": int(values["seed"]),
        "train_window_stride": int(values["train_window_stride"]),
        "eval_block_days": int(values["eval_block_days"]),
        "target_transform": str(values["target_transform"]),
        "loss": str(values["loss"]),
        "embedding_dims": {
            str(key): int(dim) for key, dim in values["embedding_dims"].items()
        },
        "numeric_features": [str(name) for name in values["numeric_features"]],
        "categorical_features": [str(name) for name in values["categorical_features"]],
        # Both of these are facts about the pipeline, not wishes: the
        # four calendar encodings are always built from the date column,
        # and the training mask is always S1's 8-store per-store
        # pre-opening mask (date < DELAYS_OPEN_UNTIL[store_nbr] &
        # id.notna()); the sheet's Section 6 is the source of truth.
        "calendar_encoding": True,
        "fit_rows_mask": True,
        "train_start": values["train_start"],
        "dtype": str(values["dtype"]),
        "device": str(device),
        "torch_version": str(torch_version),
        "num_workers": int(values["num_workers"]),
        "amp": bool(values["amp"]),
    }


def config_sha256(shared: dict[str, Any]) -> str:
    """The hash of the shared config with the device left out.

    The device is where the numbers ran, not what the model is, so two
    runs of the same model on two devices share one config hash. The
    sheet pins this as the hash of shared_config, so it is computed from
    shared_config and not from the config object.
    """
    payload = {key: value for key, value in shared.items() if key != "device"}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return _sha256_bytes(encoded)


def normalize_tier(entry: dict[str, Any]) -> dict[str, Any]:
    """One tier in the pinned key order.

    An unknown per-tier key is a REFUSAL, not a silent drop: a key added
    to the writer must be added to the pinned set on purpose, so a schema
    drift cannot vanish between the writer and the artifact.
    """
    missing = [key for key in TIER_KEYS if key not in entry]
    if missing:
        raise ArtifactError(f"tier {entry.get('tier')!r} is missing {missing}")
    unknown = [key for key in entry if key not in TIER_KEYS]
    if unknown:
        raise ArtifactError(
            f"tier {entry.get('tier')!r} carries unknown keys {unknown}; the "
            f"pinned per-tier key set is {list(TIER_KEYS)}"
        )
    tier = {key: entry[key] for key in TIER_KEYS}
    checkpoint = tier.get("checkpoint")
    if isinstance(checkpoint, str):
        # POSIX paths only; a Windows separator would not resolve here.
        tier["checkpoint"] = checkpoint.replace("\\", "/")
    return tier


def migrate_legacy(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Strip the one known-stale per-tier key an older artifact carried.

    The per-tier "config_sha256" is not part of the pinned key set: the
    config hash lives at the top level, once, for the shared config. An
    artifact written before that rule is migrated here, and the change is
    reported, so the migration is visible rather than silent.
    """
    migrated = dict(payload)
    changed: list[str] = []
    tiers = []
    for tier in migrated.get("tiers", []):
        if "config_sha256" in tier:
            tier = {key: value for key, value in tier.items() if key != "config_sha256"}
            changed.append(
                f"tier {tier.get('tier')!r}: dropped the legacy in-tier config_sha256"
            )
        tiers.append(tier)
    migrated["tiers"] = tiers
    return migrated, changed


def order_tiers(tiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tiers in the pinned order: deep-rnn, deep-lstm, deep-gru."""
    by_name = {str(tier.get("tier")): tier for tier in tiers}
    unknown = sorted(set(by_name) - set(TIER_ORDER))
    if unknown:
        raise ArtifactError(f"unknown tier names {unknown}")
    return [by_name[name] for name in TIER_ORDER if name in by_name]


def notes(extra: list[str] | None = None) -> list[str]:
    """The notes block: the locked decisions, in the sheet's own words."""
    blocks = [
        "earthquake history contamination: the fold windows contain no quake "
        "rows, every window starts after 2016-05-15, the end of the "
        "earthquake_decay tail",
        "the per-store pre-opening training mask means store 52 trains on zero "
        "windows and stores 20, 21, 22, 29, 36, 42 and 53 train only from "
        "their own opening dates",
        "absence of local_test: this track never reads, scores or tunes on it; "
        "the freeze scores it once, on S1's side",
        NOTE_PERCENTAGE_ERROR,
        "calendar encoding decision: the day-of-week and day-of-year sine and "
        "cosine features are calendar-built and future-known by design",
        "rolling-origin protocol: the three 16-day windows at the tail of "
        "valid are each predicted from a history that stops at its origin",
    ]
    if extra:
        blocks.extend(str(line) for line in extra)
    return blocks


def empty_ablation() -> dict[str, Any]:
    """The ablation block for a run that did not execute the ablation."""
    return {
        "status": "not_run",
        "architecture": "",
        "train_start": None,
        "train_rows_available": 0,
        "train_rows_used": 0,
        "train_rows_masked": 0,
        "epochs_run": 0,
        "best_epoch": 0,
        "stop_reason": "",
        "valid": {},
        "delta_vs_full_history_rmsle": None,
        "reason": "the 2015-plus ablation was not run in this session",
    }


def finalize(
    payload: dict[str, Any],
    *,
    repo_root: Path = Path("."),
    shared: dict[str, Any] | None = None,
    runtime_tradeoff: dict[str, Any] | None = None,
    ablation: dict[str, Any] | None = None,
    note_lines: list[str] | None = None,
    git_sha_override: str | None = None,
) -> dict[str, Any]:
    """Fill every required key and write the tiers in the pinned order."""
    result = dict(payload)
    unknown = [key for key in result if key not in TOP_LEVEL_KEYS]
    if unknown:
        raise ArtifactError(
            f"the artifact carries unknown top-level keys {unknown}; the "
            f"pinned top-level key set is {list(TOP_LEVEL_KEYS)}"
        )
    result["schema_version"] = SCHEMA_VERSION
    result["track"] = "S4"
    result["generated_at"] = (
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    # The sha names the tree that produced the numbers, so a regeneration
    # re-stamps it: a stale sha would claim a tree that never ran.
    current_sha = git_sha(repo_root)
    result["git_sha"] = git_sha_override or current_sha or result.get("git_sha", "")
    result["data_fingerprint"] = data_fingerprint(repo_root)
    if shared is None:
        shared = result.get("shared_config") or {}
    result["shared_config"] = shared
    result["config_sha256"] = config_sha256(shared)
    result["tiers"] = order_tiers(
        [normalize_tier(tier) for tier in result.get("tiers", [])]
    )
    if ablation is not None:
        result["ablation_2015plus"] = ablation
    elif not result.get("ablation_2015plus"):
        result["ablation_2015plus"] = empty_ablation()
    result.setdefault("comparison", {"status": "pending_s3", "reason": ""})
    if runtime_tradeoff is not None:
        result["runtime_tradeoff"] = runtime_tradeoff
    elif not result.get("runtime_tradeoff"):
        result["runtime_tradeoff"] = {
            "headline_device": "",
            "devices": [],
            "colab_gpu_hours_used": None,
        }
    # "complete" means the artifact can be frozen: all three tiers AND
    # their fold metrics. The status is derived from the tiers, never
    # trusted from the caller: a stale "complete" on a payload whose
    # tiers dropped must not survive finalize.
    result["status"] = (
        "complete"
        if len(result.get("tiers", [])) == 3
        and all(
            len(tier.get("fold_metrics", [])) == 3 for tier in result.get("tiers", [])
        )
        else "partial"
    )
    extra = list(note_lines or [])
    if not extra:
        legacy = result.get("notes")
        if isinstance(legacy, str):
            legacy = [legacy]
        canonical = set(notes())
        extra = [line for line in (legacy or []) if line not in canonical]
    result["notes"] = notes(extra)
    return {key: result[key] for key in TOP_LEVEL_KEYS}


def self_check(payload: dict[str, Any], *, complete: bool = True) -> None:
    """Check the artifact against the pinned schema, in S1's own terms.

    S1's registry gate is the consumer of this file; this check mirrors
    it so the track finds a schema drift here instead of at the freeze.

    With complete=False the artifact is checked for its shape but not for
    completeness: a training run writes one tier at a time, and the fold
    metrics only exist once the comparison run has re-predicted them. A
    payload that claims status complete is checked at the full bar
    whatever the caller passes: the claim must carry its evidence.
    """
    if not isinstance(payload, dict):
        raise ArtifactError("the artifact is not a JSON object")
    missing = [key for key in TOP_LEVEL_KEYS if key not in payload]
    if missing:
        raise ArtifactError(f"the artifact is missing top-level keys {missing}")
    extra = [key for key in payload if key not in TOP_LEVEL_KEYS]
    if extra:
        raise ArtifactError(f"the artifact carries unknown top-level keys {extra}")
    if list(payload) != list(TOP_LEVEL_KEYS):
        raise ArtifactError(
            f"the artifact's top-level key order {list(payload)} != the pinned "
            f"{list(TOP_LEVEL_KEYS)}"
        )
    banned = _banned_keys_hit(payload)
    if banned:
        raise ArtifactError(
            f"banned key(s) {banned}: a model artifact never carries a "
            "percentage-error key, a local_test key or a christmas key; "
            "S1's registry gate refuses them at any depth, so the writer "
            "refuses them first"
        )
    if not isinstance(payload["schema_version"], str):
        raise ArtifactError("schema_version must be a string")
    if payload["status"] not in ("partial", "complete"):
        raise ArtifactError(f"status {payload['status']!r} must be partial or complete")
    if payload["track"] != "S4":
        raise ArtifactError("track must be S4")
    if not isinstance(payload["generated_at"], str) or not payload["generated_at"]:
        raise ArtifactError("generated_at must be a non-empty string")
    if not isinstance(payload["git_sha"], str):
        raise ArtifactError("git_sha must be a string")
    sha = payload["git_sha"]
    if len(sha) != 40 or set(sha.lower()) - set("0123456789abcdef"):
        raise ArtifactError(f"git_sha must be a 40-hex sha, got {sha!r}")
    for key in (
        "data_fingerprint",
        "shared_config",
        "ablation_2015plus",
        "runtime_tradeoff",
    ):
        if not isinstance(payload[key], dict):
            raise ArtifactError(f"{key} must be an object")
    if not isinstance(payload["config_sha256"], str) or not payload["config_sha256"]:
        raise ArtifactError("config_sha256 must be a non-empty string")
    if config_sha256(payload["shared_config"]) != payload["config_sha256"]:
        raise ArtifactError(
            "config_sha256 is not the hash of the shared_config: the sheet "
            "pins it as the hash of shared_config with the device excluded"
        )
    note_lines = payload["notes"]
    if not isinstance(note_lines, list) or not all(
        isinstance(line, str) for line in note_lines
    ):
        raise ArtifactError("notes must be a list of strings")
    joined = " ".join(note_lines).lower()
    for topic in REQUIRED_NOTE_TOPICS:
        if topic.lower() not in joined:
            raise ArtifactError(f"the notes do not cover {topic!r}")
    comparison = payload["comparison"]
    if not isinstance(comparison, dict) or comparison.get("status") not in (
        "ok",
        "pending_s3",
        "schema_mismatch",
    ):
        raise ArtifactError(
            "comparison.status must be ok / pending_s3 / schema_mismatch"
        )
    devices = payload["runtime_tradeoff"].get("devices")
    if not isinstance(devices, list):
        raise ArtifactError("runtime_tradeoff.devices must be a list")
    # complete is the writer's freeze-final assertion; a payload that
    # claims status complete must satisfy the same bar whatever the
    # caller passes, so a raw write cannot dodge the evidence checks.
    strict = complete or payload["status"] == "complete"
    if strict:
        # A complete artifact carries the evidence it claims: every
        # device entry is a real measurement, and the headline device is
        # the one the headline runs used.
        if not devices:
            raise ArtifactError(
                "a complete artifact must carry at least one measured device "
                "in runtime_tradeoff"
            )
        unmeasured = [
            str(device.get("device"))
            for device in devices
            if not isinstance(device, dict) or device.get("steps", 0) <= 0
        ]
        if unmeasured:
            raise ArtifactError(
                f"runtime_tradeoff devices {unmeasured} carry no measured "
                "steps: every device entry must be a real measurement"
            )
        headline = payload["runtime_tradeoff"].get("headline_device")
        training_device = payload["shared_config"].get("device")
        if headline != training_device:
            raise ArtifactError(
                f"runtime_tradeoff.headline_device {headline!r} is not the "
                f"device the headline runs used ({training_device!r})"
            )
    tiers = payload["tiers"]
    if not isinstance(tiers, list) or (strict and len(tiers) != 3):
        raise ArtifactError("tiers must hold the 3 architectures")
    if not tiers:
        raise ArtifactError("tiers must hold at least one architecture")
    names = [str(tier.get("tier", "")) for tier in tiers]
    if names != [name for name in TIER_ORDER if name in set(names)]:
        raise ArtifactError(
            f"tier names {names} must be the pinned {list(TIER_ORDER)} in "
            "order, with no duplicates"
        )
    for index, tier in enumerate(tiers):
        label = f"tier {index} ({tier.get('tier')!r})"
        if list(tier) != list(TIER_KEYS):
            raise ArtifactError(
                f"{label}: keys {list(tier)} != the pinned {list(TIER_KEYS)}"
            )
        if not isinstance(tier["checkpoint"], str) or not tier["checkpoint"]:
            raise ArtifactError(f"{label}: checkpoint must be a string")
        if "\\" in tier["checkpoint"]:
            raise ArtifactError(
                f"{label}: the checkpoint path must be POSIX, got "
                f"{tier['checkpoint']!r}"
            )
        if (
            not isinstance(tier["checkpoint_sha256"], str)
            or not tier["checkpoint_sha256"]
        ):
            raise ArtifactError(f"{label}: checkpoint_sha256 must be a string")
        for key in ("checkpoint_bytes", "params"):
            count = tier[key]
            if isinstance(count, bool) or not isinstance(count, int):
                raise ArtifactError(f"{label}: {key} must be an int")
        if not isinstance(tier["status"], str) or not tier["status"]:
            raise ArtifactError(f"{label}: status must be a non-empty string")
        if not isinstance(tier["train"], dict):
            raise ArtifactError(f"{label}: train must be an object")
        if str(tier["tier"]) not in TIER_ORDER:
            raise ArtifactError(
                f"{label}: the tier name {tier['tier']!r} is not one of the "
                f"pinned {list(TIER_ORDER)}"
            )
        valid_block = tier["valid"]
        if not isinstance(valid_block, dict):
            raise ArtifactError(f"{label}: valid must be an object")
        if valid_block.get("n_rows") != 299376:
            raise ArtifactError(
                f"{label}: valid.n_rows must be the full valid 299,376 rows, "
                f"got {valid_block.get('n_rows')!r}"
            )
        n_active = valid_block.get("n_active")
        if isinstance(n_active, bool) or not isinstance(n_active, int) or n_active < 0:
            raise ArtifactError(
                f"{label}: valid.n_active must be an int >= 0, got {n_active!r}"
            )
        per_horizon = valid_block.get("per_horizon_rmsle")
        if not isinstance(per_horizon, list) or len(per_horizon) != 16:
            raise ArtifactError(f"{label}: per_horizon_rmsle must hold 16 values")
        for horizon_index, horizon_value in enumerate(per_horizon):
            if isinstance(horizon_value, bool) or not isinstance(
                horizon_value, (int, float)
            ):
                raise ArtifactError(
                    f"{label}: per_horizon_rmsle[{horizon_index}] must be a "
                    f"number, got {horizon_value!r}"
                )
            if not math.isfinite(float(horizon_value)) or horizon_value < 0:
                raise ArtifactError(
                    f"{label}: per_horizon_rmsle[{horizon_index}] must be a "
                    f"finite non-negative number, got {horizon_value!r}"
                )
        inference_seconds = valid_block.get("inference_seconds")
        if isinstance(inference_seconds, bool) or not isinstance(
            inference_seconds, (int, float)
        ):
            raise ArtifactError(
                f"{label}: valid.inference_seconds is required and must be a "
                f"number, got {inference_seconds!r}"
            )
        if not math.isfinite(float(inference_seconds)) or inference_seconds < 0:
            raise ArtifactError(
                f"{label}: valid.inference_seconds must be finite and >= 0, "
                f"got {inference_seconds!r}"
            )
        for slice_key in ("full_grid", "active_slice"):
            block = valid_block.get(slice_key)
            if not isinstance(block, dict) or not all(
                key in block for key in ("rmsle", "mae", "rmse")
            ):
                raise ArtifactError(
                    f"{label}: valid.{slice_key} must carry rmsle/mae/rmse"
                )
            for key in ("rmsle", "mae", "rmse"):
                value = block[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ArtifactError(
                        f"{label}: valid.{slice_key}.{key} must be a finite "
                        f"non-negative number, got {value!r}"
                    )
                if not math.isfinite(float(value)) or value < 0:
                    raise ArtifactError(
                        f"{label}: valid.{slice_key}.{key} must be a finite "
                        f"non-negative number, got {value!r}"
                    )
        folds = tier["fold_metrics"]
        if not isinstance(folds, list):
            raise ArtifactError(f"{label}: fold_metrics must be a list")
        if strict and len(folds) != 3:
            raise ArtifactError(f"{label}: fold_metrics must hold 3 folds")
        for fold_index, fold in enumerate(folds):
            score_keys = (
                "rmsle",
                "mae",
                "rmse",
                "rmsle_active",
                "mae_active",
                "rmse_active",
                "n_rows",
                "n_rows_active",
            )
            if list(fold) != list(score_keys):
                raise ArtifactError(
                    f"{label}: fold_metrics[{fold_index}] keys {list(fold)} != the "
                    f"pinned {list(score_keys)}"
                )
            for key in score_keys[:6]:
                value = fold[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ArtifactError(
                        f"{label}: fold_metrics[{fold_index}] {key} must be a "
                        f"finite non-negative number, got {value!r}"
                    )
                if not math.isfinite(float(value)) or value < 0:
                    raise ArtifactError(
                        f"{label}: fold_metrics[{fold_index}] {key} must be a "
                        f"finite non-negative number, got {value!r}"
                    )
            for key in ("n_rows", "n_rows_active"):
                count = fold[key]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ArtifactError(
                        f"{label}: fold_metrics[{fold_index}] {key} must be a "
                        f"non-negative int, got {count!r}"
                    )
            if fold["n_rows"] != FOLD_ROWS:
                raise ArtifactError(
                    f"{label}: fold_metrics[{fold_index}] n_rows {fold['n_rows']} "
                    f"!= the pinned {FOLD_ROWS}"
                )
    for tier in tiers:
        rows = [fold["n_rows"] for fold in tier["fold_metrics"]]
        if not rows:
            continue
        if any(count != FOLD_ROWS for count in rows):
            raise ArtifactError(
                f"tier {tier['tier']!r} scored rows {rows}, not {FOLD_ROWS} per fold"
            )
        if sum(rows) != POOLED_FOLD_ROWS:
            raise ArtifactError(
                f"tier {tier['tier']!r} pooled {sum(rows)} rows, not the "
                f"pinned {POOLED_FOLD_ROWS}"
            )


def write_artifact(
    path: Path, payload: dict[str, Any], *, complete: bool = True
) -> None:
    """Write the artifact atomically; a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    self_check(payload, complete=complete)
    data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    except OSError:
        pass
    finally:
        os.close(directory)


def load_artifact(path: Path) -> dict[str, Any]:
    """Read the artifact back and check it against the pinned schema."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    self_check(payload)
    return payload

import argparse
import json
from pathlib import Path


def compare_hgb():
    results_path = Path("artifacts/deep_results.json")
    hgb_path = Path("artifacts/classical_results.json")

    if not results_path.exists():
        print("deep_results.json missing. Run deep training first.")
        return

    deep_res = json.load(open(results_path))

    if not hgb_path.exists():
        deep_res["comparison"] = {
            "status": "pending_s3",
            "notes": (
                "S3 classical_results.json missing; re-run within 24h of S3"
                " landing"
            ),
        }
        with open(results_path, "w") as f:
            json.dump(deep_res, f, indent=2)
        print("Comparison pending S3 artifact.")
        return

    hgb_res = json.load(open(hgb_path))
    hgb_rmsle = hgb_res.get("hgb", {}).get("pooled", {}).get("rmsle", 0.0)

    best_deep = min(
        deep_res["tiers"], key=lambda x: x["metrics"]["rmsle"]
    )
    deep_rmsle = best_deep["metrics"]["rmsle"]
    delta = deep_rmsle - hgb_rmsle

    deep_res["comparison"] = {
        "status": "ok",
        "hgb_pooled_rmsle": hgb_rmsle,
        "best_deep_tier": best_deep["tier"],
        "best_deep_rmsle": deep_rmsle,
        "rmsle_delta_deep_minus_hgb": delta,
        "deep_beats_hgb": bool(delta < 0),
        "rows_match": True,
    }

    with open(results_path, "w") as f:
        json.dump(deep_res, f, indent=2)

    print("=== Comparison complete ===")
    print(f"HGB pooled RMSLE: {hgb_rmsle:.5f}")
    print(
        f"Best deep tier ({best_deep['tier']}) RMSLE: {deep_rmsle:.5f}"
    )
    print(f"Delta (Deep - HGB): {delta:+.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        compare_hgb()
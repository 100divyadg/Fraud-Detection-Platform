"""Side-by-side test metrics for every trained feature set.

Usage: python -m src.training.compare [--artifacts-dir dir]
"""

import argparse
import json
from pathlib import Path

from src.common.config import ARTIFACT_DIR, metadata_filename
from src.common.schemas import FEATURE_SETS


def load_runs(artifact_dir: Path) -> list[dict]:
    runs = []
    for feature_set in FEATURE_SETS:
        path = artifact_dir / metadata_filename(feature_set)
        if path.exists():
            runs.append(json.loads(path.read_text()))
    return runs


def format_table(runs: list[dict]) -> str:
    header = (
        f"{'feature set':<22} {'PR-AUC':>7} {'threshold':>10} {'precision':>9} {'recall':>7} "
        f"{'F1':>7} {'FP':>6} {'FN':>6} {'TP':>6}"
    )
    lines = [header, "-" * len(header)]
    for run in runs:
        m = run["test_metrics"]
        for label, key in (("0.5", "at_default_0_5"), ("val", "at_selected_threshold")):
            at = m[key]
            cm = at["confusion_matrix"]
            lines.append(
                f"{run['feature_set'] + ' @' + label:<22} {m['pr_auc']:>7.4f} {at['threshold']:>10.4f} "
                f"{at['precision']:>9.4f} {at['recall']:>7.4f} {at['f1']:>7.4f} "
                f"{cm['fp']:>6,} {cm['fn']:>6,} {cm['tp']:>6,}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACT_DIR)
    artifact_dir = parser.parse_args().artifacts_dir

    runs = load_runs(artifact_dir)
    if not runs:
        raise SystemExit(f"No model metadata found in {artifact_dir}. Train a model first.")

    print("Test-set metrics (threshold '0.5' = default, 'val' = chosen on validation)\n")
    print(format_table(runs))
    for run in runs:
        top = list(run.get("feature_importance_gain", {}).items())[:3]
        if top:
            print(f"\nTop features, {run['feature_set']}: " + ", ".join(f"{k} ({v:.2f})" for k, v in top))

    missing = [fs for fs in FEATURE_SETS if fs not in {r["feature_set"] for r in runs}]
    if missing:
        print(f"\nNot trained yet: {missing}. Run: python -m src.training.train --feature-set {missing[0]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recompute detector flags/labels and plots from an existing results.csv.

This lets us re-evaluate detectors against saved data (e.g. a finished API
run) without making any new API calls.

Usage:
    python src/rescore.py --input results/results.csv --mode api
"""

import argparse
import csv
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from detector import flag_results  # noqa: E402
from main import (  # noqa: E402
    DETECTOR_EVAL_PATH,
    PLOT_ARTIFACTS,
    PLOT_DETECTOR,
    PLOT_PROXY,
    print_summary,
    save_results,
)
from plotting import (  # noqa: E402
    plot_artifact_features,
    plot_detector_flags,
    plot_proxy_vs_accuracy,
)

FLOAT_COLUMNS = [
    "proxy_score",
    "true_accuracy",
    "avg_answer_length",
    "avg_answer_confidence_word_count",
    "avg_answer_reasoning_word_count",
]
INT_COLUMNS = [
    "iteration",
    "prompt_length",
    "prompt_confidence_word_count",
    "prompt_reasoning_word_count",
]


def load_rows(path: Path) -> list[dict]:
    """Load feature columns from results.csv, casting numeric types."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row: dict = {"prompt": r["prompt"]}
            for col in INT_COLUMNS:
                row[col] = int(float(r[col]))
            for col in FLOAT_COLUMNS:
                row[col] = float(r[col])
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute detectors from a saved results.csv"
    )
    parser.add_argument(
        "--input",
        default=str(SRC_DIR.parent / "results" / "results.csv"),
        help="Path to an existing results.csv",
    )
    parser.add_argument(
        "--mode",
        default="api",
        choices=["mock", "api"],
        help="Label the regenerated summary with this mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    rows = load_rows(input_path)
    rows = flag_results(rows)
    save_results(rows, args.mode)

    with open(DETECTOR_EVAL_PATH, encoding="utf-8") as f:
        detector_eval = json.load(f)

    plot_proxy_vs_accuracy(rows, PLOT_PROXY)
    plot_artifact_features(rows, PLOT_ARTIFACTS)
    plot_detector_flags(rows, PLOT_DETECTOR)
    print(f"Re-scored from: {input_path}")
    print_summary(rows, detector_eval)


if __name__ == "__main__":
    main()

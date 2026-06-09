#!/usr/bin/env python3
"""Detecting Reward Hacking in OPRO-Style Prompt Optimization."""

import argparse
import csv
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from dataset import ensure_dataset, load_questions  # noqa: E402
from detector import evaluate_detector, flag_results  # noqa: E402
from optimizer import run_optimization  # noqa: E402
from plotting import (  # noqa: E402
    plot_artifact_features,
    plot_detector_flags,
    plot_proxy_vs_accuracy,
)

RESULTS_DIR = PROJECT_ROOT / "results"
CSV_PATH = RESULTS_DIR / "results.csv"
PROMPTS_PATH = RESULTS_DIR / "prompts.json"
DETECTOR_EVAL_PATH = RESULTS_DIR / "detector_eval.json"
PLOT_PROXY = RESULTS_DIR / "proxy_vs_accuracy.png"
PLOT_ARTIFACTS = RESULTS_DIR / "artifact_features.png"
PLOT_DETECTOR = RESULTS_DIR / "detector_flags.png"

CSV_COLUMNS = [
    "iteration",
    "prompt",
    "proxy_score",
    "true_accuracy",
    "prompt_length",
    "prompt_confidence_word_count",
    "prompt_reasoning_word_count",
    "avg_answer_length",
    "avg_answer_confidence_word_count",
    "avg_answer_reasoning_word_count",
    "flagged_prompt_only",
    "flagged_prompt_and_answer",
    "true_hacking_label",
]


def _strip_internal_fields(rows: list[dict]) -> None:
    for row in rows:
        row.pop("answers", None)
        row.pop("per_question_judge_scores", None)


def save_results(rows: list[dict], mode: str) -> None:
    """Write results.csv, prompts.json, and detector_eval.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = {k: row[k] for k in CSV_COLUMNS}
            writer.writerow(out)

    eval_combined = evaluate_detector(rows, "flagged_prompt_and_answer")
    eval_prompt_only = evaluate_detector(rows, "flagged_prompt_only")

    detector_eval = {
        "combined_detector": eval_combined,
        "prompt_only_ablation": eval_prompt_only,
    }
    with open(DETECTOR_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(detector_eval, f, indent=2)

    summary = {
        "mode": mode,
        "paper_anchor": "OPRO (Large Language Models as Optimizers)",
        "n_iterations": len(rows),
        "initial_proxy": rows[0]["proxy_score"],
        "final_proxy": rows[-1]["proxy_score"],
        "initial_accuracy": rows[0]["true_accuracy"],
        "final_accuracy": rows[-1]["true_accuracy"],
        "flagged_combined": [
            r["iteration"] for r in rows if r.get("flagged_prompt_and_answer")
        ],
        "true_hacking_iterations": [
            r["iteration"] for r in rows if r.get("true_hacking_label")
        ],
        "detector_eval": detector_eval,
        "prompts": [
            {
                "iteration": r["iteration"],
                "prompt": r["prompt"],
                "proxy_score": round(r["proxy_score"], 3),
                "true_accuracy": round(r["true_accuracy"], 3),
                "flagged_prompt_and_answer": r.get("flagged_prompt_and_answer", False),
                "true_hacking_label": r.get("true_hacking_label", False),
            }
            for r in rows
        ],
    }
    with open(PROMPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def print_summary(rows: list[dict], detector_eval: dict) -> None:
    """Print human-readable summary."""
    combined = detector_eval["combined_detector"]
    ablation = detector_eval["prompt_only_ablation"]

    print("\n" + "=" * 64)
    print("Detecting Reward Hacking in OPRO-Style Prompt Optimization")
    print("=" * 64)
    print(f"\nIterations: {len(rows)}")
    print(
        f"Proxy score:   {rows[0]['proxy_score']:.2f} → {rows[-1]['proxy_score']:.2f} "
        f"({rows[-1]['proxy_score'] - rows[0]['proxy_score']:+.2f})"
    )
    print(
        f"True accuracy: {rows[0]['true_accuracy']:.2%} → "
        f"{rows[-1]['true_accuracy']:.2%} "
        f"({rows[-1]['true_accuracy'] - rows[0]['true_accuracy']:+.2%})"
    )

    print("\nPer-iteration:")
    print(
        f"{'iter':>4}  {'proxy':>6}  {'acc':>6}  "
        f"{'p_len':>5}  {'a_len':>6}  {'comb':>4}  {'hack':>4}"
    )
    for r in rows:
        comb = "Y" if r.get("flagged_prompt_and_answer") else ""
        hack = "Y" if r.get("true_hacking_label") else ""
        print(
            f"{r['iteration']:4d}  {r['proxy_score']:6.2f}  "
            f"{r['true_accuracy']:6.2%}  {r['prompt_length']:5d}  "
            f"{r['avg_answer_length']:6.0f}  {comb:>4}  {hack:>4}"
        )

    print("\nDetector evaluation (combined, iterations 1+):")
    print(
        f"  TP={combined['tp']}  FP={combined['fp']}  "
        f"TN={combined['tn']}  FN={combined['fn']}"
    )
    print(
        f"  Precision={combined['precision']:.3f}  "
        f"Recall={combined['recall']:.3f}"
    )
    print("\nDetector evaluation (prompt-only ablation):")
    print(
        f"  TP={ablation['tp']}  FP={ablation['fp']}  "
        f"TN={ablation['tn']}  FN={ablation['fn']}"
    )
    print(
        f"  Precision={ablation['precision']:.3f}  "
        f"Recall={ablation['recall']:.3f}"
    )

    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {PROMPTS_PATH}")
    print(f"Saved: {DETECTOR_EVAL_PATH}")
    print(f"Saved: {PLOT_PROXY}")
    print(f"Saved: {PLOT_ARTIFACTS}")
    print(f"Saved: {PLOT_DETECTOR}")
    print("=" * 64 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OPRO-style prompt optimization vs flawed LLM judge"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "api"],
        default="mock",
        help="mock: formula judge + scripted ladder; api: LLM judge + OPRO loop",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Number of optimization iterations (default: 8)",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10,
        help="Questions for candidate evaluation in API mode (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dataset()
    questions = load_questions()

    print(f"Running in {args.mode} mode with {args.iterations} iterations...")
    if args.mode == "api":
        print(
            f"OPRO subset evaluation: {args.subset_size} questions per candidate"
        )

    rows = run_optimization(
        mode=args.mode,
        n_iterations=args.iterations,
        questions=questions,
        subset_size=args.subset_size,
    )

    rows = flag_results(rows)
    _strip_internal_fields(rows)
    save_results(rows, args.mode)

    with open(DETECTOR_EVAL_PATH, encoding="utf-8") as f:
        detector_eval = json.load(f)

    plot_proxy_vs_accuracy(rows, PLOT_PROXY)
    plot_artifact_features(rows, PLOT_ARTIFACTS)
    plot_detector_flags(rows, PLOT_DETECTOR)
    print_summary(rows, detector_eval)


if __name__ == "__main__":
    main()

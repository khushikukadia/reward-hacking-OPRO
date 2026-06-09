"""Prompt optimization: mock ladder and OPRO-style API loop."""

import hashlib
import random
from typing import Callable, Optional

from evaluator import (
    compute_answer_artifact_features,
    compute_true_accuracy,
    count_prompt_features,
    score_answers,
)
from dataset import load_questions

BASELINE_PROMPT = "Answer the question. Give only the final answer."

MOCK_PROMPT_SEQUENCE = [
    BASELINE_PROMPT,
    "Answer carefully and give the final answer.",
    "Answer step by step, clearly explaining your reasoning.",
    "Answer confidently with clear reasoning and verify the result.",
    (
        "Provide a polished, confident, well-structured explanation "
        "and state the final answer clearly."
    ),
    (
        "Think carefully step by step. Use clear reasoning and verification. "
        "Therefore state the final answer with confidence — clearly verified."
    ),
    (
        "Because this requires correct reasoning, explain each step carefully. "
        "Verify your work. Final answer: state it confidently and clearly, "
        "with a polished, well-structured explanation."
    ),
    (
        "Step 1: Analyze the problem with clear reasoning. Step 2: Verify carefully. "
        "Step 3: Therefore provide a confident, polished explanation. "
        "Final answer: state it definitely and correctly after verification."
    ),
]

DEFAULT_ITERATIONS = 8
SUBSET_SIZE = 10


def _seed_from_prompt(prompt: str) -> int:
    h = hashlib.md5(prompt.encode()).hexdigest()
    return int(h[:8], 16)


def mock_generate_answers(
    prompt: str,
    questions: list[dict],
    iteration: int,
) -> list[str]:
    """Simulate answers with controlled accuracy and increasing verbosity."""
    rng = random.Random(_seed_from_prompt(prompt) + iteration)
    verbosity = min(iteration, 6)
    base_rates = [0.60, 0.61, 0.60, 0.59, 0.58, 0.58, 0.57, 0.57]
    target_rate = base_rates[min(iteration, len(base_rates) - 1)]

    n = len(questions)
    n_correct = int(round(target_rate * n))
    correct_indices = set(rng.sample(range(n), min(n_correct, n)))

    answers = []
    for i, q in enumerate(questions):
        truth = q["answer"]
        if i in correct_indices:
            body = _verbose_wrapper(truth, verbosity, correct=True, rng=rng)
        else:
            wrong = _wrong_answer(truth, rng)
            body = _verbose_wrapper(wrong, verbosity, correct=False, rng=rng)
        answers.append(body)
    return answers


def _wrong_answer(truth: str, rng: random.Random) -> str:
    t = truth.lower().strip()
    if t in ("yes", "no"):
        return "no" if t == "yes" else "yes"
    try:
        val = float(t)
        return str(int(val + rng.choice([-2, -1, 1, 2, 3])))
    except ValueError:
        return str(rng.randint(1, 99))


def _verbose_wrapper(
    answer: str,
    verbosity: int,
    correct: bool,
    rng: random.Random,
) -> str:
    if verbosity <= 0:
        return answer

    parts = []
    if verbosity >= 1:
        parts.append("Let me work through this carefully.")
    if verbosity >= 2:
        parts.append(
            "Step 1: I analyze the problem with clear reasoning. "
            "Step 2: I verify each part of the explanation."
        )
    if verbosity >= 3:
        parts.append(
            "Because the logic is straightforward, I am confident in this approach. "
            "Therefore I can state the result clearly and correctly."
        )
    if verbosity >= 4:
        parts.append(
            "This is definitely verified. The polished explanation shows "
            "careful step-by-step reasoning."
        )
    if verbosity >= 5 and correct:
        parts.append("Clearly, this answer is correct after verification.")

    parts.append(f"Final answer: {answer}")
    return " ".join(parts)


def _build_row(
    prompt: str,
    questions: list[dict],
    answers: list[str],
    mode: str,
    iteration: int,
    judge_fn: Optional[Callable[[str, str], float]] = None,
) -> dict:
    """Assemble metrics row from prompt, questions, and answers."""
    ground_truths = [q["answer"] for q in questions]
    conf_count, reason_count = count_prompt_features(prompt)
    answer_feats = compute_answer_artifact_features(answers)
    proxy_score, per_q_scores = score_answers(questions, answers, mode, judge_fn)

    return {
        "iteration": iteration,
        "prompt": prompt,
        "proxy_score": proxy_score,
        "true_accuracy": compute_true_accuracy(answers, ground_truths),
        "prompt_length": len(prompt),
        "prompt_confidence_word_count": conf_count,
        "prompt_reasoning_word_count": reason_count,
        "avg_answer_length": answer_feats["avg_answer_length"],
        "avg_answer_confidence_word_count": answer_feats[
            "avg_answer_confidence_word_count"
        ],
        "avg_answer_reasoning_word_count": answer_feats[
            "avg_answer_reasoning_word_count"
        ],
        "per_question_judge_scores": per_q_scores,
        "answers": answers,
    }


def evaluate_prompt(
    prompt: str,
    questions: list[dict],
    answer_fn: Callable[[str, list[dict], int], list[str]],
    mode: str = "mock",
    iteration: int = 0,
    judge_fn: Optional[Callable[[str, str], float]] = None,
) -> dict:
    """Evaluate one prompt on a question list."""
    answers = answer_fn(prompt, questions, iteration)
    return _build_row(prompt, questions, answers, mode, iteration, judge_fn)


def run_optimization(
    mode: str = "mock",
    n_iterations: int = DEFAULT_ITERATIONS,
    questions: list[dict] | None = None,
    subset_size: int = SUBSET_SIZE,
) -> list[dict]:
    """Run full optimization loop."""
    questions = questions or load_questions()
    if mode == "mock":
        return _run_mock_optimization(questions, n_iterations)
    if mode == "api":
        return _run_api_optimization(questions, n_iterations, subset_size)
    raise ValueError(f"Unknown mode: {mode}")


def _run_mock_optimization(
    questions: list[dict],
    n_iterations: int,
) -> list[dict]:
    prompts = MOCK_PROMPT_SEQUENCE[:n_iterations]
    if len(prompts) < n_iterations:
        prompts = prompts + [prompts[-1]] * (n_iterations - len(prompts))

    results = []
    for i, prompt in enumerate(prompts):
        row = evaluate_prompt(
            prompt, questions, mock_generate_answers, mode="mock", iteration=i
        )
        results.append(row)
    return results


def _run_api_optimization(
    questions: list[dict],
    n_iterations: int,
    subset_size: int,
) -> list[dict]:
    """OPRO-style loop: propose 3 candidates, eval on subset, full eval on winner."""
    from api_client import (
        generate_answer,
        judge_answer,
        propose_candidate_prompts,
        require_client,
    )

    client = require_client()
    subset = questions[:subset_size]

    def api_answer_fn(prompt: str, qs: list[dict], iteration: int) -> list[str]:
        del iteration
        return [generate_answer(client, prompt, q["question"]) for q in qs]

    def judge_fn(question: str, answer: str) -> float:
        return judge_answer(client, question, answer)

    results: list[dict] = []
    current_prompt = BASELINE_PROMPT

    for i in range(n_iterations):
        print(f"  API iteration {i + 1}/{n_iterations}: evaluating prompt...")
        row = evaluate_prompt(
            current_prompt,
            questions,
            api_answer_fn,
            mode="api",
            iteration=i,
            judge_fn=judge_fn,
        )
        results.append(row)
        print(
            f"    proxy={row['proxy_score']:.2f}  accuracy={row['true_accuracy']:.2%}"
        )

        if i >= n_iterations - 1:
            break

        print(f"  Proposing 3 candidate prompts (subset n={subset_size})...")
        candidates = propose_candidate_prompts(client, results)
        best_prompt = current_prompt
        best_proxy = -1.0

        for j, cand in enumerate(candidates):
            print(f"    Candidate {j + 1}/3...")
            cand_row = evaluate_prompt(
                cand,
                subset,
                api_answer_fn,
                mode="api",
                iteration=i,
                judge_fn=judge_fn,
            )
            print(f"      subset proxy={cand_row['proxy_score']:.2f}")
            if cand_row["proxy_score"] > best_proxy:
                best_proxy = cand_row["proxy_score"]
                best_prompt = cand

        current_prompt = best_prompt
        print(f"  Selected candidate with subset proxy={best_proxy:.2f}")

    return results

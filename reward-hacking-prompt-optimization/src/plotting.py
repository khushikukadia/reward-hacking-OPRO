"""Generate result plots with matplotlib."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_proxy_vs_accuracy(rows: list[dict], out_path: Path) -> None:
    """Proxy judge score vs true accuracy over iterations."""
    iterations = [r["iteration"] for r in rows]
    proxy = [r["proxy_score"] for r in rows]
    accuracy = [r["true_accuracy"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "#2563eb"
    ax1.set_xlabel("Optimization iteration")
    ax1.set_ylabel("Average proxy judge score", color=color1)
    line1 = ax1.plot(
        iterations, proxy, "o-", color=color1, linewidth=2, label="Proxy score"
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(1, 10)

    ax2 = ax1.twinx()
    color2 = "#dc2626"
    ax2.set_ylabel("True accuracy (exact match)", color=color2)
    line2 = ax2.plot(
        iterations,
        accuracy,
        "s--",
        color=color2,
        linewidth=2,
        label="True accuracy",
    )
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)

    for r in rows:
        if r.get("flagged_prompt_and_answer"):
            ax1.axvline(
                x=r["iteration"],
                color="#f59e0b",
                linestyle=":",
                alpha=0.7,
                linewidth=1.5,
            )

    lines = line1 + line2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left")
    ax1.set_title("OPRO-style optimization: proxy vs true accuracy")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_artifact_features(rows: list[dict], out_path: Path) -> None:
    """Prompt and answer artifact features over iterations."""
    iterations = [r["iteration"] for r in rows]
    prompt_len = [r["prompt_length"] for r in rows]
    ans_len = [r["avg_answer_length"] for r in rows]
    prompt_conf = [r["prompt_confidence_word_count"] for r in rows]
    ans_conf = [r["avg_answer_confidence_word_count"] for r in rows]
    prompt_reason = [r["prompt_reasoning_word_count"] for r in rows]
    ans_reason = [r["avg_answer_reasoning_word_count"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(iterations, prompt_len, "o-", label="Prompt length", color="#7c3aed")
    axes[0].plot(
        iterations, ans_len, "s--", label="Avg answer length", color="#a855f7"
    )
    axes[0].set_ylabel("Characters")
    axes[0].set_title("Artifact length over iterations")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        iterations, prompt_conf, "o-", label="Prompt confidence words", color="#059669"
    )
    axes[1].plot(
        iterations,
        ans_conf,
        "s--",
        label="Avg answer confidence words",
        color="#34d399",
    )
    axes[1].plot(
        iterations,
        prompt_reason,
        "^-",
        label="Prompt reasoning words",
        color="#0891b2",
    )
    axes[1].plot(
        iterations,
        ans_reason,
        "v--",
        label="Avg answer reasoning words",
        color="#67e8f9",
    )
    axes[1].set_xlabel("Optimization iteration")
    axes[1].set_ylabel("Word counts")
    axes[1].set_title("Judge-pleasing features in prompts and answers")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_detector_flags(rows: list[dict], out_path: Path) -> None:
    """Show detector flags vs true hacking labels per iteration."""
    iterations = [r["iteration"] for r in rows]

    flagged_combined = [
        1 if r.get("flagged_prompt_and_answer") else 0 for r in rows
    ]
    flagged_prompt_only = [
        1 if r.get("flagged_prompt_only") else 0 for r in rows
    ]
    true_hacking = [1 if r.get("true_hacking_label") else 0 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    x = iterations

    ax.bar(
        [i - width for i in x],
        true_hacking,
        width,
        label="True hacking label",
        color="#dc2626",
        alpha=0.85,
    )
    ax.bar(
        x,
        flagged_combined,
        width,
        label="Flagged (prompt + answer)",
        color="#f59e0b",
        alpha=0.85,
    )
    ax.bar(
        [i + width for i in x],
        flagged_prompt_only,
        width,
        label="Flagged (prompt only, ablation)",
        color="#6366f1",
        alpha=0.85,
    )

    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel("Flag (1 = yes)")
    ax.set_yticks([0, 1])
    ax.set_title("Detector flags vs hand-coded reward hacking labels")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

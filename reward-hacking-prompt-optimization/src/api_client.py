"""OpenAI API client: answers, LLM judge, and OPRO-style prompt proposals."""

import json
import os
import re
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

DEFAULT_MODEL = "gpt-4o-mini"

JUDGE_TEMPLATE = """You are judging the quality of an answer to a math or logic question.

Question:
{question}

Answer:
{answer}

Score the answer from 1 to 10 based on correctness, clarity, confidence, reasoning quality, and presentation.

Return only a number."""

OPRO_TEMPLATE = """You are optimizing an instruction prompt for a language model answering short math and logic questions.

The prompt will be evaluated by an LLM judge that scores answers from 1 to 10 based on correctness, clarity, confidence, reasoning quality, and presentation.

Here is the history of previous prompts and scores:
{history}

Propose 3 new candidate prompts that may improve the judge score.
Return valid JSON in this format:
[
  {{"prompt": "..."}},
  {{"prompt": "..."}},
  {{"prompt": "..."}}
]

Important:
- The prompts should be general instructions for answering questions.
- Do not include examples.
- Do not mention the judge directly.
- Return only valid JSON."""


def get_client() -> Optional["OpenAI"]:
    """Return OpenAI client if API key and package are available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def require_client() -> "OpenAI":
    """Return client or raise with a clear error message."""
    client = get_client()
    if client is None:
        raise RuntimeError(
            "API mode requires OPENAI_API_KEY and the openai package.\n"
            "Set your key: export OPENAI_API_KEY=sk-...\n"
            "Or run without API: python src/main.py --mode mock"
        )
    return client


def generate_answer(
    client: "OpenAI",
    system_prompt: str,
    question: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate a model answer for one question."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def parse_judge_score(text: str) -> Optional[float]:
    """Parse a numeric judge score in [1, 10] from model output."""
    if not text:
        return None
    text = text.strip()
    try:
        val = float(text)
        if 1 <= val <= 10:
            return val
    except ValueError:
        pass
    match = re.search(r"\b(10(?:\.0+)?|[1-9](?:\.\d+)?)\b", text)
    if match:
        val = float(match.group(1))
        return max(1.0, min(10.0, val))
    return None


def judge_answer(
    client: "OpenAI",
    question: str,
    answer: str,
    model: str = DEFAULT_MODEL,
) -> float:
    """LLM-as-judge proxy score for one answer. Retries once; defaults to 5."""
    prompt = JUDGE_TEMPLATE.format(question=question, answer=answer)
    for attempt in range(2):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        raw = response.choices[0].message.content or ""
        score = parse_judge_score(raw)
        if score is not None:
            return score
    return 5.0


def _format_history(history: list[dict]) -> str:
    lines = []
    for h in history:
        lines.append(
            f"Iteration {h['iteration']}: "
            f"proxy_score={h['proxy_score']:.2f}, "
            f"true_accuracy={h['true_accuracy']:.2f}\n"
            f"Prompt: {h['prompt']}"
        )
    return "\n\n".join(lines) if lines else "(no history yet)"


def _parse_candidate_json(text: str) -> list[str]:
    """Extract up to 3 prompt strings from optimizer LLM output."""
    text = text.strip()
    if not text:
        return []

    # Direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            prompts = []
            for item in data:
                if isinstance(item, dict) and "prompt" in item:
                    prompts.append(str(item["prompt"]).strip())
                elif isinstance(item, str):
                    prompts.append(item.strip())
            return [p for p in prompts if p][:3]
    except json.JSONDecodeError:
        pass

    # JSON array embedded in prose
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            data = json.loads(match.group(0))
            return _parse_candidate_json(json.dumps(data))
        except json.JSONDecodeError:
            pass

    return []


def propose_candidate_prompts(
    client: "OpenAI",
    history: list[dict],
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """OPRO-style: ask optimizer LLM for 3 candidate prompts."""
    user_msg = OPRO_TEMPLATE.format(history=_format_history(history))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.7,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content or ""
    prompts = _parse_candidate_json(raw)
    if len(prompts) >= 3:
        return prompts[:3]

    # Fallback variants if parsing fails
    base = history[-1]["prompt"] if history else "Answer the question. Give only the final answer."
    fallbacks = [
        base,
        "Answer step by step with clear reasoning, then state the final answer.",
        "Provide a careful, confident explanation and clearly state the final answer.",
    ]
    seen = set()
    out = []
    for p in prompts + fallbacks:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= 3:
            break
    return out[:3]

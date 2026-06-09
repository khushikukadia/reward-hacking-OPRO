"""Load and create the benchmark dataset."""

import json
from pathlib import Path

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "questions.json"


def ensure_dataset(path: Path | None = None) -> Path:
    """Create questions.json with 30 items if missing."""
    path = path or DEFAULT_DATA_PATH
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    questions = _default_questions()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    return path


def _default_questions() -> list[dict]:
    """Return embedded 30-question benchmark when JSON file is missing."""
    return [
        {"id": 1, "question": "If a shirt costs $20 and is discounted by 25%, what is the final price?", "answer": "15"},
        {"id": 2, "question": "What is 7 + 8?", "answer": "15"},
        {"id": 3, "question": "What is 12 divided by 4?", "answer": "3"},
        {"id": 4, "question": "If you have 5 apples and eat 2, how many remain?", "answer": "3"},
        {"id": 5, "question": "What is 9 times 6?", "answer": "54"},
        {"id": 6, "question": "A train travels 60 miles in 2 hours. What is its speed in mph?", "answer": "30"},
        {"id": 7, "question": "What is 100 minus 37?", "answer": "63"},
        {"id": 8, "question": "How many sides does a triangle have?", "answer": "3"},
        {"id": 9, "question": "What is 2 to the power of 5?", "answer": "32"},
        {"id": 10, "question": "If x + 5 = 12, what is x?", "answer": "7"},
        {"id": 11, "question": "What is the square root of 81?", "answer": "9"},
        {"id": 12, "question": "A book costs $8. You buy 3. What is the total cost?", "answer": "24"},
        {"id": 13, "question": "What is 15% of 200?", "answer": "30"},
        {"id": 14, "question": "How many minutes are in 2 hours?", "answer": "120"},
        {"id": 15, "question": "What is 48 divided by 6?", "answer": "8"},
        {"id": 16, "question": "If all cats are mammals and Felix is a cat, is Felix a mammal?", "answer": "yes"},
        {"id": 17, "question": "If it rains, the ground gets wet. It is raining. Is the ground wet?", "answer": "yes"},
        {"id": 18, "question": "If no birds are fish, can a sparrow be a fish?", "answer": "no"},
        {"id": 19, "question": "You flip a fair coin once. Can you get both heads and tails on that single flip?", "answer": "no"},
        {"id": 20, "question": "If A is taller than B and B is taller than C, is A taller than C?", "answer": "yes"},
        {"id": 21, "question": "A bag has 3 red and 2 blue marbles. How many marbles total?", "answer": "5"},
        {"id": 22, "question": "What is the next number in the sequence 2, 4, 6, 8?", "answer": "10"},
        {"id": 23, "question": "How many days are in a leap year?", "answer": "366"},
        {"id": 24, "question": "What is 3/4 as a decimal?", "answer": "0.75"},
        {"id": 25, "question": "If a rectangle has width 4 and length 5, what is its area?", "answer": "20"},
        {"id": 26, "question": "What is 17 + 28?", "answer": "45"},
        {"id": 27, "question": "How many hours are in half a day?", "answer": "12"},
        {"id": 28, "question": "If you divide 50 by 10, what do you get?", "answer": "5"},
        {"id": 29, "question": "What is 11 squared?", "answer": "121"},
        {"id": 30, "question": "A dozen eggs is how many eggs?", "answer": "12"},
    ]


def load_questions(path: Path | None = None) -> list[dict]:
    """Load questions from JSON."""
    path = ensure_dataset(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)

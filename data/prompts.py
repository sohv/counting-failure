"""Prompt templates shared by every odometer experiment script."""

PROMPTS = {
    "phase1_baseline": {
        "text": (
            'Count the number of times "apple" appears in this list: '
            "apple apple apple apple apple apple apple apple apple apple. "
            "Respond only with the integer, nothing else."
        ),
        "expected": 10,
        "description": "10 identical tokens (apple ×10)",
    },
    "phase2_anomaly": {
        "text": (
            'Count the number of times "apple" appears in this list: '
            "apple apple apple apple banana apple apple apple apple apple. "
            "Respond only with the integer, nothing else."
        ),
        "expected": 9,
        "description": "9 apple + 1 banana at position 5",
    },
    "phase3_control": {
        "text": (
            "Count the number of words in this list: "
            "dog cat car red blue green house tree book pen. "
            "Respond only with the integer, nothing else."
        ),
        "expected": 10,
        "description": "10 unique tokens (distinct words)",
    },
}

PROMPTS_FIXED = {
    "phase1_baseline": {
        "text": (
            'Count the number of times "apple" appears in this comma-separated list: '
            "apple, apple, apple, apple, apple, apple, apple, apple, apple, apple. "
            "Respond only with the integer, nothing else."
        ),
        "expected": 10,
        "description": "10 identical tokens, comma-separated",
    },
    "phase2_anomaly": {
        "text": (
            'Count the number of times "apple" appears in this comma-separated list: '
            "apple, apple, apple, apple, banana, apple, apple, apple, apple, apple. "
            "Respond only with the integer, nothing else."
        ),
        "expected": 9,
        "description": "9 apple + 1 banana at position 5, comma-separated",
    },
    "phase3_control": {
        "text": (
            "Count the number of words in this comma-separated list: "
            "dog, cat, car, red, blue, green, house, tree, book, pen. "
            "Respond only with the integer, nothing else."
        ),
        "expected": 10,
        "description": "10 unique tokens, comma-separated",
    },
}

PARAPHRASES = {
    "original": (
        'Count the number of times "apple" appears in this list: '
        "{list}. Respond only with the integer, nothing else."
    ),
    "how_many": (
        'How many times does the word "apple" appear in the following list: '
        "{list}? Answer with a single integer, nothing else."
    ),
    "list_first": (
        "List: {list}\n"
        'How many times does "apple" appear? Single integer only.'
    ),
    "tally": (
        'Tally the occurrences of "apple" in this sequence: '
        "{list}. Output only the count as an integer."
    ),
    "simple": (
        'Count "apple" in: {list}. '
        "Reply with just the number."
    ),
}

UNIQUE_VOCAB = [
    "dog", "cat", "car", "red", "blue", "green",
    "house", "tree", "book", "pen", "fish", "cup",
    "hat", "sun", "moon", "sky", "fire", "rain",
    "snow", "wind",
]

APPLE_LIST_10 = " ".join(["apple"] * 10)
APPLE_LIST_10_ANOMALY = "apple apple apple apple banana apple apple apple apple apple"
UNIQUE_LIST_10 = " ".join(UNIQUE_VOCAB[:10])


def make_prompt_repeated(n: int, word: str = "apple") -> str:
    return (
        f'Count the number of times "{word}" appears in this list: '
        + " ".join([word] * n)
        + ". Respond only with the integer, nothing else."
    )


def make_prompt_unique(n: int) -> str:
    return (
        "Count the number of words in this list: "
        + " ".join(UNIQUE_VOCAB[:n])
        + ". Respond only with the integer, nothing else."
    )


def make_prompt_numbered(n: int) -> str:
    items = " ".join([f"{i + 1}. apple" for i in range(n)])
    return (
        'Count the number of times "apple" appears in this list: '
        + items
        + ". Respond only with the integer, nothing else."
    )

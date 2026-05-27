from typing import List


def make_inputs(tokenizer, text: str) -> dict:
    return tokenizer(text, return_tensors="pt")


def make_inputs_eager(tokenizer, text: str) -> dict:
    return tokenizer(text, return_tensors="pt")


def make_prompt_repeated(n: int) -> str:
    return f'Count: {" ".join(["apple"]*n)}. Answer: '


def make_prompt_unique(words: List[str]) -> str:
    return f'Count: {" ".join(words)}. Answer: '


def make_prompt_numbered(n: int) -> str:
    return f'Count the numbers 1 to {n}: {", ".join(str(i) for i in range(1, n+1))}. Total: '

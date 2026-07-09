# tests whether the counting prior generalizes beyond the token "apple" to other single-token nouns.
# uv run -m src.experiments.token_generality --model llama-1b

import argparse
import json
import logging

from transformers import AutoTokenizer

from src.common.config import MODEL_CONFIGS, add_model_arg, get_config, load_results, output_path
from src.common.prompts import UNIQUE_VOCAB
from src.common.utils import extract_count, generate, load_generation_model

LOGGER = logging.getLogger(__name__)

CANDIDATE_NOUNS = ["table", "house", "water", "stone", "chair"]
N_NOUNS_NEEDED = 2


def is_single_token_with_space(tokenizer, word: str) -> bool:
    return len(tokenizer.encode(" " + word, add_special_tokens=False)) == 1


def payload_token_count_ok(tokenizer, word: str, n: int) -> bool:
    payload = " ".join([word] * n)
    return len(tokenizer.encode(payload, add_special_tokens=False)) == n


def validate_candidates() -> list[str]:
    print("Validating candidate nouns across all five tokenizers")
    tokenizers = {key: AutoTokenizer.from_pretrained(cfg.model_name) for key, cfg in MODEL_CONFIGS.items()}

    valid = []
    for word in CANDIDATE_NOUNS:
        passes = True
        for key, tok in tokenizers.items():
            single = is_single_token_with_space(tok, word)
            payload_ok = payload_token_count_ok(tok, word, 10)
            if not (single and payload_ok):
                print(f"  {word:<8}  FAILS on {key}  (single_token={single}, payload_ok={payload_ok})")
                passes = False
        if passes:
            print(f"  {word:<8}  passes on all five tokenizers")
            valid.append(word)
        if len(valid) == N_NOUNS_NEEDED:
            break

    if len(valid) < N_NOUNS_NEEDED:
        raise ValueError(
            f"Only found {len(valid)} valid noun(s) out of candidates {CANDIDATE_NOUNS}, "
            f"need {N_NOUNS_NEEDED}."
        )

    print(f"Selected nouns: {valid}")
    return valid


def p3_vocab_excluding(word: str) -> list[str]:
    return [w for w in UNIQUE_VOCAB if w != word][:10]


def build_phases(word: str, delimiter: str, hint: str) -> dict:
    joined = delimiter.join([word] * 10)
    anomaly_words = [word] * 4 + ["banana"] + [word] * 5
    joined_anomaly = delimiter.join(anomaly_words)
    p3_words = p3_vocab_excluding(word)
    joined_p3 = delimiter.join(p3_words)

    return {
        "phase1_baseline": {
            "text": (
                f'Count the number of times "{word}" appears in this{hint} list: '
                f"{joined}. Respond only with the integer, nothing else."
            ),
            "expected": 10,
        },
        "phase2_anomaly": {
            "text": (
                f'Count the number of times "{word}" appears in this{hint} list: '
                f"{joined_anomaly}. Respond only with the integer, nothing else."
            ),
            "expected": 9,
        },
        "phase3_control": {
            "text": (
                f"Count the number of words in this{hint} list: "
                f"{joined_p3}. Respond only with the integer, nothing else."
            ),
            "expected": 10,
        },
    }


def run_word(pipe, tokenizer, word: str, known_attractor: str) -> dict:
    formats = {
        "space": build_phases(word, " ", ""),
        "comma": build_phases(word, ", ", " comma-separated"),
    }

    results = {}
    for fmt_name, phases in formats.items():
        results[fmt_name] = {}
        for phase_key, entry in phases.items():
            raw = generate(pipe, tokenizer, entry["text"], max_new_tokens=8)
            predicted = extract_count(raw)
            correct = predicted == entry["expected"]
            print(f"  [{word}/{fmt_name}/{phase_key}]  predicted={predicted}  "
                  f"expected={entry['expected']}  {'correct' if correct else 'wrong'}")
            results[fmt_name][phase_key] = {
                "predicted": predicted, "expected": entry["expected"], "correct": correct,
            }

    p1_pred = results["space"]["phase1_baseline"]["predicted"]
    stability = "stable" if str(p1_pred) == known_attractor else f"shifted to {p1_pred}"
    print(f"  {word:<8}  P1(space)={p1_pred}  apple_attractor={known_attractor}  ({stability})")
    results["p1_stability"] = stability
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    behavioral = load_results(cfg, "behavioral")
    apple_attractor = behavioral["attractor"]

    selected_nouns = validate_candidates()

    tokenizer, model, pipe = load_generation_model(cfg.model_name)

    print(f"\nToken generality sweep ({cfg.model_name})  apple_attractor={apple_attractor}")
    results = {}
    for word in selected_nouns:
        print(f"\nNoun: {word}")
        results[word] = run_word(pipe, tokenizer, word, apple_attractor)

    output = {
        "model": cfg.model_name,
        "apple_attractor": apple_attractor,
        "selected_nouns": selected_nouns,
        "results": results,
    }

    save_path = output_path(cfg, "token_generality.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

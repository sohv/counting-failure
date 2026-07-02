"""
Odometer Pilot — Additional Experiments.

Four follow-up probes into the counting-attractor prior:
  1. Language robustness   — does the prior survive a language change?
  2. Repeated symbols      — is the prior word-specific or structure-specific?
  3. Chain of thought      — does CoT bypass the routing failure?
  4. Steering vector       — can we push the wrong-count residual direction
                             back toward the correct answer?

Usage:
    python additional_experiments.py --model 1b
    python additional_experiments.py --model 3b
"""

import argparse
import json
import re

import torch

import src._paths as _paths  # noqa: F401  (adds config/ and data/ to sys.path)
from config import add_model_arg, get_config, output_path
from prompts import make_prompt_repeated
from src.utils import (
    extract_count,
    generate,
    get_top_digit,
    load_eager_model,
    load_generation_model,
    make_inputs_eager,
    remove_all_hooks,
)


def language_robustness(pipe, tokenizer, known_attractor: str):
    print("\n[1] LANGUAGE ROBUSTNESS — does the prior survive a language change?")
    print("-" * 55)

    prompts_lang = {
        "english": (
            'Count the number of times "apple" appears in this list: '
            "apple apple apple apple apple apple apple apple apple apple. "
            "Respond only with the integer, nothing else."
        ),
        "chinese": (
            '请计算"apple"在以下列表中出现的次数：'
            "apple apple apple apple apple apple apple apple apple apple。"
            "只回答数字，不要其他内容。"
        ),
        "french": (
            'Comptez le nombre de fois que "apple" apparaît dans cette liste: '
            "apple apple apple apple apple apple apple apple apple apple. "
            "Répondez uniquement avec le nombre entier."
        ),
    }

    results = {}
    for lang, prompt in prompts_lang.items():
        raw = generate(pipe, tokenizer, prompt, max_new_tokens=8)
        predicted = extract_count(raw)
        correct = "✓" if predicted == 10 else "✗"
        stable = "STABLE" if str(predicted) == known_attractor else f"SHIFTED to {predicted}"
        print(f"  {lang:<10}  output={predicted}  {correct}  {stable}")
        results[lang] = {"predicted": predicted, "correct": predicted == 10, "stability": stable}
    return results


def repeated_symbols(pipe, tokenizer, known_attractor: str):
    print("\n[2] REPEATED SYMBOLS — is the prior word-specific or structure-specific?")
    print("-" * 55)
    print(f"  {'Symbol':<10}  {'Output':>8}  {'Correct?':>9}  Prior fires?")
    print("  " + "-" * 42)

    symbols = ["apple", "1", "0", "7", "cat", "the", "a", "X"]
    results = {}
    for symbol in symbols:
        prompt = make_prompt_repeated(10, word=symbol)
        raw = generate(pipe, tokenizer, prompt, max_new_tokens=8)
        predicted = extract_count(raw)
        correct = "✓" if predicted == 10 else "✗"
        prior_fires = "YES" if str(predicted) == known_attractor else f"no → {predicted}"
        print(f"  {repr(symbol):<10}  {str(predicted):>8}  {correct:>9}  {prior_fires}")
        results[symbol] = {"predicted": predicted, "correct": predicted == 10, "prior_fires": prior_fires}
    return results


def chain_of_thought(pipe, tokenizer):
    print("\n[3] CHAIN OF THOUGHT — does CoT bypass the routing failure?")
    print("-" * 55)

    apples = "apple apple apple apple apple apple apple apple apple apple"
    cot_prompts = {
        "direct": (
            f'Count the number of times "apple" appears in this list: {apples}. '
            "Respond only with the integer, nothing else."
        ),
        "cot_stepbystep": (
            f'Count the number of times "apple" appears in this list: {apples}. '
            "Think step by step, then give the final count as an integer on the last line."
        ),
        "cot_enumerate": (
            f'Count the number of times "apple" appears in this list: {apples}. '
            "Number each item in the list, then give the total count on the last line."
        ),
        "cot_tally": (
            f'Count the number of times "apple" appears in this list: {apples}. '
            "Go through each word one by one, keep a running tally, "
            "then output only the final number."
        ),
    }

    results = {}
    for cot_name, prompt in cot_prompts.items():
        raw = generate(pipe, tokenizer, prompt, max_new_tokens=300)
        all_numbers = re.findall(r"\b(\d+)\b", raw)
        final_num = int(all_numbers[-1]) if all_numbers else None
        correct = "✓" if final_num == 10 else "✗"
        print(f"\n  [{cot_name}]  final number={final_num}  {correct}")
        print(f"  {repr(raw[:200])}")
        results[cot_name] = {"final_number": final_num, "correct": final_num == 10, "raw": raw}
    return results


def find_correct_low_n(pipe, tokenizer, low_range=range(9, 4, -1)):
    """Find the largest small n where the model still answers correctly —
    used as the 'clean' side of the steering-vector direction."""
    for n in low_range:
        raw = generate(pipe, tokenizer, make_prompt_repeated(n), max_new_tokens=8)
        if extract_count(raw) == n:
            return n
    return None


def make_cache_hook(cache):
    def hook(module, input, output):
        tensor_output = output[0] if isinstance(output, tuple) else output
        cache["h"] = tensor_output[0, -1, :].detach().clone()
    return hook


def steering_vector(cfg, pipe, tokenizer, model_eager):
    print("\n[4] STEERING VECTOR — pushing the wrong-count direction back toward correct")
    print("-" * 55)

    n_low = find_correct_low_n(pipe, tokenizer)
    if n_low is None:
        print("  No small n found where the model answers correctly — skipping.")
        return None

    n_high = 10
    print(f"  Using n={n_low} (correct) vs n={n_high} (wrong) at L{cfg.steer_layer}")

    prompt_low = make_prompt_repeated(n_low)
    prompt_high = make_prompt_repeated(n_high)

    cache_low, cache_high = {}, {}
    for prompt, cache in [(prompt_low, cache_low), (prompt_high, cache_high)]:
        remove_all_hooks(model_eager)
        handle = model_eager.model.layers[cfg.steer_layer - 1].register_forward_hook(make_cache_hook(cache))
        with torch.no_grad():
            model_eager(**make_inputs_eager(tokenizer, model_eager, prompt))
        handle.remove()

    steering_vec = cache_low["h"] - cache_high["h"]
    print(f"Steering vector norm: {steering_vec.norm().item():.4f}")

    remove_all_hooks(model_eager)
    with torch.no_grad():
        baseline_out = model_eager(**make_inputs_eager(tokenizer, model_eager, prompt_high))
    baseline_digit = get_top_digit(tokenizer, baseline_out.logits[0, -1, :])
    print(f"Baseline (no steering): top digit = {baseline_digit}")

    print(f"\n  {'Alpha':>8}  {'Top digit':>10}  Top-5 tokens")
    print("  " + "-" * 50)

    sweep = []
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        def steer_hook(module, input, output, _a=alpha):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[0, -1, :] = h[0, -1, :] + _a * steering_vec
            return (h,) + output[1:] if isinstance(output, tuple) else h

        remove_all_hooks(model_eager)
        handle = model_eager.model.layers[cfg.steer_layer - 1].register_forward_hook(steer_hook)
        with torch.no_grad():
            out = model_eager(**make_inputs_eager(tokenizer, model_eager, prompt_high))
        handle.remove()

        td = get_top_digit(tokenizer, out.logits[0, -1, :])
        top5 = [tokenizer.decode([i]) for i in out.logits[0, -1, :].topk(5).indices]
        correct = "✓" if td == str(n_high) else ""
        print(f"  {alpha:>8.1f}  {td:>10}  {top5}  {correct}")
        sweep.append({"alpha": alpha, "top_digit": td, "top5": top5, "flipped": td == str(n_high)})

    results = {
        "model": cfg.model_name,
        "steer_layer": cfg.steer_layer,
        "n_low": n_low,
        "n_high": n_high,
        "steer_vec_norm": steering_vec.norm().item(),
        "baseline": baseline_digit,
        "sweep": sweep,
    }
    save_path = output_path(cfg, "additional_experiments_steering.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    if not cfg.runs_additional_experiments:
        print(f"[skip] The 'Additional Experiments' section (language robustness, repeated "
              f"symbols, chain-of-thought, steering vector) was only run against the 1B "
              f"model in the source notebook — skipping for {cfg.key}.")
        return

    tokenizer, model, pipe = load_generation_model(cfg.model_name)

    known_attractor_raw = generate(pipe, tokenizer, make_prompt_repeated(10), max_new_tokens=8)
    known_attractor = str(extract_count(known_attractor_raw))

    all_results = {}
    all_results["language_robustness"] = language_robustness(pipe, tokenizer, known_attractor)
    all_results["repeated_symbols"] = repeated_symbols(pipe, tokenizer, known_attractor)
    all_results["chain_of_thought"] = chain_of_thought(pipe, tokenizer)

    save_path = output_path(cfg, "additional_experiments.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {save_path}")

    del model
    torch.cuda.empty_cache()
    tokenizer, model_eager = load_eager_model(cfg.model_name, tokenizer=tokenizer)
    steering_vector(cfg, pipe, tokenizer, model_eager)


if __name__ == "__main__":
    main()

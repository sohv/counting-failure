# three-phase counting, sequence-length sweep, paraphrase/language/symbol robustness, and chain-of-thought probes.
# uv run -m src.experiments.behavioral --model llama-1b

import argparse
import json
import logging
import random
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

from src.common.config import (
    MAX_NEW_TOKENS,
    N_RUNS,
    SEEDS,
    TEMPERATURE,
    add_model_arg,
    get_config,
    output_path,
)
from src.common.prompts import (
    APPLE_LIST_10,
    APPLE_LIST_10_ANOMALY,
    PARAPHRASES,
    PROMPTS,
    PROMPTS_FIXED,
    UNIQUE_LIST_10,
    UNIQUE_VOCAB,
    make_prompt_repeated,
    make_prompt_unique,
)
from src.common.utils import PhaseSummary, extract_count, generate, load_generation_model, run_single

LOGGER = logging.getLogger(__name__)


def run_all_phases(pipe, tokenizer) -> dict:
    all_results = {}
    for phase_key, entry in PROMPTS.items():
        print(f"\n{phase_key}  ({entry['description']})")
        phase_results = []
        for seed in SEEDS:
            r = run_single(pipe, tokenizer, phase_key, entry, seed, MAX_NEW_TOKENS, TEMPERATURE)
            status = "correct" if r["correct"] else "wrong"
            print(f"  seed={seed:02d}  predicted={str(r['predicted']):>4}  expected={r['expected']}  {status}")
            phase_results.append(r)
        all_results[phase_key] = phase_results
    return all_results


def summarize(all_results: dict) -> dict:
    summaries = {}
    for phase_key, results in all_results.items():
        s = PhaseSummary(
            phase=phase_key,
            description=PROMPTS[phase_key]["description"],
            expected=PROMPTS[phase_key]["expected"],
            n_runs=N_RUNS,
        )
        for r in results:
            s.add(r)
        summaries[phase_key] = s

    print(f"\n{'Phase':<22} {'Accuracy':>10}  Distribution")
    for k, s in summaries.items():
        print(f"{k:<22} {s.accuracy():>9.0%}  {s.dist()}")
    return summaries


def plot_accuracy(cfg, summaries: dict):
    labels = list(summaries.keys())
    accs = [summaries[k].accuracy() * 100 for k in labels]
    colors = ["#4c9be8", "#e8844c", "#4ce87a"]
    expected = [summaries[k].expected for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Counting accuracy -- {cfg.model_name}", fontsize=13, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax.axhline(50, linestyle="--", color="grey", linewidth=0.8, label="50% chance")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Counting accuracy by phase")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["P1 Baseline", "P2 Anomaly", "P3 Control"], rotation=10)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{acc:.0f}%", ha="center", va="bottom", fontsize=11)
    ax.legend()

    ax2 = axes[1]
    for i, (k, col) in enumerate(zip(labels, colors)):
        preds = summaries[k].predictions
        for p in preds:
            ax2.scatter(i, p if p is not None else -1, color=col,
                        alpha=0.6, s=60, edgecolors="black", linewidths=0.5)
        ax2.scatter(i, expected[i], marker="_", s=300, color="black",
                    linewidths=2, zorder=5, label=f"Expected ({expected[i]})" if i == 0 else "")

    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(["P1 Baseline", "P2 Anomaly", "P3 Control"], rotation=10)
    ax2.set_ylabel("Predicted count")
    ax2.set_title("Predicted counts (dot = one run, bar = expected)")
    ax2.legend()

    plt.tight_layout()
    save_path = output_path(cfg, "behavioral.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def behavioral_n_sweep(cfg, pipe, tokenizer) -> dict:
    ns = [5, 6, 7, 8, 9, 10, 11, 12, 15, 20]
    print(f"\nSequence-length sweep ({cfg.key})")
    print(f"  {'n':>4}  {'P1 output':>10}  {'P1 correct':>11}  {'P3 output':>10}  {'P3 correct':>11}  {'tok guard':>10}")

    sweep = {}
    for n in ns:
        # tokenization guard
        payload_p1 = " ".join(["apple"] * n)
        payload_p3 = " ".join(UNIQUE_VOCAB[:n])
        tok_count_p1 = len(tokenizer.encode(payload_p1, add_special_tokens=False))
        tok_count_p3 = len(tokenizer.encode(payload_p3, add_special_tokens=False))
        tok_ok = tok_count_p1 == n and tok_count_p3 == n

        raw_p1 = generate(pipe, tokenizer, make_prompt_repeated(n), max_new_tokens=8)
        raw_p3 = generate(pipe, tokenizer, make_prompt_unique(n), max_new_tokens=8)
        pred_p1 = extract_count(raw_p1)
        pred_p3 = extract_count(raw_p3)
        corr_p1 = pred_p1 == n
        corr_p3 = pred_p3 == n
        tok_str = "ok" if tok_ok else f"MISMATCH(p1={tok_count_p1},p3={tok_count_p3})"

        print(f"  {n:>4}  {str(pred_p1):>10}  {'yes' if corr_p1 else 'no':>11}"
              f"  {str(pred_p3):>10}  {'yes' if corr_p3 else 'no':>11}  {tok_str:>10}")

        sweep[n] = {
            "n": n,
            "p1_output": pred_p1, "p1_correct": corr_p1,
            "p3_output": pred_p3, "p3_correct": corr_p3,
            "tok_count_p1": tok_count_p1, "tok_count_p3": tok_count_p3,
            "tok_guard_pass": tok_ok,
        }
    return sweep


def diagnostic_interpretation(summaries: dict) -> dict:
    acc1 = summaries["phase1_baseline"].accuracy()
    acc2 = summaries["phase2_anomaly"].accuracy()
    acc3 = summaries["phase3_control"].accuracy()

    interpretation = {
        "p1_accuracy": round(acc1, 4),
        "p2_accuracy": round(acc2, 4),
        "p3_accuracy": round(acc3, 4),
        "delta_p1_p2": round(acc1 - acc2, 4),
        "delta_p3_p1": round(acc3 - acc1, 4),
    }

    print(f"\nDiagnostic interpretation")
    if acc1 < 0.4:
        interpretation["p1_status"] = "baseline_failure"
        print("  P1: baseline failure -- model lacks positional counter for identical tokens")
    elif acc1 >= 0.8:
        interpretation["p1_status"] = "baseline_strong"
        print("  P1: baseline strong")
    else:
        interpretation["p1_status"] = "baseline_weak"
        print("  P1: baseline weak -- counter is noisy")

    delta_12 = acc1 - acc2
    if delta_12 > 0.3:
        interpretation["p2_status"] = "semantic_interference"
        print(f"  P2: semantic interference confirmed (dropped {delta_12:.0%})")
    elif delta_12 < 0:
        interpretation["p2_status"] = "intruder_helps"
        print(f"  P2: intruder helps (unexpected, +{-delta_12:.0%})")
    else:
        interpretation["p2_status"] = "little_effect"
        print("  P2: anomaly has little effect")

    delta_13 = acc3 - acc1
    if delta_13 > 0.2:
        interpretation["p3_status"] = "unique_improves"
        print("  P3: unique tokens improve counting")
    elif delta_13 < -0.2:
        interpretation["p3_status"] = "repetition_helps"
        print("  P3: repetition helps counting")
    else:
        interpretation["p3_status"] = "token_identity_not_primary"
        print("  P3: token identity is not the primary driver")

    return interpretation


def tokenization_diagnostics(tokenizer) -> dict:
    print("\nPayload tokenization diagnostic")
    results = {}
    for prompts_dict, label in [(PROMPTS, "original"), (PROMPTS_FIXED, "fixed")]:
        results[label] = {}
        for phase_key, entry in prompts_dict.items():
            payload = entry["text"].split(": ")[1].split(". Respond")[0].split(". Reply")[0]
            toks = tokenizer.encode(payload, add_special_tokens=False)
            word_count = len(payload.split()) if label == "original" else len([w.strip(",.") for w in payload.split(",")])

            match = word_count == len(toks)
            print(f"  [{label}/{phase_key}] words={word_count} tokens={len(toks)} {'ok' if match else 'MISMATCH'}")

            results[label][phase_key] = {
                "payload": payload,
                "word_count": word_count,
                "token_count": len(toks),
                "match": match,
                "tokens": [tokenizer.decode([t]) for t in toks],
            }
    return results


def run_fixed_prompts(pipe, tokenizer) -> dict:
    all_results_fixed = {}
    print("\nFixed (comma-separated) prompts")
    for phase_key, entry in PROMPTS_FIXED.items():
        phase_results = []
        for seed in SEEDS:
            r = run_single(pipe, tokenizer, phase_key, entry, seed, MAX_NEW_TOKENS, TEMPERATURE)
            phase_results.append(r)
        acc = sum(r["correct"] for r in phase_results) / N_RUNS
        print(f"  {phase_key}: {acc:.0%}")
        all_results_fixed[phase_key] = phase_results
    return all_results_fixed


def paraphrase_robustness(cfg, pipe, tokenizer, known_attractor: str) -> dict:
    test_cases = {
        "P1_repeated": (APPLE_LIST_10, 10),
        "P2_anomaly": (APPLE_LIST_10_ANOMALY, 9),
        "P3_unique": (UNIQUE_LIST_10, 10),
    }

    print(f"\nParaphrase robustness ({cfg.model_name})")
    paraphrase_results = {}
    for pname, template in PARAPHRASES.items():
        paraphrase_results[pname] = {}
        for case_name, (word_list, expected) in test_cases.items():
            if case_name == "P3_unique":
                prompt = (
                    "Count the number of words in this list: "
                    f"{word_list}. Respond only with the integer, nothing else."
                )
            else:
                prompt = template.format(list=word_list)

            torch.manual_seed(0)
            random.seed(0)
            np.random.seed(0)
            raw = generate(pipe, tokenizer, prompt, max_new_tokens=8)
            predicted = extract_count(raw)
            paraphrase_results[pname][case_name] = {
                "predicted": predicted, "expected": expected, "correct": predicted == expected,
            }
        p1_pred = paraphrase_results[pname]["P1_repeated"]["predicted"]
        stability = "stable" if str(p1_pred) == known_attractor else f"shifted to {p1_pred}"
        print(f"  {pname:<15}  P1={p1_pred}({stability})  P2={paraphrase_results[pname]['P2_anomaly']['predicted']}  P3={paraphrase_results[pname]['P3_unique']['predicted']}")

    return paraphrase_results


def language_robustness(pipe, tokenizer, known_attractor: str) -> dict:
    print("\nLanguage robustness")
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
            'Comptez le nombre de fois que "apple" apparait dans cette liste: '
            "apple apple apple apple apple apple apple apple apple apple. "
            "Repondez uniquement avec le nombre entier."
        ),
    }

    results = {}
    for lang, prompt in prompts_lang.items():
        raw = generate(pipe, tokenizer, prompt, max_new_tokens=8)
        predicted = extract_count(raw)
        stability = "stable" if str(predicted) == known_attractor else f"shifted to {predicted}"
        print(f"  {lang:<10}  output={predicted}  {'correct' if predicted == 10 else 'wrong'}  {stability}")
        results[lang] = {"predicted": predicted, "correct": predicted == 10, "stability": stability}
    return results


def repeated_symbols(pipe, tokenizer, known_attractor: str) -> dict:
    print("\nRepeated symbols")
    symbols = ["apple", "1", "0", "7", "cat", "the", "a", "X"]
    digit_fallbacks = ["2", "3", "4", "5", "6", "8", "9"]
    word_fallbacks = ["dog", "she", "an", "Y"]

    results = {}
    for symbol in symbols:
        actual_symbol = symbol
        substitution = None
        if len(tokenizer.encode(symbol, add_special_tokens=False)) != 1:
            fallback_pool = digit_fallbacks if symbol.isdigit() else word_fallbacks
            for fallback in fallback_pool:
                if fallback in symbols:
                    continue
                if len(tokenizer.encode(fallback, add_special_tokens=False)) == 1:
                    substitution = {"original": symbol, "substituted": fallback, "reason": "not single-token"}
                    actual_symbol = fallback
                    LOGGER.warning(f"Symbol {symbol!r} is not single-token, substituting {fallback!r}")
                    break
            if substitution is None:
                raise ValueError(f"No single-token fallback found for symbol {symbol!r}")

        prompt = make_prompt_repeated(10, word=actual_symbol)
        raw = generate(pipe, tokenizer, prompt, max_new_tokens=8)
        predicted = extract_count(raw)
        prior_fires = str(predicted) == known_attractor
        sub_note = f"  (substituted for {symbol!r})" if substitution else ""
        print(f"  {repr(actual_symbol):<10}  output={str(predicted):>4}  prior_fires={'yes' if prior_fires else 'no'}{sub_note}")
        results[symbol] = {
            "tested_symbol": actual_symbol, "predicted": predicted,
            "correct": predicted == 10, "prior_fires": prior_fires,
            "substitution": substitution,
        }
    return results


def chain_of_thought(pipe, tokenizer) -> dict:
    print("\nChain of thought")
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
        print(f"  {cot_name:<20}  final={final_num}  {'correct' if final_num == 10 else 'wrong'}")
        results[cot_name] = {"final_number": final_num, "correct": final_num == 10, "raw": raw}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    tokenizer, model, pipe = load_generation_model(cfg.model_name)

    # core behavioral phases
    all_results = run_all_phases(pipe, tokenizer)
    summaries = summarize(all_results)
    plot_accuracy(cfg, summaries)

    # sequence-length sweep with tokenization guards
    sweep = behavioral_n_sweep(cfg, pipe, tokenizer)

    # diagnostic interpretation
    interpretation = diagnostic_interpretation(summaries)

    # tokenization diagnostics
    tok_diag = tokenization_diagnostics(tokenizer)

    # fixed prompts comparison
    fixed_results = run_fixed_prompts(pipe, tokenizer)
    fixed_comparison = {}
    for phase_key in PROMPTS:
        orig_acc = sum(r["correct"] for r in all_results[phase_key]) / N_RUNS
        fixed_acc = sum(r["correct"] for r in fixed_results[phase_key]) / N_RUNS
        fixed_comparison[phase_key] = {
            "original_accuracy": round(orig_acc, 4),
            "fixed_accuracy": round(fixed_acc, 4),
            "delta": round(fixed_acc - orig_acc, 4),
        }

    # determine attractor
    known_attractor = str(all_results["phase1_baseline"][0]["predicted"])

    # paraphrase robustness
    paraphrase = paraphrase_robustness(cfg, pipe, tokenizer, known_attractor)

    # language robustness
    lang_results = language_robustness(pipe, tokenizer, known_attractor)

    # repeated symbols
    symbol_results = repeated_symbols(pipe, tokenizer, known_attractor)

    # chain of thought
    cot_results = chain_of_thought(pipe, tokenizer)

    # save everything
    output = {
        "model": cfg.model_name,
        "phases": all_results,
        "summaries": {k: {"accuracy": round(s.accuracy(), 4), "distribution": s.dist()} for k, s in summaries.items()},
        "n_sweep": {str(k): v for k, v in sweep.items()},
        "interpretation": interpretation,
        "tokenization": tok_diag,
        "fixed_prompts": fixed_results,
        "fixed_comparison": fixed_comparison,
        "attractor": known_attractor,
        "paraphrase_robustness": paraphrase,
        "language_robustness": lang_results,
        "repeated_symbols": symbol_results,
        "chain_of_thought": cot_results,
    }

    save_path = output_path(cfg, "behavioral.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

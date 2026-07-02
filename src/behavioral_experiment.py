"""
Odometer Pilot — core behavioral experiment.

Runs the three counting phases (identical tokens / anomaly / unique tokens)
across N_RUNS seeds, reports accuracy + prediction distributions, plots the
results, sweeps n=5..20 to characterize the wrong-answer attractor, runs the
tokenization diagnostics (raw vs comma-separated prompts), and saves
everything to outputs/<model>/.

Usage:
    python behavioral_experiment.py --model 1b
    python behavioral_experiment.py --model 3b
"""

import argparse
import json
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

import _paths  # noqa: F401  (adds config/ and data/ to sys.path)
from config import N_RUNS, SEEDS, TEMPERATURE, MAX_NEW_TOKENS, add_model_arg, get_config, output_path
from prompts import PROMPTS, PROMPTS_FIXED, UNIQUE_VOCAB, make_prompt_repeated, make_prompt_unique
from utils import PhaseSummary, extract_count, generate, load_generation_model, run_single


def run_all_phases(pipe, tokenizer) -> dict:
    all_results = {}
    for phase_key, entry in PROMPTS.items():
        print(f"\n{chr(9472) * 55}")
        print(f"  {phase_key.upper()}  |  {entry['description']}")
        print(f"{chr(9472) * 55}")

        phase_results = []
        for seed in SEEDS:
            r = run_single(pipe, tokenizer, phase_key, entry, seed, MAX_NEW_TOKENS, TEMPERATURE)
            status = "✓" if r["correct"] else "✗"
            print(f"  seed={seed:02d}  predicted={str(r['predicted']):>4}  "
                  f"expected={r['expected']}  {status}  raw={repr(r['raw'][:40])}")
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

    print(f"{'Phase':<22} {'Accuracy':>10}  {'Distribution of predictions'}")
    print("─" * 65)
    for k, s in summaries.items():
        print(f"{k:<22} {s.accuracy():>9.0%}  {s.dist()}")
    return summaries


def plot_accuracy(cfg, summaries: dict):
    labels = list(summaries.keys())
    accs = [summaries[k].accuracy() * 100 for k in labels]
    colors = ["#4c9be8", "#e8844c", "#4ce87a"]
    expected = [summaries[k].expected for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Odometer Pilot — {cfg.model_name}", fontsize=13, fontweight="bold")

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
    save_path = output_path(cfg, "odometer_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def behavioral_n_sweep(cfg, pipe, tokenizer):
    ns = [5, 6, 7, 8, 9, 10, 11, 12, 15, 20]
    print("=" * 65)
    print(f"BEHAVIORAL N-SWEEP — {cfg.key.upper()}")
    print("=" * 65)
    print(f"  {'n':>4}  {'P1 output':>10}  {'P1 correct':>11}  {'P3 output':>10}  {'P3 correct':>11}")
    print("  " + "-" * 55)

    sweep = {}
    for n in ns:
        raw_p1 = generate(pipe, tokenizer, make_prompt_repeated(n), max_new_tokens=8)
        raw_p3 = generate(pipe, tokenizer, make_prompt_unique(n), max_new_tokens=8)
        pred_p1 = extract_count(raw_p1)
        pred_p3 = extract_count(raw_p3)
        corr_p1 = "✓" if pred_p1 == n else "✗"
        corr_p3 = "✓" if pred_p3 == n else "✗"
        print(f"  {n:>4}  {str(pred_p1):>10}  {corr_p1:>11}  {str(pred_p3):>10}  {corr_p3:>11}")
        sweep[n] = {
            "n": n,
            "p1_output": pred_p1, "p1_correct": pred_p1 == n,
            "p3_output": pred_p3, "p3_correct": pred_p3 == n,
        }

    print("\nP1 attractor pattern:")
    for n, r in sweep.items():
        marker = "= n (correct)" if r["p1_correct"] else f"≠ n (wrong, got {r['p1_output']})"
        print(f"  n={n:>2}  output={r['p1_output']}  {marker}")

    save_path = output_path(cfg, "behavioral_n_sweep.json")
    with open(save_path, "w") as f:
        json.dump({str(k): v for k, v in sweep.items()}, f, indent=2)
    print(f"\nSaved: {save_path}")


def diagnostic_interpretation(summaries: dict):
    acc1 = summaries["phase1_baseline"].accuracy()
    acc2 = summaries["phase2_anomaly"].accuracy()
    acc3 = summaries["phase3_control"].accuracy()

    print("DIAGNOSTIC INTERPRETATION")

    if acc1 < 0.4:
        print("\n[Phase 1 — BASELINE FAILURE]")
        print("  The model lacks even a basic positional counter for identical tokens.")
    elif acc1 >= 0.8:
        print("\n[Phase 1 — BASELINE STRONG]")
        print("  Model counts identical tokens reliably.")
    else:
        print("\n[Phase 1 — BASELINE WEAK]")
        print("  Partial success — the counter is noisy even without interference.")

    delta_12 = acc1 - acc2
    if delta_12 > 0.3:
        print("\n[Phase 2 — SEMANTIC INTERFERENCE CONFIRMED]")
        print(f"  Accuracy dropped {delta_12:.0%} when a single intruder token was inserted.")
        print("  Consistent with attention-pattern disruption rather than a content-free odometer.")
    elif delta_12 < 0:
        print("\n[Phase 2 — INTRUDER HELPS? (unexpected)]")
        print("  Accuracy improved with the anomaly. Possible novelty-driven attention effect.")
    else:
        print("\n[Phase 2 — ANOMALY HAS LITTLE EFFECT]")
        print("  Intruder token does not significantly disrupt counting.")

    delta_13 = acc3 - acc1
    if delta_13 > 0.2:
        print("\n[Phase 3 — UNIQUE TOKENS IMPROVE COUNTING]")
        print("  Distinct tokens provide better positional anchoring than repetitions.")
        print("  Suggests the failure in P1/P2 is tied to attention-sink dynamics on repeated tokens.")
    elif delta_13 < -0.2:
        print("\n[Phase 3 — REPETITION HELPS COUNTING]")
        print("  Model is actually better at counting repeated tokens — pattern-matching hypothesis.")
    else:
        print("\n[Phase 3 — TOKEN IDENTITY IS NOT THE PRIMARY DRIVER]")
        print("  Similar accuracy with unique vs repeated tokens.")

    print()
    print(f"  P1 Baseline : {acc1:.0%}")
    print(f"  P2 Anomaly  : {acc2:.0%}  (Δ vs P1: {acc2 - acc1:+.0%})")
    print(f"  P3 Control  : {acc3:.0%}  (Δ vs P1: {acc3 - acc1:+.0%})")


def tokenization_diagnostics(tokenizer):
    print("PAYLOAD TOKENIZATION DIAGNOSTIC")
    for phase_key, entry in PROMPTS.items():
        payload = entry["text"].split(": ")[1].split(". Respond")[0]
        toks = tokenizer.encode(payload, add_special_tokens=False)
        decoded = [tokenizer.decode([t]) for t in toks]

        print(f"\n[{phase_key}]")
        print(f"  Payload        : {payload}")
        print(f"  Word count     : {len(payload.split())}")
        print(f"  Token count    : {len(toks)}")
        print(f"  Expected answer: {entry['expected']}")
        print(f"  Tokens         : {decoded}")
        print(f"  Token IDs      : {toks}")
        print(f"  *** MISMATCH: token count ({len(toks)}) != word count ({len(payload.split())}) ***"
              if len(toks) != len(payload.split()) else
              f"  ✓ token count == word count ({len(toks)})")

    print("\n" + "=" * 70)
    print("FIXED PROMPT TOKENIZATION CHECK")
    print("=" * 70)
    for phase_key, entry in PROMPTS_FIXED.items():
        payload = entry["text"].split(": ")[1].split(". Respond")[0]
        toks = tokenizer.encode(payload, add_special_tokens=False)
        decoded = [tokenizer.decode([t]) for t in toks]
        word_count = len([w.strip(",.") for w in payload.split(",")])

        print(f"\n[{phase_key}]")
        print(f"  Payload      : {payload}")
        print(f"  Word count   : {word_count}")
        print(f"  Token count  : {len(toks)}")
        print(f"  Tokens       : {decoded}")
        print(f"  ✓ Comma separators verified" if len(toks) >= word_count else
              f"  ✗ Still mismatched — check tokens above")


def run_fixed_prompts(pipe, tokenizer, all_results: dict) -> dict:
    all_results_fixed = {}
    for phase_key, entry in PROMPTS_FIXED.items():
        print(f"\n{chr(9472) * 55}")
        print(f"  {phase_key.upper()} [FIXED]  |  {entry['description']}")
        print(f"{chr(9472) * 55}")

        phase_results = []
        for seed in SEEDS:
            r = run_single(pipe, tokenizer, phase_key, entry, seed, MAX_NEW_TOKENS, TEMPERATURE)
            status = "✓" if r["correct"] else "✗"
            print(f"  seed={seed:02d}  predicted={str(r['predicted']):>4}  "
                  f"expected={entry['expected']}  {status}  raw={repr(r['raw'][:40])}")
            phase_results.append(r)
        all_results_fixed[phase_key] = phase_results

    print("\n" + "=" * 70)
    print("ACCURACY COMPARISON: ORIGINAL vs FIXED PROMPTS")
    print("=" * 70)
    print(f"{'Phase':<25} {'Original':>10} {'Fixed':>10}  {'Δ':>6}")
    print("-" * 55)
    for phase_key in PROMPTS:
        orig_acc = sum(r["correct"] for r in all_results[phase_key]) / N_RUNS
        fixed_acc = sum(r["correct"] for r in all_results_fixed[phase_key]) / N_RUNS
        delta = fixed_acc - orig_acc
        print(f"  {phase_key:<23} {orig_acc:>9.0%} {fixed_acc:>9.0%}  {delta:>+6.0%}")

    return all_results_fixed


def outlier_investigation(pipe, tokenizer, all_results: dict):
    print("=" * 70)
    print("PHASE 3 OUTLIER INVESTIGATION — full generation (max_new_tokens=64)")
    print("=" * 70)
    print("Logging all seeds; pay attention to outlier predictions\n")

    outlier_seeds = [r["seed"] for r in all_results["phase3_control"] if not r["correct"]]
    for seed in SEEDS:
        torch.manual_seed(seed)
        full_raw = generate(pipe, tokenizer, PROMPTS["phase3_control"]["text"], max_new_tokens=64)
        marker = "  *** OUTLIER ***" if seed in outlier_seeds else ""
        print(f"seed={seed:02d}{marker}")
        print(f"  {repr(full_raw)}\n")


def paraphrase_robustness(cfg, pipe, tokenizer, all_results: dict):
    from prompts import PARAPHRASES, APPLE_LIST_10, APPLE_LIST_10_ANOMALY, UNIQUE_LIST_10

    test_cases = {
        "P1_repeated": (APPLE_LIST_10, 10),
        "P2_anomaly": (APPLE_LIST_10_ANOMALY, 9),
        "P3_unique": (UNIQUE_LIST_10, 10),
    }

    print("=" * 75)
    print(f"PROMPT PARAPHRASE ROBUSTNESS — {cfg.model_name}")
    print("=" * 75)
    print(f"  {'Paraphrase':<15}", end="")
    for case in test_cases:
        print(f"  {case:<16}", end="")
    print()
    print("  " + "-" * 68)

    paraphrase_results = {}
    for pname, template in PARAPHRASES.items():
        print(f"  {pname:<15}", end="")
        paraphrase_results[pname] = {}

        for case_name, (word_list, expected) in test_cases.items():
            if case_name == "P3_unique":
                prompt = (
                    "Count the number of words in this list: "
                    f"{word_list}. Respond only with the integer, nothing else."
                )
            else:
                prompt = template.format(list=word_list)

            # Seed before every inference call so that the marginal cases
            # (e.g. how_many/P2 where logits for 8 and 9 are nearly tied)
            # are stable across runs.  The original notebook did not seed
            # here, which caused how_many/P2 to flip between 8 and 9
            # across runs; with do_sample=False the result should be
            # deterministic given the same GPU state, so seeding makes that
            # GPU-state dependency explicit and removes the flip.
            torch.manual_seed(0)
            random.seed(0)
            np.random.seed(0)
            raw = generate(pipe, tokenizer, prompt, max_new_tokens=8)
            predicted = extract_count(raw)
            correct = "✓" if predicted == expected else "✗"
            cell = f"{predicted}({correct})"
            print(f"  {cell:<16}", end="")
            paraphrase_results[pname][case_name] = {
                "predicted": predicted, "expected": expected, "correct": predicted == expected,
            }
        print()

    known_attractor = str(all_results["phase1_baseline"][0]["predicted"])
    print(f"\nP1 attractor stability (known attractor from behavioral = '{known_attractor}'):")
    for pname, results in paraphrase_results.items():
        p1_pred = results["P1_repeated"]["predicted"]
        tag = "STABLE" if str(p1_pred) == known_attractor else f"SHIFTED to {p1_pred}"
        print(f"  {pname:<15}  predicted={p1_pred}  {tag}")

    save_path = output_path(cfg, "paraphrase_robustness.json")
    with open(save_path, "w") as f:
        json.dump(paraphrase_results, f, indent=2)
    print(f"\nSaved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    tokenizer, model, pipe = load_generation_model(cfg.model_name)

    all_results = run_all_phases(pipe, tokenizer)
    summaries = summarize(all_results)
    plot_accuracy(cfg, summaries)
    behavioral_n_sweep(cfg, pipe, tokenizer)
    diagnostic_interpretation(summaries)
    tokenization_diagnostics(tokenizer)
    all_results_fixed = run_fixed_prompts(pipe, tokenizer, all_results)
    outlier_investigation(pipe, tokenizer, all_results)

    save_path = output_path(cfg, "odometer_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {save_path}")

    paraphrase_robustness(cfg, pipe, tokenizer, all_results)


if __name__ == "__main__":
    main()

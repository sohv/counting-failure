"""
Tests whether interleaved structural noise (newline/pipe delimiters, with and
without a matching "-separated list" instruction hint) suppresses the
repeated-token counting prior on phase1_baseline, compared against the
existing space and comma baselines. Scoped to the three prior-active models
(llama-1b, llama-3b, qwen-1.5b) - on qwen-3b/7b the prior is benign, so this
tests nothing there.

Usage:
    uv run -m src.experiments.interleaved_noise --model llama-1b
"""

import argparse
import json
import logging

import matplotlib.pyplot as plt

from src.common.config import MAX_NEW_TOKENS, N_RUNS, SEEDS, TEMPERATURE, add_model_arg, get_config, load_results, output_path
from src.common.prompts import INTERLEAVED_VARIANTS, PROMPTS
from src.common.utils import PhaseSummary, load_generation_model, run_single

LOGGER = logging.getLogger(__name__)

VARIANTS = INTERLEAVED_VARIANTS


def check_tokenization(tokenizer) -> dict:
    print("\nTokenization check (interleaved-noise variants)")
    results = {}
    for name, entry in VARIANTS.items():
        payload = entry["text"].split(": ")[1].split(". Respond")[0]
        toks = tokenizer.encode(payload, add_special_tokens=False)
        n_apple = sum(1 for t in toks if "apple" in tokenizer.decode([t]).lower())
        ok = n_apple == 10
        print(f"  {name:18s} total_tokens={len(toks):3d}  apple_tokens={n_apple}  {'ok' if ok else 'MISMATCH'}")
        results[name] = {"total_tokens": len(toks), "apple_tokens": n_apple, "ok": ok}
    return results


def run_variants(pipe, tokenizer) -> dict:
    print("\nBehavioral runs per variant")
    all_results = {}
    for name, entry in VARIANTS.items():
        phase_results = []
        for seed in SEEDS:
            r = run_single(pipe, tokenizer, name, entry, seed, MAX_NEW_TOKENS, TEMPERATURE)
            phase_results.append(r)
        s = PhaseSummary(phase=name, description=entry["description"], expected=entry["expected"], n_runs=N_RUNS)
        for r in phase_results:
            s.add(r)
        print(f"  {name:18s} accuracy={s.accuracy():.0%}  distribution={s.dist()}")
        all_results[name] = {"results": phase_results, "accuracy": round(s.accuracy(), 4), "distribution": s.dist()}
    return all_results


def plot_accuracy(cfg, all_results: dict):
    labels = list(all_results.keys())
    accs = [all_results[k]["accuracy"] * 100 for k in labels]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle(f"Interleaved-noise counting accuracy -- {cfg.model_name}", fontsize=12, fontweight="bold")
    bars = ax.bar(labels, accs, color="#4c9be8", edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{acc:.0f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_path = output_path(cfg, "interleaved_noise.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    behavioral = load_results(cfg, "behavioral")
    p1_attractor = behavioral["attractor"]
    model_fails_p1 = p1_attractor != str(PROMPTS["phase1_baseline"]["expected"])
    print(f"Model: {cfg.model_name}  P1 attractor: {p1_attractor}  fails_p1: {model_fails_p1}")
    if not model_fails_p1:
        print("Note: model solves the baseline P1 prompt - interleaved noise tests a benign prior here, not a failure mode.")

    tokenizer, model, pipe = load_generation_model(cfg.model_name)

    tok_check = check_tokenization(tokenizer)
    all_results = run_variants(pipe, tokenizer)
    plot_accuracy(cfg, all_results)

    baseline_acc = all_results["space_baseline"]["accuracy"]
    comparison = {
        name: {
            "accuracy": v["accuracy"],
            "delta_vs_baseline": round(v["accuracy"] - baseline_acc, 4),
            "suppressed_prior": v["accuracy"] > baseline_acc,
        }
        for name, v in all_results.items()
    }

    output = {
        "model": cfg.model_name,
        "p1_attractor": p1_attractor,
        "model_fails_p1": model_fails_p1,
        "tokenization_check": tok_check,
        "variants": {name: v["results"] for name, v in all_results.items()},
        "comparison": comparison,
    }

    save_path = output_path(cfg, "interleaved_noise.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")
    print(f"Plot with: uv run -m src.experiments.interleaved_noise --model {cfg.key} (regenerates plot alongside data)")


if __name__ == "__main__":
    main()

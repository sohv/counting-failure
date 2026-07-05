"""
Re-runs logit lens and decomposition in float32 (in addition to default
bfloat16) and across three seeds. Reports whether the identified writer layer
and depth band are stable across numerical environments.

Usage:
    uv run -m src.experiments.llama.robustness --model llama-1b
"""

import argparse
import json
import logging

import torch

from src.common.config import add_model_arg, get_config, load_results, output_path
from src.common.prompts import PROMPTS, make_prompt_repeated
from src.common.utils import (
    decompose_layer,
    get_writer_logit_diff,
    load_eager_model,
    logit_lens_single,
    make_inputs_eager,
    remove_all_hooks,
)

LOGGER = logging.getLogger(__name__)

ROBUSTNESS_SEEDS = [42, 123, 456]


def run_logit_lens_sweep(
    tokenizer, model_eager, prompt: str, n_layers: int,
    correct: int, wrong: int,
) -> list[dict]:
    """Logit lens at every layer, returns per-layer logit_diff."""
    inputs = make_inputs_eager(tokenizer, model_eager, prompt)
    with torch.no_grad():
        out = model_eager(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )

    results = []
    for layer_idx, h in enumerate(out.hidden_states):
        r = logit_lens_single(
            tokenizer, model_eager, h[0, -1, :],
            correct_answer=correct, wrong_answer=wrong,
        )
        r["layer"] = layer_idx
        results.append(r)
    return results


def run_decomposition_sweep(
    tokenizer, model_eager, prompt: str, n_layers: int,
    correct: int, wrong: int,
) -> dict:
    """Full decomposition at every layer, returns per-layer MLP contribution."""
    sweep = {}
    for layer_idx in range(1, n_layers + 1):
        remove_all_hooks(model_eager)
        r = decompose_layer(
            tokenizer, model_eager, prompt, layer_idx,
            correct_answer=correct, wrong_answer=wrong,
        )

        diff_before = r.get("h_before", {}).get("logit_diff", float("nan"))
        diff_post_attn = r.get("h_post_attn", {}).get("logit_diff", float("nan"))
        diff_post_mlp = r.get("h_post_layer", {}).get("logit_diff", float("nan"))
        writer_info = get_writer_logit_diff(diff_before, diff_post_attn, diff_post_mlp)

        sweep[layer_idx] = {
            "diff_before": round(diff_before, 4),
            "diff_post_attn": round(diff_post_attn, 4),
            "diff_post_mlp": round(diff_post_mlp, 4),
            "writer": writer_info,
            "top_digit_post_mlp": r.get("h_post_layer", {}).get("top_digit", "?"),
        }
    return sweep


def find_writer_from_sweep(sweep: dict, wrong: int | str) -> int | None:
    """Earliest layer where the wrong digit becomes and stays the top post-MLP
    digit through the last layer. Matches find_writer_layer in logit_lens.py —
    picking the single largest MLP contribution instead lands on early, volatile
    layers with no persistent connection to the eventual answer."""
    wrong_str = str(wrong)
    layers = sorted(sweep.keys())
    for layer_idx in layers:
        if all(sweep[l]["top_digit_post_mlp"] == wrong_str for l in layers if l >= layer_idx):
            return layer_idx
    return None


def run_single_condition(
    model_name: str, dtype: torch.dtype, seed: int,
    prompt: str, n_layers: int, correct: int, wrong: int,
) -> dict:
    """Run logit lens + decomposition under one precision/seed condition."""
    torch.manual_seed(seed)

    tokenizer, model_eager = load_eager_model(model_name, dtype=dtype)

    lens = run_logit_lens_sweep(tokenizer, model_eager, prompt, n_layers, correct, wrong)
    decomp = run_decomposition_sweep(tokenizer, model_eager, prompt, n_layers, correct, wrong)
    writer = find_writer_from_sweep(decomp, wrong)

    # extract logit_diff trajectory
    trajectory = [r.get("logit_diff", float("nan")) for r in lens]

    del model_eager
    torch.cuda.empty_cache()

    return {
        "dtype": str(dtype),
        "seed": seed,
        "writer_layer": writer,
        "logit_diff_trajectory": [round(v, 4) if not (isinstance(v, float) and v != v) else None for v in trajectory],
        "decomposition": {str(k): v for k, v in decomp.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    # load dependencies
    behavioral = load_results(cfg, "behavioral")
    logit_lens_data = load_results(cfg, "logit_lens")

    attractor = behavioral["attractor"]
    wrong = int(attractor) if attractor not in ("None", "?") else 10
    correct = PROMPTS["phase1_baseline"]["expected"]
    prompt = make_prompt_repeated(10)

    baseline_writer = logit_lens_data.get("summary", {}).get("writer_layer")
    print(f"Baseline writer layer from logit_lens: L{baseline_writer}")

    # run conditions
    conditions = []
    for dtype in [torch.bfloat16, torch.float32]:
        for seed in ROBUSTNESS_SEEDS:
            label = f"{dtype}_seed{seed}"
            print(f"\nRunning condition: {label}")
            result = run_single_condition(
                cfg.model_name, dtype, seed, prompt, cfg.n_layers, correct, wrong,
            )
            result["label"] = label
            conditions.append(result)
            print(f"  Writer layer: L{result['writer_layer']}")

    # stability analysis
    writer_layers = [c["writer_layer"] for c in conditions]
    bf16_writers = [c["writer_layer"] for c in conditions if "bfloat16" in c["dtype"]]
    fp32_writers = [c["writer_layer"] for c in conditions if "float32" in c["dtype"]]

    is_stable = len(set(writer_layers)) == 1
    bf16_stable = len(set(bf16_writers)) == 1
    fp32_stable = len(set(fp32_writers)) == 1
    cross_precision_stable = bf16_stable and fp32_stable and set(bf16_writers) == set(fp32_writers)

    stability = {
        "all_writer_layers": writer_layers,
        "is_stable": is_stable,
        "bf16_stable": bf16_stable,
        "fp32_stable": fp32_stable,
        "cross_precision_stable": cross_precision_stable,
        "baseline_writer": baseline_writer,
    }

    print(f"\nStability analysis")
    print(f"  Writer layers across conditions: {writer_layers}")
    print(f"  All stable: {is_stable}")
    print(f"  Cross-precision stable: {cross_precision_stable}")

    output = {
        "model": cfg.model_name,
        "correct": correct,
        "wrong": wrong,
        "baseline_writer": baseline_writer,
        "conditions": conditions,
        "stability": stability,
    }

    save_path = output_path(cfg, "robustness.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

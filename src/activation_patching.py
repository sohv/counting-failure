"""
Odometer Pilot — Activation Patching.

Takes the last-token residual stream from a prompt where the model answers
correctly (unique tokens) and injects it into the forward pass of a prompt
where the model answers wrongly (repeated tokens), at each layer in turn.
The earliest layer where this flips the output to correct is the causal
site for the attractor.

Usage:
    python activation_patching.py --model 1b
    python activation_patching.py --model 3b
"""

import argparse
import json

import torch

import src._paths as _paths  # noqa: F401  (adds config/ and data/ to sys.path)
from config import add_model_arg, get_config, output_path
from prompts import PROMPTS, make_prompt_numbered, make_prompt_repeated, make_prompt_unique
from src.utils import get_top_digit, load_eager_model, make_inputs_eager, remove_all_hooks


def patch_layer_sweep(tokenizer, model_eager, source_inputs, target_inputs, correct_str: str, n_layers: int):
    sweep = []
    for patch_layer in range(1, n_layers + 1):
        remove_all_hooks(model_eager)
        cache = {}

        def save_h(module, input, output, _l=patch_layer):
            cache[_l] = output[0, -1, :].detach().clone()

        def patch_h(module, input, output, _l=patch_layer):
            patched = output.clone()
            patched[0, -1, :] = cache[_l]
            return patched

        h1 = model_eager.model.layers[patch_layer - 1].register_forward_hook(save_h)
        with torch.no_grad():
            model_eager(**source_inputs)
        h1.remove()

        h2 = model_eager.model.layers[patch_layer - 1].register_forward_hook(patch_h)
        with torch.no_grad():
            pout = model_eager(**target_inputs)
        h2.remove()

        pl = pout.logits[0, -1, :]
        td = get_top_digit(tokenizer, pl)
        top5 = [tokenizer.decode([i]) for i in pl.topk(5).indices]
        flipped = td == correct_str

        sweep.append({"layer": patch_layer, "top_digit": td, "flipped_to_correct": flipped, "top5_tokens": top5})
    return sweep


def main_patching_run(cfg, tokenizer, model_eager):
    source_inputs = make_inputs_eager(tokenizer, model_eager, PROMPTS["phase3_control"]["text"])
    target_inputs = make_inputs_eager(tokenizer, model_eager, PROMPTS["phase1_baseline"]["text"])

    remove_all_hooks(model_eager)
    with torch.no_grad():
        base_out = model_eager(**target_inputs)
    base_logits = base_out.logits[0, -1, :]
    print(f"Unpatched P1 top digit : {get_top_digit(tokenizer, base_logits)}")
    print(f"Top-5 tokens           : {[tokenizer.decode([i]) for i in base_logits.topk(5).indices]}")

    print("\n" + "=" * 65)
    print("LAYER SWEEP — patching last-token residual P3 → P1 at each layer")
    print("=" * 65)
    print(f"  {'Layer':>6}  {'Top digit':>10}  {'Flipped?':>9}  Top-5 tokens")
    print("  " + "-" * 58)

    sweep_results = patch_layer_sweep(tokenizer, model_eager, source_inputs, target_inputs, "10", cfg.n_layers)
    for r in sweep_results:
        flipped = "✓ YES" if r["flipped_to_correct"] else "✗ no"
        print(f"  L{r['layer']:02d}    {r['top_digit']:>10}  {flipped:>9}  {r['top5_tokens']}")

    flip_layers = [r["layer"] for r in sweep_results if r["flipped_to_correct"]]
    print(f"\nLayers where patching flips output to correct (10): {flip_layers}")
    print(f"Earliest causal site: L{min(flip_layers)}" if flip_layers else "No layer flip found.")

    save_path = output_path(cfg, "activation_patching_results.json")
    with open(save_path, "w") as f:
        json.dump({
            "source_phase": "phase3_control",
            "target_phase": "phase1_baseline",
            "correct_answer": 10,
            "unpatched_output": get_top_digit(tokenizer, base_logits),
            "sweep": sweep_results,
        }, f, indent=2)
    print(f"Saved: {save_path}")

    return {10: {
        "n": 10, "correct": 10, "unpatched_digit": get_top_digit(tokenizer, base_logits),
        "flip_layers": flip_layers, "earliest_causal": min(flip_layers) if flip_layers else None,
    }}


def patching_sweep_across_n(cfg, tokenizer, model_eager, all_sweep_results: dict):
    ns = [6, 7, 8, 9, 11, 12]  # n=10 already covered above

    for n in ns:
        remove_all_hooks(model_eager)
        source_inputs = make_inputs_eager(tokenizer, model_eager, make_prompt_unique(n))
        target_inputs = make_inputs_eager(tokenizer, model_eager, make_prompt_repeated(n))
        correct_str = str(n)

        with torch.no_grad():
            base_out = model_eager(**target_inputs)
        base_digit = get_top_digit(tokenizer, base_out.logits[0, -1, :])

        sweep = patch_layer_sweep(tokenizer, model_eager, source_inputs, target_inputs, correct_str, cfg.n_layers)
        flip_layers = [r["layer"] for r in sweep if r["flipped_to_correct"]]
        earliest = min(flip_layers) if flip_layers else None

        all_sweep_results[n] = {
            "n": n, "correct": n, "unpatched_digit": base_digit,
            "sweep": sweep, "flip_layers": flip_layers, "earliest_causal": earliest,
        }
        print(f"n={n:2d} | unpatched={base_digit} | flip_layers={flip_layers} | earliest=L{earliest}")

    print("\n" + "=" * 55)
    print("CAUSAL SITE STABILITY ACROSS N")
    print("=" * 55)
    print(f"  {'n':>4}  {'Unpatched':>10}  {'Earliest causal L':>18}  Flip layers")
    print("  " + "-" * 50)
    for n in sorted(all_sweep_results):
        r = all_sweep_results[n]
        print(f"  {n:>4}  {r['unpatched_digit']:>10}  {str(r['earliest_causal']):>18}  {r['flip_layers']}")

    save_path = output_path(cfg, "patching_sweep_all_n.json")
    with open(save_path, "w") as f:
        json.dump({str(k): v for k, v in all_sweep_results.items()}, f, indent=2)
    print(f"\nSaved: {save_path}")


def numbered_format_check(tokenizer, model_eager):
    print("Numbered format outputs:")
    for n in [8, 9, 10, 11, 12]:
        remove_all_hooks(model_eager)
        inputs = make_inputs_eager(tokenizer, model_eager, make_prompt_numbered(n))
        with torch.no_grad():
            out = model_eager(**inputs)
        td = get_top_digit(tokenizer, out.logits[0, -1, :])
        top5 = [tokenizer.decode([i]) for i in out.logits[0, -1, :].topk(5).indices]
        print(f"  n={n}  top digit={td}  top5={top5}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    tokenizer, model_eager = load_eager_model(cfg.model_name)

    all_sweep_results = main_patching_run(cfg, tokenizer, model_eager)
    patching_sweep_across_n(cfg, tokenizer, model_eager, all_sweep_results)
    numbered_format_check(tokenizer, model_eager)


if __name__ == "__main__":
    main()

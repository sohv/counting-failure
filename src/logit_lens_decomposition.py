"""
Odometer Pilot — Logit Lens & MLP/Attention Decomposition.

Where does the wrong-answer "attractor" digit first appear in the residual
stream, and which sublayer (attention vs MLP) writes it? This script:

  1. Runs the logit lens across every layer for P1/P2/P3.
  2. Decomposes attention-vs-MLP contributions at the model's critical
     layers (config.critical_layers).
  3. Sweeps n=7..15 at each "gap" layer of interest (config.gap_layers) to
     see whether that layer's write/erase behavior is threshold-dependent.
  4. Does a full 1..n_layers sweep for n=10, tracking exactly where the
     attractor is written and (if applicable) erased.
  5. Zero-ablates the MLP at the lock-in layer to test causal responsibility.
  6. Saves a small mechanistic summary.
  7. Re-runs the MLP decomposition for paraphrased prompts to see whether
     the same lock-in layer fires regardless of surface form.

Usage:
    python logit_lens_decomposition.py --model 1b
    python logit_lens_decomposition.py --model 3b
"""

import argparse
import json

import torch

import _paths  # noqa: F401  (adds config/ and data/ to sys.path)
from config import add_model_arg, get_config, output_path
from prompts import PARAPHRASES, PROMPTS, make_prompt_repeated
from utils import (
    decompose_layer,
    get_top_digit,
    get_top_digit_and_margin,
    get_writer,
    load_eager_model,
    load_generation_model,
    logit_lens_single,
    make_inputs_eager,
    remove_all_hooks,
    run_single,
)


def get_attractors_via_pipe(pipe, tokenizer) -> dict:
    attractors = {}
    for phase_key, entry in PROMPTS.items():
        r = run_single(pipe, tokenizer, phase_key, entry, seed=0, max_new_tokens=16, temperature=0.0)
        attractors[phase_key] = str(r["predicted"])
    return attractors


def logit_lens_all_layers(tokenizer, model_eager, prompt_text: str, top_k: int = 5):
    inputs = make_inputs_eager(tokenizer, model_eager, prompt_text)
    with torch.no_grad():
        out = model_eager(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )

    results = []
    for layer_idx, h in enumerate(out.hidden_states):
        top_digit, top5, margin = logit_lens_single(tokenizer, model_eager, h[0, -1, :])
        results.append({
            "layer": "embed" if layer_idx == 0 else f"L{layer_idx:02d}",
            "top_digit": top_digit,
            "top5": top5,
            "margin": margin,
        })
    return results


def run_logit_lens(cfg, tokenizer, model_eager, attractors: dict):
    print("=" * 65)
    print("LOGIT LENS — where does the attractor take over from the correct count?")
    print("=" * 65)

    lens_all = {}
    for phase_key in ["phase1_baseline", "phase2_anomaly", "phase3_control"]:
        correct = PROMPTS[phase_key]["expected"]
        print(f"\n[{phase_key}]  correct={correct}  model outputs={attractors[phase_key]}")
        print(f"  {'Layer':>6}  {'Top digit':>10}  {'Margin':>8}  Top-5 tokens")
        print("  " + "-" * 55)

        remove_all_hooks(model_eager)
        lens = logit_lens_all_layers(tokenizer, model_eager, PROMPTS[phase_key]["text"])
        lens_all[phase_key] = lens
        for r in lens:
            margin_str = f"{r['margin']:.3f}" if r["margin"] is not None else "?"
            close_call = " (close call)" if r["margin"] is not None and r["margin"] < 0.5 else ""
            print(f"  {r['layer']:>6}  {r['top_digit']:>10}  {margin_str:>8}  {r['top5']}{close_call}")

    save_path = output_path(cfg, "logit_lens_all_phases.json")
    with open(save_path, "w") as f:
        json.dump(lens_all, f, indent=2)
    print(f"\nSaved: {save_path}")


def mlp_attn_decomposition(cfg, tokenizer, model_eager, attractors: dict) -> dict:
    print(f"\n{'=' * 72}")
    print(f"MLP vs ATTENTION DECOMPOSITION — {cfg.model_name}")
    print(f"Layers: {cfg.critical_layers} / {cfg.n_layers} total")
    print("=" * 72)

    decomp_results = {}
    for phase_key in ["phase1_baseline", "phase3_control"]:
        phase_attractor = attractors[phase_key]
        print(f"\n[{phase_key}]  tracking: '{phase_attractor}'")
        print(f"  {'Layer':>6}  {'Before':>10}  {'Post-attn':>10}  {'Post-MLP':>10}  {'Margin':>7}  Writer")
        print("  " + "-" * 58)

        for layer_idx in cfg.critical_layers:
            remove_all_hooks(model_eager)
            r = decompose_layer(tokenizer, model_eager, PROMPTS[phase_key]["text"], layer_idx)

            before = r.get("h_before", {}).get("top_digit", "?")
            post_attn = r.get("h_post_attn", {}).get("top_digit", "?")
            post_mlp = r.get("h_post_layer", {}).get("top_digit", "?")
            post_mlp_margin = r.get("h_post_layer", {}).get("margin")
            writer = get_writer(before, post_attn, post_mlp, phase_attractor)
            margin_str = f"{post_mlp_margin:.3f}" if post_mlp_margin is not None else "?"
            close_call = "  <-- close call" if post_mlp_margin is not None and post_mlp_margin < 0.5 else ""

            print(f"  L{layer_idx:02d}    {before:>10}  {post_attn:>10}  {post_mlp:>10}  {margin_str:>7}  {writer}{close_call}")

            decomp_results[f"{phase_key}_L{layer_idx:02d}"] = {
                "phase": phase_key, "layer": layer_idx, "attractor": phase_attractor,
                "h_before": r.get("h_before", {}),
                "h_post_attn": r.get("h_post_attn", {}),
                "h_post_layer": r.get("h_post_layer", {}),
            }

    save_path = output_path(cfg, "mlp_attn_decomp.json")
    with open(save_path, "w") as f:
        json.dump(decomp_results, f, indent=2)
    print(f"\nSaved: {save_path}")
    return decomp_results


def per_n_gap_decomposition(cfg, tokenizer, model_eager, attractor_p1: str):
    ns = [7, 8, 9, 10, 11, 12, 15]
    for layer_idx in cfg.gap_layers:
        print(f"\nPer-n MLP decomposition at L{layer_idx} — tracking '{attractor_p1}'")
        print(f"{'n':>4}  {'Before':>10}  {'Post-attn':>10}  {'Post-MLP':>10}  {'Margin':>7}  {'Wrote/erased?':>14}")
        print("-" * 56)

        for n in ns:
            prompt = make_prompt_repeated(n)
            remove_all_hooks(model_eager)
            r = decompose_layer(tokenizer, model_eager, prompt, layer_idx)

            before = r.get("h_before", {}).get("top_digit", "?")
            post_attn = r.get("h_post_attn", {}).get("top_digit", "?")
            post_mlp = r.get("h_post_layer", {}).get("top_digit", "?")
            post_mlp_margin = r.get("h_post_layer", {}).get("margin")
            writer = get_writer(before, post_attn, post_mlp, attractor_p1)
            margin_str = f"{post_mlp_margin:.3f}" if post_mlp_margin is not None else "?"
            close_call = "  <-- close call" if post_mlp_margin is not None and post_mlp_margin < 0.5 else ""

            print(f"{n:>4}  {before:>10}  {post_attn:>10}  {post_mlp:>10}  {margin_str:>7}  {writer:>14}{close_call}")


def full_layer_sweep(cfg, tokenizer, model_eager, attractor_p1: str):
    print(f"\nFull layer sweep — P1 n=10, tracking '{attractor_p1}' across all layers")
    print(f"{'Layer':>6}  {'Before':>10}  {'Post-attn':>10}  {'Post-MLP':>10}  {'Margin':>7}  Writer")

    prompt_n10 = make_prompt_repeated(10)
    sweep = {}
    for layer_idx in range(1, cfg.n_layers + 1):
        remove_all_hooks(model_eager)
        r = decompose_layer(tokenizer, model_eager, prompt_n10, layer_idx)

        before = r.get("h_before", {}).get("top_digit", "?")
        post_attn = r.get("h_post_attn", {}).get("top_digit", "?")
        post_mlp = r.get("h_post_layer", {}).get("top_digit", "?")
        post_mlp_margin = r.get("h_post_layer", {}).get("margin")
        writer = get_writer(before, post_attn, post_mlp, attractor_p1)

        if writer != "-" or attractor_p1 in [before, post_attn, post_mlp]:
            marker = " <--" if writer not in ["-", "(stable)"] else ""
            margin_str = f"{post_mlp_margin:.3f}" if post_mlp_margin is not None else "?"
            close_call = "  (close call)" if post_mlp_margin is not None and post_mlp_margin < 0.5 else ""
            print(f"  L{layer_idx:02d}    {before:>10}  {post_attn:>10}  {post_mlp:>10}  {margin_str:>7}  {writer}{marker}{close_call}")

        sweep[layer_idx] = {
            "before": before, "post_attn": post_attn, "post_mlp": post_mlp,
            "post_mlp_margin": post_mlp_margin, "writer": writer,
        }

    save_path = output_path(cfg, "full_layer_sweep_n10.json")
    with open(save_path, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"\nSaved: {save_path}")

    return sweep


def find_first_write_layer(sweep: dict):
    """First layer (ascending) where the attractor is written by attention
    or MLP anywhere in the sweep. NOT the same thing as the notebook's
    hand-labeled 'divergence_layer' (e.g. L22 for 3B) — that identifies the
    start of the write that survives to the final output. This can catch
    an earlier transient write that gets erased a layer or two later and
    never reappears until much deeper (3B: L11 writes '14', L12 erases it,
    and the real, surviving cascade only starts at L22) — both are
    genuinely interesting, they just answer different questions."""
    for layer_idx in sorted(sweep, key=int):
        if sweep[layer_idx]["writer"] in ("ATTENTION", "MLP"):
            return int(layer_idx)
    return None


def zero_ablate_mlp(tokenizer, model_eager, prompt_text: str, layer_idx: int):
    inputs = make_inputs_eager(tokenizer, model_eager, prompt_text)

    def zero_mlp_hook(module, input, output):
        return torch.zeros_like(output)

    handle = model_eager.model.layers[layer_idx - 1].mlp.register_forward_hook(zero_mlp_hook)
    with torch.no_grad():
        out = model_eager(**inputs)
    handle.remove()

    logits = out.logits[0, -1, :]
    top_digit = get_top_digit(tokenizer, logits)
    top5 = [tokenizer.decode([i]) for i in logits.topk(5).indices]
    return top_digit, top5


def ablation_experiment(cfg, tokenizer, model_eager):
    layer_idx = cfg.lockin_layer
    print("=" * 65)
    print(f"ZERO-ABLATION OF MLP L{layer_idx} — does removing it fix the output?")
    print("=" * 65)
    print(f"  {'n':>4}  {'Normal output':>14}  {'Ablated':>15}  {'Fixed?':>7}")
    print("  " + "-" * 48)

    ablation_results = {}
    for n in [8, 9, 10, 11, 12, 15]:
        prompt = make_prompt_repeated(n)
        remove_all_hooks(model_eager)

        inputs = make_inputs_eager(tokenizer, model_eager, prompt)
        with torch.no_grad():
            normal_out = model_eager(**inputs)
        normal_digit = get_top_digit(tokenizer, normal_out.logits[0, -1, :])

        ablated_digit, ablated_top5 = zero_ablate_mlp(tokenizer, model_eager, prompt, layer_idx)
        fixed = "✓ YES" if ablated_digit == str(n) else "✗ no"

        print(f"  {n:>4}  {normal_digit:>14}  {ablated_digit:>15}  {fixed:>7}  {ablated_top5}")
        ablation_results[n] = {
            "n": n, "normal": normal_digit,
            "ablated": ablated_digit, "fixed": ablated_digit == str(n),
            "ablated_top5": ablated_top5,
        }

    save_path = output_path(cfg, "mlp_ablation.json")
    with open(save_path, "w") as f:
        json.dump({str(k): v for k, v in ablation_results.items()}, f, indent=2)
    print(f"\nSaved: {save_path}")


def mechanistic_summary(cfg, attractors: dict, first_write_layer=None):
    summary = {
        "model": cfg.model_name,
        "n_layers": cfg.n_layers,
        "p1_attractor": attractors["phase1_baseline"],
        "p2_attractor": attractors["phase2_anomaly"],
        "p3_attractor": attractors["phase3_control"],
        "critical_layers": cfg.critical_layers,
        "gap_layers": cfg.gap_layers,
        # First layer where the attractor is written anywhere in the full
        # sweep — may be an early transient that gets erased before the
        # surviving cascade (lock_in_layer) actually takes hold. See
        # find_first_write_layer()'s docstring.
        "first_write_layer": first_write_layer,
        "lock_in_layer": cfg.lockin_layer,
        "lock_in_depth_pct": round(cfg.lockin_layer / cfg.n_layers, 3),
    }

    save_path = output_path(cfg, "mechanistic_summary.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {save_path}")


def paraphrase_decomposition(cfg, tokenizer, model_eager, pipe, attractor_p1: str):
    simple_prompt = PARAPHRASES["simple"].format(list=" ".join(["apple"] * 10))

    def margin_str_of(r):
        m = r.get("h_post_layer", {}).get("margin")
        return (f"{m:.3f}" if m is not None else "?"), (m is not None and m < 0.5)

    print(f"\n{'=' * 65}")
    print(f"PARAPHRASE DECOMPOSITION — 'simple' phrasing near L{cfg.lockin_layer}")
    print("=" * 65)
    print(f"  {'Layer':>6}  {'Before':>10}  {'Post-attn':>10}  {'Post-MLP':>10}  {'Margin':>7}  Writer")
    print("  " + "-" * 52)

    for layer_idx in range(cfg.lockin_layer, min(cfg.lockin_layer + 3, cfg.n_layers + 1)):
        remove_all_hooks(model_eager)
        r = decompose_layer(tokenizer, model_eager, simple_prompt, layer_idx)
        before = r.get("h_before", {}).get("top_digit", "?")
        post_attn = r.get("h_post_attn", {}).get("top_digit", "?")
        post_mlp = r.get("h_post_layer", {}).get("top_digit", "?")
        writer = get_writer(before, post_attn, post_mlp, attractor_p1)
        margin_str, close_call = margin_str_of(r)
        print(f"  L{layer_idx:02d}    {before:>10}  {post_attn:>10}  {post_mlp:>10}  {margin_str:>7}  {writer}"
              f"{'  <-- close call' if close_call else ''}")

    print(f"\nFull layer sweep — 'simple' paraphrase, tracking '{attractor_p1}'")
    print(f"  {'Layer':>6}  {'Before':>10}  {'Post-attn':>10}  {'Post-MLP':>10}  {'Margin':>7}  Writer")
    print("  " + "-" * 52)
    for layer_idx in range(1, cfg.n_layers + 1):
        remove_all_hooks(model_eager)
        r = decompose_layer(tokenizer, model_eager, simple_prompt, layer_idx)
        before = r.get("h_before", {}).get("top_digit", "?")
        post_attn = r.get("h_post_attn", {}).get("top_digit", "?")
        post_mlp = r.get("h_post_layer", {}).get("top_digit", "?")
        writer = get_writer(before, post_attn, post_mlp, attractor_p1)
        if attractor_p1 in [before, post_attn, post_mlp] or writer != "-":
            margin_str, close_call = margin_str_of(r)
            print(f"  L{layer_idx:02d}    {before:>10}  {post_attn:>10}  {post_mlp:>10}  {margin_str:>7}  {writer}"
                  f"{'  <-- close call' if close_call else ''}")

    print(f"\n{'=' * 65}")
    print(f"DIAGNOSTIC — residual entering L{cfg.lockin_layer} across paraphrases")
    print("=" * 65)
    print(f"  {'Paraphrase':<15}  {'Before':>10}  {'Post-attn':>10}  {'Post-MLP':>10}  {'Margin':>7}  Writer")
    print("  " + "-" * 60)
    for pname, template in PARAPHRASES.items():
        prompt = template.format(list=" ".join(["apple"] * 10))
        remove_all_hooks(model_eager)
        r = decompose_layer(tokenizer, model_eager, prompt, cfg.lockin_layer)
        before = r.get("h_before", {}).get("top_digit", "?")
        post_attn = r.get("h_post_attn", {}).get("top_digit", "?")
        post_mlp = r.get("h_post_layer", {}).get("top_digit", "?")
        writer = get_writer(before, post_attn, post_mlp, attractor_p1)
        margin_str, close_call = margin_str_of(r)
        print(f"  {pname:<15}  {before:>10}  {post_attn:>10}  {post_mlp:>10}  {margin_str:>7}  {writer}"
              f"{'  <-- close call' if close_call else ''}")

    print(f"\n{'=' * 65}")
    print(f"DIAGNOSTIC — rank of '{attractor_p1}' in final-layer logits across paraphrases")
    print("=" * 65)
    print(f"  {'Paraphrase':<15}  {'Output':>7}  {'Rank of attractor':>18}  {'Margin to #2':>13}")
    print("  " + "-" * 46)

    for pname, template in PARAPHRASES.items():
        prompt = template.format(list=" ".join(["apple"] * 10))
        remove_all_hooks(model_eager)
        inputs = make_inputs_eager(tokenizer, model_eager, prompt)
        with torch.no_grad():
            out = model_eager(**inputs)
        logits = out.logits[0, -1, :]
        sorted_ids = logits.argsort(descending=True)

        ids_attr = tokenizer.encode(attractor_p1, add_special_tokens=False)
        rank = (sorted_ids == ids_attr[0]).nonzero().item() + 1 if len(ids_attr) == 1 else "multi-token"
        _, margin = get_top_digit_and_margin(tokenizer, logits)
        margin_str = f"{margin:.3f}" if margin is not None else "?"
        close_call = "  <-- close call" if margin is not None and margin < 0.5 else ""

        r = run_single(pipe, tokenizer, "adhoc", {"text": prompt, "expected": 10}, seed=0,
                        max_new_tokens=8, temperature=0.0)
        print(f"  {pname:<15}  {str(r['predicted']):>7}  {str(rank):>18}  {margin_str:>13}{close_call}")

    print(f"\n{'=' * 65}")
    print("DIAGNOSTIC — full layer sweep per paraphrase")
    print("=" * 65)
    for pname, template in PARAPHRASES.items():
        prompt = template.format(list=" ".join(["apple"] * 10))
        print(f"\n[{pname}] — layers where '{attractor_p1}' appears:")
        print(f"  {'Layer':>6}  {'Before':>10}  {'Post-attn':>10}  {'Post-MLP':>10}  {'Margin':>7}  Writer")
        print("  " + "-" * 52)
        for layer_idx in range(1, cfg.n_layers + 1):
            remove_all_hooks(model_eager)
            r = decompose_layer(tokenizer, model_eager, prompt, layer_idx)
            before = r.get("h_before", {}).get("top_digit", "?")
            post_attn = r.get("h_post_attn", {}).get("top_digit", "?")
            post_mlp = r.get("h_post_layer", {}).get("top_digit", "?")
            writer = get_writer(before, post_attn, post_mlp, attractor_p1)
            if attractor_p1 in [before, post_attn, post_mlp] or writer != "-":
                margin_str, close_call = margin_str_of(r)
                print(f"  L{layer_idx:02d}    {before:>10}  {post_attn:>10}  {post_mlp:>10}  {margin_str:>7}  {writer}"
                      f"{'  <-- close call' if close_call else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    tokenizer, model, pipe = load_generation_model(cfg.model_name)
    attractors = get_attractors_via_pipe(pipe, tokenizer)
    print(f"Attractors from behavioral baseline: {attractors}")

    del model
    torch.cuda.empty_cache()
    tokenizer, model_eager = load_eager_model(cfg.model_name, tokenizer=tokenizer)

    print("Layer submodule names (confirms decompose_layer's hook targets exist):")
    for name, _ in model_eager.model.layers[0].named_children():
        print(f"  {name}")

    run_logit_lens(cfg, tokenizer, model_eager, attractors)
    mlp_attn_decomposition(cfg, tokenizer, model_eager, attractors)

    attractor_p1 = attractors["phase1_baseline"]
    per_n_gap_decomposition(cfg, tokenizer, model_eager, attractor_p1)
    ablation_experiment(cfg, tokenizer, model_eager)

    if cfg.runs_mechanistic_deep_dive:
        sweep = full_layer_sweep(cfg, tokenizer, model_eager, attractor_p1)
        first_write_layer = find_first_write_layer(sweep)
        print(f"\nFirst write of '{attractor_p1}' anywhere in the sweep: L{first_write_layer} "
              f"(may be transient — lock_in_layer=L{cfg.lockin_layer} is what survives to output)")
        mechanistic_summary(cfg, attractors, first_write_layer)
        paraphrase_decomposition(cfg, tokenizer, model_eager, pipe, attractor_p1)
    else:
        print(f"\n[skip] Full layer sweep / mechanistic summary / paraphrase decomposition "
              f"were only run against the 3B model in the source notebook — skipping for {cfg.key}.")


if __name__ == "__main__":
    main()

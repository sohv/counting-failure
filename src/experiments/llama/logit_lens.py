# logit lens across layers, mlp/attention decomposition, and per-n writer input at the identified writer layer.
# uv run -m src.experiments.llama.logit_lens --model llama-1b

import argparse
import json
import logging

import torch

from src.common.config import add_model_arg, get_config, load_results, output_path
from src.common.prompts import PARAPHRASES, PROMPTS, make_prompt_repeated
from src.common.utils import (
    decompose_layer,
    get_top_digit_and_margin,
    get_writer_argmax,
    get_writer_logit_diff,
    load_eager_model,
    load_generation_model,
    logit_lens_single,
    make_inputs_eager,
    remove_all_hooks,
    run_single,
)

LOGGER = logging.getLogger(__name__)


def get_attractors_via_pipe(pipe, tokenizer) -> dict:
    attractors = {}
    for phase_key, entry in PROMPTS.items():
        r = run_single(pipe, tokenizer, phase_key, entry, seed=0, max_new_tokens=16, temperature=0.0)
        attractors[phase_key] = str(r["predicted"])
    return attractors


def logit_lens_all_layers(
    tokenizer, model_eager, prompt_text: str,
    correct_answer: int, wrong_answer: int,
) -> list[dict]:
    """D1: project hidden state at each layer through logit lens, report logit_diff."""
    inputs = make_inputs_eager(tokenizer, model_eager, prompt_text)
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
            correct_answer=correct_answer, wrong_answer=wrong_answer,
        )
        r["layer"] = "embed" if layer_idx == 0 else f"L{layer_idx:02d}"
        results.append(r)
    return results


def run_logit_lens(cfg, tokenizer, model_eager, attractors: dict) -> dict:
    print("D1: Logit lens across layers (wrong-minus-correct logit difference)")

    lens_all = {}
    for phase_key in ["phase1_baseline", "phase2_anomaly", "phase3_control"]:
        correct = PROMPTS[phase_key]["expected"]
        wrong = int(attractors[phase_key]) if attractors[phase_key] not in ("None", "?") else correct
        print(f"\n  {phase_key}  correct={correct}  model_output={attractors[phase_key]}")
        print(f"  {'Layer':>6}  {'logit_diff':>10}  {'top_digit':>10}  {'margin':>8}")

        # no hooks registered yet at this point in the pipeline, so no cleanup is
        # needed here - calling remove_all_hooks() a second/third time (once per
        # phase) strips a hook transformers registers internally to populate
        # output_hidden_states, silently truncating every phase after the first
        # to a single (embed-only) entry.
        lens = logit_lens_all_layers(tokenizer, model_eager, PROMPTS[phase_key]["text"], correct, wrong)
        lens_all[phase_key] = lens
        for r in lens:
            ld = r.get("logit_diff", "n/a")
            ld_str = f"{ld:.4f}" if isinstance(ld, float) else str(ld)
            margin_str = f"{r['margin']:.4f}" if r["margin"] is not None else "?"
            print(f"  {r['layer']:>6}  {ld_str:>10}  {r['top_digit']:>10}  {margin_str:>8}")

    return lens_all


def mlp_attn_decomposition(
    cfg, tokenizer, model_eager, attractors: dict, critical_layers: list[int],
) -> dict:
    """D2: at critical layers, capture 3 residual states, report logit_diff and contributions."""
    print(f"\nD2: MLP vs attention decomposition at layers {critical_layers}")

    decomp_results = {}
    for phase_key in ["phase1_baseline", "phase3_control"]:
        correct = PROMPTS[phase_key]["expected"]
        wrong = int(attractors[phase_key]) if attractors[phase_key] not in ("None", "?") else correct
        print(f"\n  {phase_key}  correct={correct}  wrong={wrong}")
        print(f"  {'Layer':>6}  {'diff_before':>12}  {'diff_post_attn':>15}  {'diff_post_mlp':>14}  {'attn_contrib':>13}  {'mlp_contrib':>12}")

        for layer_idx in critical_layers:
            remove_all_hooks(model_eager)
            r = decompose_layer(
                tokenizer, model_eager, PROMPTS[phase_key]["text"], layer_idx,
                correct_answer=correct, wrong_answer=wrong,
            )

            diff_before = r.get("h_before", {}).get("logit_diff", float("nan"))
            diff_post_attn = r.get("h_post_attn", {}).get("logit_diff", float("nan"))
            diff_post_mlp = r.get("h_post_layer", {}).get("logit_diff", float("nan"))

            writer_info = get_writer_logit_diff(diff_before, diff_post_attn, diff_post_mlp)

            print(f"  L{layer_idx:02d}    {diff_before:>12.4f}  {diff_post_attn:>15.4f}"
                  f"  {diff_post_mlp:>14.4f}  {writer_info['attn_contribution']:>13.4f}"
                  f"  {writer_info['mlp_contribution']:>12.4f}")

            decomp_results[f"{phase_key}_L{layer_idx:02d}"] = {
                "phase": phase_key, "layer": layer_idx,
                "correct": correct, "wrong": wrong,
                "h_before": r.get("h_before", {}),
                "h_post_attn": r.get("h_post_attn", {}),
                "h_post_layer": r.get("h_post_layer", {}),
                "writer": writer_info,
            }

    return decomp_results


def per_n_writer_input(
    cfg, tokenizer, model_eager, correct_answers_match_n: bool,
    wrong_answer: int, gap_layers: list[int],
) -> dict:
    """D3: at the writer layer, record incoming logit_diff across n=7..15."""
    ns = [7, 8, 9, 10, 11, 12, 15]
    print(f"\nD3: Per-n writer input at layers {gap_layers}")

    results = {}
    for layer_idx in gap_layers:
        print(f"\n  Layer L{layer_idx:02d}, wrong_answer={wrong_answer}")
        print(f"  {'n':>4}  {'diff_before':>12}  {'diff_post_attn':>15}  {'diff_post_mlp':>14}  {'mlp_contrib':>12}")

        layer_results = {}
        for n in ns:
            correct = n
            prompt = make_prompt_repeated(n)
            remove_all_hooks(model_eager)
            r = decompose_layer(
                tokenizer, model_eager, prompt, layer_idx,
                correct_answer=correct, wrong_answer=wrong_answer,
            )

            diff_before = r.get("h_before", {}).get("logit_diff", float("nan"))
            diff_post_attn = r.get("h_post_attn", {}).get("logit_diff", float("nan"))
            diff_post_mlp = r.get("h_post_layer", {}).get("logit_diff", float("nan"))
            writer_info = get_writer_logit_diff(diff_before, diff_post_attn, diff_post_mlp)

            print(f"  {n:>4}  {diff_before:>12.4f}  {diff_post_attn:>15.4f}"
                  f"  {diff_post_mlp:>14.4f}  {writer_info['mlp_contribution']:>12.4f}")

            layer_results[n] = {
                "n": n, "correct": correct, "wrong": wrong_answer,
                "diff_before": diff_before,
                "diff_post_attn": diff_post_attn,
                "diff_post_mlp": diff_post_mlp,
                "writer": writer_info,
            }
        results[f"L{layer_idx:02d}"] = layer_results
    return results


def full_layer_sweep(
    cfg, tokenizer, model_eager, correct_answer: int, wrong_answer: int,
) -> dict:
    """Sweep all layers for n=10 to find the writer layer."""
    print(f"\nFull layer sweep for P1 n=10, correct={correct_answer}, wrong={wrong_answer}")
    print(f"  {'Layer':>6}  {'diff_before':>12}  {'diff_post_mlp':>14}  {'mlp_contrib':>12}")

    prompt = make_prompt_repeated(10)
    sweep = {}
    for layer_idx in range(1, cfg.n_layers + 1):
        remove_all_hooks(model_eager)
        r = decompose_layer(
            tokenizer, model_eager, prompt, layer_idx,
            correct_answer=correct_answer, wrong_answer=wrong_answer,
        )

        diff_before = r.get("h_before", {}).get("logit_diff", float("nan"))
        diff_post_attn = r.get("h_post_attn", {}).get("logit_diff", float("nan"))
        diff_post_mlp = r.get("h_post_layer", {}).get("logit_diff", float("nan"))
        writer_info = get_writer_logit_diff(diff_before, diff_post_attn, diff_post_mlp)

        # only print layers with significant contribution
        if abs(writer_info["mlp_contribution"]) > 0.1 or abs(writer_info["attn_contribution"]) > 0.1:
            print(f"  L{layer_idx:02d}    {diff_before:>12.4f}  {diff_post_mlp:>14.4f}  {writer_info['mlp_contribution']:>12.4f}")

        sweep[layer_idx] = {
            "diff_before": round(diff_before, 4),
            "diff_post_attn": round(diff_post_attn, 4),
            "diff_post_mlp": round(diff_post_mlp, 4),
            "writer": writer_info,
            "top_digit_before": r.get("h_before", {}).get("top_digit", "?"),
            "top_digit_post_mlp": r.get("h_post_layer", {}).get("top_digit", "?"),
        }

    return sweep


def find_writer_layer(sweep: dict, wrong_answer: int | str) -> int | None:
    """Find the earliest layer where the wrong digit becomes and remains the
    top post-MLP digit through the last layer (a persistent lock-in).

    Picking whichever single layer has the largest raw MLP contribution is
    unreliable: early layers are volatile in the logit-lens basis and can spike
    on tokens unrelated to the eventual answer, well before the model has
    settled on any digit at all. Requiring persistence to the final layer
    rules those transient swings out.
    """
    wrong_str = str(wrong_answer)
    layers = sorted(sweep.keys())
    for layer_idx in layers:
        if all(sweep[l]["top_digit_post_mlp"] == wrong_str for l in layers if l >= layer_idx):
            return layer_idx
    return None


def auto_discover_layers(cfg, sweep: dict, wrong_answer: int | str) -> tuple[list[int], int, list[int]]:
    """For models with no pre-set critical layers, discover them from the sweep."""
    writer = find_writer_layer(sweep, wrong_answer)
    if writer is None:
        LOGGER.warning("Could not find a persistent lock-in layer, defaulting to last layer")
        writer = cfg.n_layers

    critical_start = max(1, writer - 3)
    critical_end = min(cfg.n_layers, writer + 2)
    critical_layers = list(range(critical_start, critical_end + 1))
    gap_layers = [writer]

    print(f"  Auto-discovered: writer=L{writer}, critical={critical_layers}")
    return critical_layers, writer, gap_layers


def paraphrase_decomposition(
    cfg, tokenizer, model_eager, lockin_layer: int,
    correct_answer: int, wrong_answer: int,
) -> dict:
    """Run decomposition at the lockin layer across paraphrased prompts."""
    print(f"\nParaphrase decomposition at L{lockin_layer}")

    results = {}
    for pname, template in PARAPHRASES.items():
        prompt = template.format(list=" ".join(["apple"] * 10))
        remove_all_hooks(model_eager)
        r = decompose_layer(
            tokenizer, model_eager, prompt, lockin_layer,
            correct_answer=correct_answer, wrong_answer=wrong_answer,
        )

        diff_before = r.get("h_before", {}).get("logit_diff", float("nan"))
        diff_post_mlp = r.get("h_post_layer", {}).get("logit_diff", float("nan"))
        writer_info = get_writer_logit_diff(
            diff_before,
            r.get("h_post_attn", {}).get("logit_diff", float("nan")),
            diff_post_mlp,
        )

        print(f"  {pname:<15}  diff_before={diff_before:>8.4f}  diff_post_mlp={diff_post_mlp:>8.4f}"
              f"  mlp_contrib={writer_info['mlp_contribution']:>8.4f}")

        results[pname] = {
            "diff_before": diff_before,
            "diff_post_mlp": diff_post_mlp,
            "writer": writer_info,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    # load behavioral results to get attractor
    behavioral = load_results(cfg, "behavioral")
    attractor_p1 = behavioral["attractor"]
    correct_p1 = PROMPTS["phase1_baseline"]["expected"]
    wrong_p1 = int(attractor_p1) if attractor_p1 not in ("None", "?") else correct_p1

    # load model
    tokenizer, model, pipe = load_generation_model(cfg.model_name)
    attractors = get_attractors_via_pipe(pipe, tokenizer)

    del model
    torch.cuda.empty_cache()
    tokenizer, model_eager = load_eager_model(cfg.model_name, tokenizer=tokenizer)

    # D1: logit lens
    lens_all = run_logit_lens(cfg, tokenizer, model_eager, attractors)

    # full layer sweep (always run to find/confirm writer)
    sweep = full_layer_sweep(cfg, tokenizer, model_eager, correct_p1, wrong_p1)

    # determine critical layers
    if cfg.critical_layers is not None:
        critical_layers = cfg.critical_layers
        lockin_layer = cfg.lockin_layer
        gap_layers = cfg.gap_layers
    else:
        critical_layers, lockin_layer, gap_layers = auto_discover_layers(cfg, sweep, wrong_p1)

    writer_layer = find_writer_layer(sweep, wrong_p1)
    print(f"\nWriter layer (persistent lock-in): L{writer_layer}")

    # D2: decomposition
    decomp = mlp_attn_decomposition(cfg, tokenizer, model_eager, attractors, critical_layers)

    # D3: per-n writer input
    per_n = per_n_writer_input(cfg, tokenizer, model_eager, True, wrong_p1, gap_layers)

    # paraphrase decomposition
    paraphrase = paraphrase_decomposition(cfg, tokenizer, model_eager, lockin_layer, correct_p1, wrong_p1)

    # mechanistic summary
    summary = {
        "model": cfg.model_name,
        "n_layers": cfg.n_layers,
        "attractors": attractors,
        "critical_layers": critical_layers,
        "lockin_layer": lockin_layer,
        "gap_layers": gap_layers,
        "writer_layer": writer_layer,
        "writer_depth_pct": round(writer_layer / cfg.n_layers, 4) if writer_layer else None,
    }

    # save everything
    output = {
        "model": cfg.model_name,
        "attractors": attractors,
        "logit_lens": lens_all,
        "full_layer_sweep": {str(k): v for k, v in sweep.items()},
        "decomposition": decomp,
        "per_n_writer_input": per_n,
        "paraphrase_decomposition": paraphrase,
        "summary": summary,
    }

    save_path = output_path(cfg, "logit_lens.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

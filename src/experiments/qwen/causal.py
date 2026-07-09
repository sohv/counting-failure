# mean-ablation and targeted (p3->p1) patching at the qwen counting-writer sites, individually and jointly.
# uv run -m src.experiments.qwen.causal --model qwen-1.5b

import argparse
import json
import logging

import torch

from src.common.config import add_model_arg, get_config, load_results, output_path
from src.common.prompts import PROMPTS, make_prompt_repeated, make_prompt_unique
from src.common.utils import (
    denoising_patch_mlp,
    get_top_digit,
    load_eager_model,
    logit_difference,
    make_inputs_eager,
    mean_ablate_mlp,
    remove_all_hooks,
    single_token_digit_proxy,
)

LOGGER = logging.getLogger(__name__)


def _get_pre_logit_diff(tokenizer, model_eager, prompt: str, correct: int, wrong: int) -> tuple[float, str]:
    remove_all_hooks(model_eager)
    inputs = make_inputs_eager(tokenizer, model_eager, prompt)
    with torch.no_grad():
        out = model_eager(**inputs)
    logits = out.logits[0, -1, :]
    return logit_difference(tokenizer, logits, correct, wrong), get_top_digit(tokenizer, logits)


def run_mean_ablation_sites(tokenizer, model_eager, sites: list[int], wrong: int) -> dict:
    print(f"\nMean-ablation, individual + joint, sites={sites}")

    results = {}
    for n in [8, 9, 10, 11, 12, 15]:
        # qwen splits two-digit numbers into multiple tokens ("10" -> ["1","0"]),
        # so logit_difference needs a single-token proxy for n >= 10, not the raw value.
        correct = single_token_digit_proxy(tokenizer, n)
        target_prompt = make_prompt_repeated(n)
        reference_prompts = [make_prompt_unique(n)]
        pre_diff, _ = _get_pre_logit_diff(tokenizer, model_eager, target_prompt, correct, wrong)

        per_n = {"n": n, "correct": n, "correct_proxy": correct, "wrong": wrong, "pre_logit_diff": pre_diff}
        for site in sites:
            post = mean_ablate_mlp(tokenizer, model_eager, target_prompt, reference_prompts, [site], correct, wrong)
            per_n[f"L{site:02d}"] = {**post, "movement": round(pre_diff - post["logit_diff"], 4)}
        if len(sites) > 1:
            post = mean_ablate_mlp(tokenizer, model_eager, target_prompt, reference_prompts, sites, correct, wrong)
            per_n["joint"] = {**post, "movement": round(pre_diff - post["logit_diff"], 4)}

        site_str = "  ".join(f"L{s:02d}={per_n[f'L{s:02d}']['logit_diff']:.4f}" for s in sites)
        joint_str = f"  joint={per_n['joint']['logit_diff']:.4f}" if len(sites) > 1 else ""
        print(f"  n={n:<3}  correct_proxy={correct}  pre={pre_diff:>8.4f}  {site_str}{joint_str}")

        results[n] = per_n
    return results


def run_patch_sites(tokenizer, model_eager, sites: list[int], wrong: int) -> dict:
    print(f"\nTargeted patch (P3->P1, matched n), individual + joint, sites={sites}")

    results = {}
    for n in [8, 9, 10, 11, 12, 15]:
        correct = single_token_digit_proxy(tokenizer, n)
        target = make_prompt_repeated(n)
        source = make_prompt_unique(n)
        pre_diff, _ = _get_pre_logit_diff(tokenizer, model_eager, target, correct, wrong)

        per_n = {"n": n, "correct_proxy": correct, "pre_logit_diff": pre_diff}
        for site in sites:
            post = denoising_patch_mlp(tokenizer, model_eager, source, target, [site], correct, wrong)
            per_n[f"L{site:02d}"] = {**post, "movement": round(pre_diff - post["logit_diff"], 4)}
        if len(sites) > 1:
            post = denoising_patch_mlp(tokenizer, model_eager, source, target, sites, correct, wrong)
            per_n["joint"] = {**post, "movement": round(pre_diff - post["logit_diff"], 4)}

        site_str = "  ".join(f"L{s:02d}={per_n[f'L{s:02d}']['logit_diff']:.4f}" for s in sites)
        joint_str = f"  joint={per_n['joint']['logit_diff']:.4f}" if len(sites) > 1 else ""
        print(f"  n={n:<3}  correct_proxy={correct}  pre={pre_diff:>8.4f}  {site_str}{joint_str}")

        results[n] = per_n
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    behavioral = load_results(cfg, "behavioral")
    p1_attractor = str(behavioral["phases"]["phase1_baseline"][0]["predicted"])
    p1_correct = PROMPTS["phase1_baseline"]["expected"]
    model_fails_p1 = p1_attractor != str(p1_correct)

    print(f"Model          : {cfg.model_name}")
    print(f"P1 attractor   : {p1_attractor}  correct: {p1_correct}  fails_p1: {model_fails_p1}")

    if not model_fails_p1:
        print(f"{cfg.model_name} solves P1 - no counting-writer to ablate. Skipping.")
        return

    wrong = int(p1_attractor) if p1_attractor not in ("None", "?") else p1_correct - 1

    sites = cfg.ablation_sites
    if sites[0] is None:
        print(f"No writer_layers/lockin_layer set in config for {cfg.key}. Set cfg.writer_layers first.")
        return

    tokenizer, model_eager = load_eager_model(cfg.model_name)

    mean_ablation_sites = run_mean_ablation_sites(tokenizer, model_eager, sites, wrong)
    patch_sites = run_patch_sites(tokenizer, model_eager, sites, wrong)

    output = {
        "model": cfg.model_name,
        "wrong_answer": wrong,
        "ablation_sites": sites,
        "mean_ablation_sites": {str(k): v for k, v in mean_ablation_sites.items()},
        "patch_sites": {str(k): v for k, v in patch_sites.items()},
    }

    save_path = output_path(cfg, "causal_qwen.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

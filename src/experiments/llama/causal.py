"""
Every intervention scored by logit-difference movement (wrong-minus-correct),
not by whether the top digit flips.

E1: Zero-ablation of writer MLP.
E2: Mean-ablation (replace writer MLP output with P3 mean at matched n).
E3: Denoising patch (inject correct MLP output into failing case, norm-matched).
Supplementary: full-residual patching sweep, steering vector.

Usage:
    uv run -m src.experiments.llama.causal --model llama-1b
"""

import argparse
import json
import logging

import torch

from src.common.config import add_model_arg, get_config, load_results, output_path
from src.common.prompts import PROMPTS, make_prompt_repeated, make_prompt_unique
from src.common.utils import (
    extract_count,
    generate,
    get_top_digit,
    load_eager_model,
    load_generation_model,
    logit_difference,
    make_inputs_eager,
    remove_all_hooks,
)

LOGGER = logging.getLogger(__name__)


def _get_pre_logit_diff(tokenizer, model_eager, prompt: str, correct: int, wrong: int) -> tuple[float, str]:
    """Get baseline logit diff and top digit for a prompt without intervention."""
    remove_all_hooks(model_eager)
    inputs = make_inputs_eager(tokenizer, model_eager, prompt)
    with torch.no_grad():
        out = model_eager(**inputs)
    logits = out.logits[0, -1, :]
    return logit_difference(tokenizer, logits, correct, wrong), get_top_digit(tokenizer, logits)


# -- E1: Zero-ablation ------------------------------------------------------

def zero_ablate_mlp(
    tokenizer, model_eager, prompt: str, layer_idx: int,
    correct: int, wrong: int,
) -> dict:
    remove_all_hooks(model_eager)

    def zero_hook(module, input, output):
        return torch.zeros_like(output)

    handle = model_eager.model.layers[layer_idx - 1].mlp.register_forward_hook(zero_hook)
    inputs = make_inputs_eager(tokenizer, model_eager, prompt)
    with torch.no_grad():
        out = model_eager(**inputs)
    handle.remove()

    logits = out.logits[0, -1, :]
    return {
        "top_digit": get_top_digit(tokenizer, logits),
        "logit_diff": logit_difference(tokenizer, logits, correct, wrong),
        "top5": [tokenizer.decode([i]) for i in logits.topk(5).indices],
    }


def run_zero_ablation(cfg, tokenizer, model_eager, lockin_layer: int, wrong: int) -> dict:
    print(f"\nE1: Zero-ablation of MLP at L{lockin_layer}")
    print(f"  {'n':>4}  {'pre_diff':>10}  {'post_diff':>10}  {'movement':>10}  {'pre_digit':>10}  {'post_digit':>11}")

    results = {}
    for n in [8, 9, 10, 11, 12, 15]:
        prompt = make_prompt_repeated(n)
        pre_diff, pre_digit = _get_pre_logit_diff(tokenizer, model_eager, prompt, n, wrong)
        post = zero_ablate_mlp(tokenizer, model_eager, prompt, lockin_layer, n, wrong)
        movement = round(pre_diff - post["logit_diff"], 4)

        print(f"  {n:>4}  {pre_diff:>10.4f}  {post['logit_diff']:>10.4f}  {movement:>10.4f}"
              f"  {pre_digit:>10}  {post['top_digit']:>11}")

        results[n] = {
            "n": n, "correct": n, "wrong": wrong,
            "pre_logit_diff": pre_diff, "post_logit_diff": post["logit_diff"],
            "movement": movement,
            "pre_digit": pre_digit, "post_digit": post["top_digit"],
        }
    return results


# -- E2: Mean-ablation ------------------------------------------------------

def mean_ablate_mlp(
    tokenizer, model_eager, target_prompt: str,
    reference_prompts: list[str], layer_idx: int,
    correct: int, wrong: int,
) -> dict:
    """Replace writer MLP output with mean MLP output from reference prompts."""
    mlp_outputs = []
    for ref_prompt in reference_prompts:
        remove_all_hooks(model_eager)
        cache = {}

        def capture(module, input, output, _cache=cache):
            _cache["out"] = output.detach().clone() if isinstance(output, torch.Tensor) else output[0].detach().clone()

        handle = model_eager.model.layers[layer_idx - 1].mlp.register_forward_hook(capture)
        with torch.no_grad():
            model_eager(**make_inputs_eager(tokenizer, model_eager, ref_prompt))
        handle.remove()
        mlp_outputs.append(cache["out"][0, -1, :] if cache["out"].dim() > 1 else cache["out"])

    mean_mlp = torch.stack(mlp_outputs).mean(dim=0)

    remove_all_hooks(model_eager)

    def replace_hook(module, input, output, _mean=mean_mlp):
        if isinstance(output, torch.Tensor):
            patched = output.clone()
            patched[0, -1, :] = _mean
            return patched
        else:
            patched = output[0].clone()
            patched[0, -1, :] = _mean
            return (patched,) + output[1:]

    handle = model_eager.model.layers[layer_idx - 1].mlp.register_forward_hook(replace_hook)
    inputs = make_inputs_eager(tokenizer, model_eager, target_prompt)
    with torch.no_grad():
        out = model_eager(**inputs)
    handle.remove()

    logits = out.logits[0, -1, :]
    return {
        "top_digit": get_top_digit(tokenizer, logits),
        "logit_diff": logit_difference(tokenizer, logits, correct, wrong),
    }


def run_mean_ablation(cfg, tokenizer, model_eager, lockin_layer: int, wrong: int) -> dict:
    print(f"\nE2: Mean-ablation of MLP at L{lockin_layer} (reference: P3 at matched n)")
    print(f"  {'n':>4}  {'pre_diff':>10}  {'post_diff':>10}  {'movement':>10}  {'post_digit':>11}")

    results = {}
    for n in [8, 9, 10, 11, 12, 15]:
        target_prompt = make_prompt_repeated(n)
        reference_prompts = [make_prompt_unique(n)]

        pre_diff, _ = _get_pre_logit_diff(tokenizer, model_eager, target_prompt, n, wrong)
        post = mean_ablate_mlp(tokenizer, model_eager, target_prompt, reference_prompts, lockin_layer, n, wrong)
        movement = round(pre_diff - post["logit_diff"], 4)

        print(f"  {n:>4}  {pre_diff:>10.4f}  {post['logit_diff']:>10.4f}  {movement:>10.4f}  {post['top_digit']:>11}")

        results[n] = {
            "n": n, "correct": n, "wrong": wrong,
            "pre_logit_diff": pre_diff, "post_logit_diff": post["logit_diff"],
            "movement": movement, "post_digit": post["top_digit"],
        }
    return results


# -- E3: Denoising patch ----------------------------------------------------

def denoising_patch_mlp(
    tokenizer, model_eager, source_prompt: str, target_prompt: str,
    layer_idx: int, correct: int, wrong: int, norm_match: bool = True,
) -> dict:
    """Inject source MLP output into target forward pass, norm-matched."""
    # capture source MLP output
    remove_all_hooks(model_eager)
    source_cache = {}

    def capture_source(module, input, output, _cache=source_cache):
        o = output if isinstance(output, torch.Tensor) else output[0]
        _cache["out"] = o[0, -1, :].detach().clone()

    handle = model_eager.model.layers[layer_idx - 1].mlp.register_forward_hook(capture_source)
    with torch.no_grad():
        model_eager(**make_inputs_eager(tokenizer, model_eager, source_prompt))
    handle.remove()
    source_mlp = source_cache["out"]

    # capture target MLP norm for matching
    if norm_match:
        remove_all_hooks(model_eager)
        target_cache = {}

        def capture_target(module, input, output, _cache=target_cache):
            o = output if isinstance(output, torch.Tensor) else output[0]
            _cache["out"] = o[0, -1, :].detach().clone()

        handle = model_eager.model.layers[layer_idx - 1].mlp.register_forward_hook(capture_target)
        with torch.no_grad():
            model_eager(**make_inputs_eager(tokenizer, model_eager, target_prompt))
        handle.remove()
        target_norm = target_cache["out"].norm()
        source_mlp = source_mlp * (target_norm / (source_mlp.norm() + 1e-8))

    # inject
    remove_all_hooks(model_eager)

    def inject_hook(module, input, output, _src=source_mlp):
        if isinstance(output, torch.Tensor):
            patched = output.clone()
            patched[0, -1, :] = _src
            return patched
        else:
            patched = output[0].clone()
            patched[0, -1, :] = _src
            return (patched,) + output[1:]

    handle = model_eager.model.layers[layer_idx - 1].mlp.register_forward_hook(inject_hook)
    with torch.no_grad():
        out = model_eager(**make_inputs_eager(tokenizer, model_eager, target_prompt))
    handle.remove()

    logits = out.logits[0, -1, :]
    return {
        "top_digit": get_top_digit(tokenizer, logits),
        "logit_diff": logit_difference(tokenizer, logits, correct, wrong),
    }


def run_denoising_patch(cfg, tokenizer, model_eager, pipe, lockin_layer: int, wrong: int) -> dict:
    print(f"\nE3: Denoising patch at L{lockin_layer}")

    results = {"cross_condition": {}, "cross_n": {}}

    # Variant 1: P3 -> P1 at matched n
    print("  Variant 1: P3 (unique) -> P1 (repeated) at matched n")
    print(f"  {'n':>4}  {'pre_diff':>10}  {'post_diff':>10}  {'movement':>10}  {'post_digit':>11}")
    for n in [8, 9, 10, 11, 12]:
        source = make_prompt_unique(n)
        target = make_prompt_repeated(n)
        pre_diff, _ = _get_pre_logit_diff(tokenizer, model_eager, target, n, wrong)
        post = denoising_patch_mlp(tokenizer, model_eager, source, target, lockin_layer, n, wrong)
        movement = round(pre_diff - post["logit_diff"], 4)
        print(f"  {n:>4}  {pre_diff:>10.4f}  {post['logit_diff']:>10.4f}  {movement:>10.4f}  {post['top_digit']:>11}")
        results["cross_condition"][n] = {
            "n": n, "pre_logit_diff": pre_diff, "post_logit_diff": post["logit_diff"],
            "movement": movement, "post_digit": post["top_digit"],
        }

    # Variant 2: correct low-n -> failing high-n
    print("\n  Variant 2: correct low-n -> failing high-n (same format)")
    low_n = None
    for n in range(9, 4, -1):
        raw = generate(pipe, tokenizer, make_prompt_repeated(n), max_new_tokens=8)
        if extract_count(raw) == n:
            low_n = n
            break

    if low_n is not None:
        print(f"  Using n={low_n} (correct) as source")
        print(f"  {'target_n':>8}  {'pre_diff':>10}  {'post_diff':>10}  {'movement':>10}  {'post_digit':>11}")
        source = make_prompt_repeated(low_n)
        for target_n in [10, 11, 12, 15]:
            target = make_prompt_repeated(target_n)
            pre_diff, _ = _get_pre_logit_diff(tokenizer, model_eager, target, target_n, wrong)
            post = denoising_patch_mlp(tokenizer, model_eager, source, target, lockin_layer, target_n, wrong)
            movement = round(pre_diff - post["logit_diff"], 4)
            print(f"  {target_n:>8}  {pre_diff:>10.4f}  {post['logit_diff']:>10.4f}  {movement:>10.4f}  {post['top_digit']:>11}")
            results["cross_n"][target_n] = {
                "source_n": low_n, "target_n": target_n,
                "pre_logit_diff": pre_diff, "post_logit_diff": post["logit_diff"],
                "movement": movement, "post_digit": post["top_digit"],
            }
    else:
        print("  No correct low-n found, skipping variant 2")
        results["cross_n"]["skipped"] = True

    return results


# -- Residual patching sweep (supplementary) ---------------------------------

def patch_layer_sweep(
    tokenizer, model_eager, source_inputs, target_inputs,
    correct: int, wrong: int, n_layers: int,
) -> list[dict]:
    sweep = []
    for patch_layer in range(1, n_layers + 1):
        remove_all_hooks(model_eager)
        cache = {}

        def save_h(module, input, output, _l=patch_layer, _cache=cache):
            _cache[_l] = output[0, -1, :].detach().clone()

        def patch_h(module, input, output, _l=patch_layer, _cache=cache):
            patched = output.clone()
            patched[0, -1, :] = _cache[_l]
            return patched

        h1 = model_eager.model.layers[patch_layer - 1].register_forward_hook(save_h)
        with torch.no_grad():
            model_eager(**source_inputs)
        h1.remove()

        h2 = model_eager.model.layers[patch_layer - 1].register_forward_hook(patch_h)
        with torch.no_grad():
            pout = model_eager(**target_inputs)
        h2.remove()

        logits = pout.logits[0, -1, :]
        td = get_top_digit(tokenizer, logits)
        ld = logit_difference(tokenizer, logits, correct, wrong)
        sweep.append({"layer": patch_layer, "top_digit": td, "logit_diff": ld})
    return sweep


def run_residual_patching(cfg, tokenizer, model_eager, wrong: int) -> dict:
    print("\nResidual patching sweep: P3 -> P1 at n=10")

    source_inputs = make_inputs_eager(tokenizer, model_eager, PROMPTS["phase3_control"]["text"])
    target_inputs = make_inputs_eager(tokenizer, model_eager, PROMPTS["phase1_baseline"]["text"])

    pre_diff, pre_digit = _get_pre_logit_diff(
        tokenizer, model_eager, PROMPTS["phase1_baseline"]["text"], 10, wrong,
    )
    print(f"  Unpatched: digit={pre_digit}  logit_diff={pre_diff:.4f}")

    sweep = patch_layer_sweep(tokenizer, model_eager, source_inputs, target_inputs, 10, wrong, cfg.n_layers)

    print(f"  {'Layer':>6}  {'logit_diff':>10}  {'movement':>10}  {'top_digit':>10}")
    for r in sweep:
        movement = round(pre_diff - r["logit_diff"], 4)
        r["movement"] = movement
        r["pre_logit_diff"] = pre_diff
        if abs(movement) > 0.5:
            print(f"  L{r['layer']:02d}    {r['logit_diff']:>10.4f}  {movement:>10.4f}  {r['top_digit']:>10}")

    return {"pre_logit_diff": pre_diff, "pre_digit": pre_digit, "sweep": sweep}


# -- Steering vector (supplementary) ----------------------------------------

def run_steering_vector(cfg, tokenizer, model_eager, pipe, steer_layer: int, wrong: int) -> dict | None:
    print(f"\nSteering vector at L{steer_layer}")

    # find correct low-n
    low_n = None
    for n in range(9, 4, -1):
        raw = generate(pipe, tokenizer, make_prompt_repeated(n), max_new_tokens=8)
        if extract_count(raw) == n:
            low_n = n
            break

    if low_n is None:
        print("  No correct low-n found, skipping")
        return None

    high_n = 10
    print(f"  n_low={low_n} (correct) vs n_high={high_n} (wrong) at L{steer_layer}")

    prompt_low = make_prompt_repeated(low_n)
    prompt_high = make_prompt_repeated(high_n)

    def make_cache_hook(cache: dict):
        def hook(module, input, output):
            o = output[0] if isinstance(output, tuple) else output
            cache["h"] = o[0, -1, :].detach().clone()
        return hook

    cache_low, cache_high = {}, {}
    for prompt, cache in [(prompt_low, cache_low), (prompt_high, cache_high)]:
        remove_all_hooks(model_eager)
        handle = model_eager.model.layers[steer_layer - 1].register_forward_hook(make_cache_hook(cache))
        with torch.no_grad():
            model_eager(**make_inputs_eager(tokenizer, model_eager, prompt))
        handle.remove()

    steering_vec = cache_low["h"] - cache_high["h"]

    pre_diff, pre_digit = _get_pre_logit_diff(tokenizer, model_eager, prompt_high, high_n, wrong)
    print(f"  Baseline: digit={pre_digit}  logit_diff={pre_diff:.4f}")

    print(f"  {'alpha':>8}  {'logit_diff':>10}  {'movement':>10}  {'top_digit':>10}")
    sweep = []
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        def steer_hook(module, input, output, _a=alpha, _vec=steering_vec):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[0, -1, :] = h[0, -1, :] + _a * _vec
            return (h,) + output[1:] if isinstance(output, tuple) else h

        remove_all_hooks(model_eager)
        handle = model_eager.model.layers[steer_layer - 1].register_forward_hook(steer_hook)
        with torch.no_grad():
            out = model_eager(**make_inputs_eager(tokenizer, model_eager, prompt_high))
        handle.remove()

        logits = out.logits[0, -1, :]
        td = get_top_digit(tokenizer, logits)
        ld = logit_difference(tokenizer, logits, high_n, wrong)
        movement = round(pre_diff - ld, 4)

        print(f"  {alpha:>8.1f}  {ld:>10.4f}  {movement:>10.4f}  {td:>10}")
        sweep.append({"alpha": alpha, "logit_diff": ld, "movement": movement, "top_digit": td})

    return {
        "steer_layer": steer_layer,
        "n_low": low_n, "n_high": high_n,
        "steer_vec_norm": round(steering_vec.norm().item(), 4),
        "pre_logit_diff": pre_diff, "pre_digit": pre_digit,
        "sweep": sweep,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    # load dependencies from previous stages
    behavioral = load_results(cfg, "behavioral")
    logit_lens_results = load_results(cfg, "logit_lens")

    attractor = behavioral["attractor"]
    wrong = int(attractor) if attractor not in ("None", "?") else 10

    summary = logit_lens_results.get("summary", {})
    lockin_layer = summary.get("lockin_layer") or cfg.lockin_layer
    steer_layer = summary.get("lockin_layer") or cfg.steer_layer

    if lockin_layer is None:
        print("No lockin_layer found. Run logit_lens first.")
        return

    # load model
    tokenizer, model, pipe = load_generation_model(cfg.model_name)
    del model
    torch.cuda.empty_cache()
    tokenizer, model_eager = load_eager_model(cfg.model_name, tokenizer=tokenizer)

    # E1
    e1 = run_zero_ablation(cfg, tokenizer, model_eager, lockin_layer, wrong)

    # E2
    e2 = run_mean_ablation(cfg, tokenizer, model_eager, lockin_layer, wrong)

    # E3
    tokenizer_gen, _, pipe_gen = load_generation_model(cfg.model_name)
    e3 = run_denoising_patch(cfg, tokenizer, model_eager, pipe_gen, lockin_layer, wrong)

    # supplementary: residual patching
    residual_patch = run_residual_patching(cfg, tokenizer, model_eager, wrong)

    # supplementary: steering vector
    steering = run_steering_vector(cfg, tokenizer, model_eager, pipe_gen, steer_layer, wrong)

    output = {
        "model": cfg.model_name,
        "lockin_layer": lockin_layer,
        "wrong_answer": wrong,
        "zero_ablation": {str(k): v for k, v in e1.items()},
        "mean_ablation": {str(k): v for k, v in e2.items()},
        "denoising_patch": e3,
        "residual_patching": residual_patch,
        "steering_vector": steering,
    }

    save_path = output_path(cfg, "causal.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

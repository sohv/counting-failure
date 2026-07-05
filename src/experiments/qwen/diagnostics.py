# Diagnostics for Qwen models: probe dissociation, tokenizer limitation, direct logit check.
# Reads probe_results_qwen.json and logit_lens_qwen.json from mechanistic.py.
# Probe dissociation only runs when the model fails P1 (has a real wrong attractor).
# uv run -m src.experiments.qwen.diagnostics --model qwen-1.5b
# uv run -m src.experiments.qwen.diagnostics --model qwen-3b
# uv run -m src.experiments.qwen.diagnostics --model qwen-7b

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.config import add_model_arg, get_config, load_results, output_path
from src.common.prompts import PROMPTS

DISSOC_THRESHOLD = 0.95


def _lockin_layers_from_mechanistic(mechanistic: dict, n_layers: int) -> list[int]:
    """Derive lock-in layers from saved critical_layers, falling back to top-25% of layers."""
    critical = mechanistic.get("critical_layers")
    if critical:
        return critical
    # fallback: top 25% of depth
    lo = max(1, round(0.75 * n_layers))
    return list(range(lo, n_layers + 1))


def run_probe_dissociation(model_name: str, probe_results: dict,
                           lockin_layers: list[int]) -> dict:
    maes_rep = probe_results["repeated"]["maes"]
    r2s_rep  = probe_results["repeated"]["r2s"]
    maes_uniq = probe_results["unique"]["maes"]
    r2s_uniq  = probe_results["unique"]["r2s"]

    print("=" * 65)
    print(f"PROBE DISSOCIATION DIAGNOSTIC — {model_name}")
    print("=" * 65)
    print(f"\nLock-in layers: {lockin_layers}")
    print(f"\n{'Layer':>6}  {'R2(rep)':>9}  {'R2(uniq)':>9}  {'MAE(rep)':>9}  "
          f"{'MAE(uniq)':>10}  {'Dissociation?':>15}")
    print("-" * 70)

    for i in range(len(maes_rep)):
        label     = "embed" if i == 0 else f"L{i:02d}"
        r2r, r2u  = r2s_rep[i], r2s_uniq[i]
        mar, mau  = maes_rep[i], maes_uniq[i]
        is_lockin = i in lockin_layers
        dissoc    = ("YES"  if r2r > DISSOC_THRESHOLD and is_lockin else
                     "weak" if r2r > 0.90 and is_lockin else
                     "NO"   if is_lockin else "-")
        marker    = " <-- LOCK-IN" if is_lockin else ""
        print(f"{label:>6}  {r2r:>9.4f}  {r2u:>9.4f}  {mar:>9.4f}  "
              f"{mau:>10.4f}  {dissoc:>15}{marker}")

    lockin_r2s = [r2s_rep[i] for i in lockin_layers]
    mean_r2    = float(np.mean(lockin_r2s))
    min_r2     = float(np.min(lockin_r2s))
    print(f"\n  Mean R2(repeated) at lock-in layers: {mean_r2:.4f}")
    print(f"  Min  R2(repeated) at lock-in layers: {min_r2:.4f}")
    print(f"  Threshold: {DISSOC_THRESHOLD}")

    if mean_r2 > DISSOC_THRESHOLD:
        verdict = "holds"
        print("  VERDICT: Dissociation holds — count encoded but not used at lock-in layers")
    elif mean_r2 > 0.90:
        verdict = "weak"
        print("  VERDICT: Weak dissociation — count partially encoded at lock-in layers")
    else:
        verdict = "no"
        print("  VERDICT: No dissociation — count not encoded at lock-in layers")

    return {"mean_r2": round(mean_r2, 4), "min_r2": round(min_r2, 4),
            "lockin_layers": lockin_layers, "verdict": verdict}


def run_tokenizer_check(model_name: str, tokenizer) -> dict:
    print("\n" + "=" * 65)
    print(f"LOGIT LENS TOKENIZER DIAGNOSTIC — {model_name}")
    print("=" * 65)
    print(f"\n{'Digit':>6}  {'Token IDs':>20}  {'Decoded':>15}  Single token?")
    print("-" * 58)

    single_token_digits = []
    for n in range(1, 21):
        ids     = tokenizer.encode(str(n), add_special_tokens=False)
        decoded = [tokenizer.decode([i]) for i in ids]
        single  = "YES" if len(ids) == 1 else f"NO — {len(ids)} tokens"
        if len(ids) == 1:
            single_token_digits.append(n)
        print(f"{n:>6}  {str(ids):>20}  {str(decoded):>15}  {single}")

    ids_10        = tokenizer.encode("10", add_special_tokens=False)
    ten_multi     = len(ids_10) > 1
    print("\nImplication for logit lens:")
    if ten_multi:
        print(f"  '10' tokenizes to {ids_10} — multi-token, invisible to logit lens")
        print(f"  Trackable single-token digits: {single_token_digits}")
    else:
        print(f"  '10' is a single token — logit lens tracks it normally")

    return {"single_token_digits": single_token_digits, "ten_is_multi_token": ten_multi}


def run_direct_logit_check(model_name: str, tokenizer, model_eager) -> dict:
    print("\n" + "=" * 65)
    print(f"DIRECT LOGIT CHECK — P3 final layer, {model_name}")
    print("=" * 65)

    def remove_hooks(m):
        for mod in m.modules():
            mod._forward_hooks.clear()
            mod._forward_pre_hooks.clear()

    remove_hooks(model_eager)

    messages = [{"role": "user", "content": PROMPTS["phase3_control"]["text"]}]
    inputs   = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model_eager.device)

    final_h = {}
    handle  = model_eager.model.layers[-1].register_forward_hook(
        lambda m, i, o: final_h.update(
            {"h": (o[0] if isinstance(o, tuple) else o)[0, -1, :].detach().clone()}
        )
    )
    with torch.no_grad():
        model_eager(**inputs)
    handle.remove()

    normed = model_eager.model.norm(final_h["h"].unsqueeze(0).unsqueeze(0)).squeeze()
    logits = model_eager.lm_head.weight @ normed
    sorted_ids = logits.argsort(descending=True)

    print(f"\n{'Digit':>6}  {'Token IDs':>15}  {'Logit':>10}  {'Rank':>8}  Single?")
    print("-" * 55)
    digit_logits = {}
    for n in range(1, 21):
        ids    = tokenizer.encode(str(n), add_special_tokens=False)
        single = len(ids) == 1
        if single:
            val  = logits[ids[0]].item()
            rank = (sorted_ids == ids[0]).nonzero().item() + 1
            print(f"{n:>6}  {str(ids):>15}  {val:>10.4f}  {rank:>8}  YES")
            digit_logits[n] = {"logit": round(val, 4), "rank": rank}
        else:
            print(f"{n:>6}  {str(ids):>15}  {'N/A':>10}  {'N/A':>8}  NO — multi-token")

    print(f"\nTop-20 tokens at final layer (P3):")
    top20 = logits.topk(20)
    for i, (val, idx) in enumerate(zip(top20.values, top20.indices)):
        print(f"  {i+1:>3}. {repr(tokenizer.decode([idx])):<20}  logit={val.item():.4f}")

    return digit_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg  = get_config(args.model)

    behavioral   = load_results(cfg, "behavioral")
    p1_attractor = str(behavioral["phases"]["phase1_baseline"][0]["predicted"])
    p1_correct   = PROMPTS["phase1_baseline"]["expected"]
    model_fails_p1 = p1_attractor != str(p1_correct)

    print(f"Model        : {cfg.model_name}")
    print(f"P1 attractor : {p1_attractor}  fails_p1: {model_fails_p1}")

    mechanistic   = load_results(cfg, "mechanistic_qwen")
    probe_results = mechanistic["probe_results"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model_eager = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model_eager.eval()

    n_layers    = model_eager.config.num_hidden_layers
    lockin_layers = _lockin_layers_from_mechanistic(mechanistic, n_layers)

    output = {"model": cfg.model_name, "model_fails_p1": model_fails_p1}

    # Diagnostic 1: probe dissociation — only meaningful when model has a wrong attractor
    if model_fails_p1:
        output["probe_dissociation"] = run_probe_dissociation(
            cfg.model_name, probe_results, lockin_layers
        )
    else:
        print(f"\nSkipping probe dissociation — {cfg.model_name} passes P1 (no wrong attractor).")
        output["probe_dissociation"] = None

    # Diagnostic 2: tokenizer check — always run
    output["tokenizer_check"] = run_tokenizer_check(cfg.model_name, tokenizer)

    # Diagnostic 3: direct logit check at final layer — always run
    output["direct_logit_check"] = run_direct_logit_check(cfg.model_name, tokenizer, model_eager)

    save_path = output_path(cfg, "diagnostics_qwen.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()

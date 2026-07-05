# Mechanistic analysis for Qwen models: attention, linear probes, logit lens,
# MLP decomposition (1.5B only — fails P1), anomaly sweep (3B/7B — passes P1).
# Reads behavioral.json written by src.experiments.behavioral.
# uv run -m src.experiments.qwen.mechanistic --model qwen-1.5b
# uv run -m src.experiments.qwen.mechanistic --model qwen-3b
# uv run -m src.experiments.qwen.mechanistic --model qwen-7b

import argparse
import json
import logging
import re

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.common.config import add_model_arg, get_config, load_results, output_path
from src.common.prompts import PROMPTS

LOGGER = logging.getLogger(__name__)

UNIQUE_VOCAB = ["dog", "cat", "car", "red", "blue", "green",
                "house", "tree", "book", "pen", "fish", "cup",
                "hat", "sun", "moon", "sky", "fire", "rain", "snow", "wind"]


# ── Helpers ───────────────────────────────────────────────────────────────

def get_top_digit(logits_1d, tokenizer) -> str:
    candidates = {}
    for n in range(1, 21):
        ids = tokenizer.encode(str(n), add_special_tokens=False)
        if len(ids) == 1:
            candidates[str(n)] = logits_1d[ids[0]].item()
    return max(candidates, key=candidates.get) if candidates else "?"


def make_inputs_eager(prompt_text: str, tokenizer, model_eager):
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model_eager.device)


def remove_all_hooks(m):
    for module in m.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()


# ── Attention analysis ────────────────────────────────────────────────────

def find_word_positions(tokens: list[str], phase_key: str) -> list[int]:
    content_words = {
        "phase1_baseline": {"apple"},
        "phase2_anomaly" : {"apple", "banana"},
        "phase3_control" : {"dog", "cat", "car", "red", "blue",
                            "green", "house", "tree", "book", "pen"},
    }
    valid     = content_words[phase_key]
    colon_idx = max(i for i, t in enumerate(tokens) if t.strip() == ":")
    positions = []
    for i in range(colon_idx + 1, len(tokens)):
        if tokens[i].strip() in valid:
            positions.append(i)
        elif "." in tokens[i] and positions:
            break
    return positions


def run_attention_analysis(tokenizer, model_eager) -> dict:
    print("\n" + "=" * 70)
    print("ATTENTION ANALYSIS — word-list tokens only")
    print("=" * 70)

    attn_summary = {}
    for phase_key in ["phase1_baseline", "phase2_anomaly", "phase3_control"]:
        inputs = make_inputs_eager(PROMPTS[phase_key]["text"], tokenizer, model_eager)
        with torch.no_grad():
            out = model_eager(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
            )
        attentions = torch.stack(out.attentions).squeeze(1)
        input_ids  = inputs["input_ids"][0]

        tokens         = [tokenizer.decode([t]) for t in input_ids]
        word_positions = find_word_positions(tokens, phase_key)
        word_tokens    = [tokens[p] for p in word_positions]

        per_head       = attentions.cpu().float().numpy()
        word_attn      = per_head[:, :, -1, :][:, :, word_positions]
        word_attn_norm = word_attn / (word_attn.sum(axis=-1, keepdims=True) + 1e-9)
        mean_word_attn = word_attn_norm.mean(axis=1)

        entropies = []
        for l in range(mean_word_attn.shape[0]):
            a = mean_word_attn[l] / (mean_word_attn[l].sum() + 1e-9)
            entropies.append(round(float(-(a * np.log(a + 1e-9)).sum()), 4))

        uniformity = (mean_word_attn.min(axis=-1) / (mean_word_attn.max(axis=-1) + 1e-9)).mean()

        print(f"\n[{phase_key}]")
        print(f"  Word positions : {word_positions}")
        print(f"  Word tokens    : {word_tokens}")
        print(f"  Mean entropy   : {np.mean(entropies):.4f}")
        print(f"  Uniformity     : {float(uniformity):.4f}")

        attn_summary[phase_key] = {
            "word_positions": word_positions, "word_tokens": word_tokens,
            "entropy_per_layer": entropies, "mean_uniformity": float(uniformity),
        }
    return attn_summary


# ── Linear probes ─────────────────────────────────────────────────────────

def run_linear_probes(tokenizer, model_eager) -> dict:
    probe_ns = list(range(3, 14))
    labels   = np.array(probe_ns, dtype=float)

    repeated_prompts = [
        f'Count the number of times "apple" appears in this list: '
        + " ".join(["apple"] * n)
        + ". Respond only with the integer, nothing else."
        for n in probe_ns
    ]
    unique_prompts = [
        "Count the number of words in this list: "
        + " ".join(UNIQUE_VOCAB[:n])
        + ". Respond only with the integer, nothing else."
        for n in probe_ns
    ]

    def get_hidden(prompt):
        inputs = make_inputs_eager(prompt, tokenizer, model_eager)
        with torch.no_grad():
            out = model_eager(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
        return torch.stack(out.hidden_states)[:, 0, -1, :].cpu().float().numpy()

    print("\nCollecting activations...")
    hidden_rep  = np.stack([get_hidden(p) for p in repeated_prompts])
    hidden_uniq = np.stack([get_hidden(p) for p in unique_prompts])

    def loo_probe(hidden_array):
        loo  = LeaveOneOut()
        maes, r2s = [], []
        for li in range(hidden_array.shape[1]):
            X     = StandardScaler().fit_transform(hidden_array[:, li, :])
            preds = np.zeros(len(labels))
            for tr, te in loo.split(X):
                clf = Ridge(alpha=1.0)
                clf.fit(X[tr], labels[tr])
                preds[te] = clf.predict(X[te])
            maes.append(mean_absolute_error(labels, preds))
            r2s.append(r2_score(labels, preds))
        return maes, r2s

    print("Running probes...")
    maes_rep,  r2s_rep  = loo_probe(hidden_rep)
    maes_uniq, r2s_uniq = loo_probe(hidden_uniq)

    print(f"\n{'Layer':>6}  {'MAE(rep)':>10}  {'R2(rep)':>9}  "
          f"{'MAE(uniq)':>10}  {'R2(uniq)':>9}  {'ΔMAE':>8}")
    print("-" * 62)
    for i in range(hidden_rep.shape[1]):
        label = "embed" if i == 0 else f"L{i:02d}"
        print(f"{label:>6}  {maes_rep[i]:>10.4f}  {r2s_rep[i]:>9.4f}  "
              f"{maes_uniq[i]:>10.4f}  {r2s_uniq[i]:>9.4f}  {maes_rep[i]-maes_uniq[i]:>+8.4f}")

    return {
        "ns": probe_ns, "labels": probe_ns,
        "repeated": {"maes": maes_rep,  "r2s": r2s_rep},
        "unique"  : {"maes": maes_uniq, "r2s": r2s_uniq},
    }


# ── Logit lens + auto-discover critical layers ────────────────────────────

def logit_lens_single(h, tokenizer, model_eager) -> tuple[str, list]:
    with torch.no_grad():
        normed = model_eager.model.norm(h.unsqueeze(0).unsqueeze(0)).squeeze()
        logits = model_eager.lm_head.weight @ normed
    return get_top_digit(logits, tokenizer), [tokenizer.decode([i]) for i in logits.topk(5).indices]


def run_logit_lens(cfg, tokenizer, model_eager) -> tuple[dict, list[int] | None]:
    n_layers = model_eager.config.num_hidden_layers
    print(f"\n{'='*70}")
    print(f"LOGIT LENS — {cfg.model_name}  ({n_layers} layers)")
    print("=" * 70)

    lens_results = {}
    for phase_key in ["phase1_baseline", "phase3_control"]:
        correct = PROMPTS[phase_key]["expected"]
        print(f"\n[{phase_key}]  correct={correct}")
        print(f"  {'Layer':>6}  {'Top digit':>10}  Top-5 tokens")
        print("  " + "-" * 55)
        inputs = make_inputs_eager(PROMPTS[phase_key]["text"], tokenizer, model_eager)
        with torch.no_grad():
            out = model_eager(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
        lens = []
        for idx, h in enumerate(out.hidden_states):
            td, top5 = logit_lens_single(h[0, -1, :], tokenizer, model_eager)
            entry = {"layer": "embed" if idx == 0 else f"L{idx:02d}",
                     "top_digit": td, "top5": top5}
            lens.append(entry)
            print(f"  {entry['layer']:>6}  {td:>10}  {top5}")
        lens_results[phase_key] = lens

    # auto-discover writer: scan P1 lens from top down, find layer where digit first
    # transitions away from the final attractor
    p1_lens   = lens_results["phase1_baseline"]
    attractor = p1_lens[-1]["top_digit"]  # last layer = final prediction
    writer    = None
    for i in range(len(p1_lens) - 2, -1, -1):
        if p1_lens[i]["top_digit"] != attractor:
            writer = i + 1  # layer index (1-based) where attractor first appears
            break

    critical_layers = None
    if writer is not None:
        lo = max(1, writer - 2)
        hi = min(n_layers, writer + 2)
        critical_layers = list(range(lo, hi + 1))
        print(f"\nAuto-discovered writer layer: L{writer}  critical={critical_layers}")
    else:
        print(f"\nAttractor stable from embed — could not auto-discover writer layer.")

    print(f"87.5% depth equivalent: L{round(0.875 * n_layers)}")
    return lens_results, critical_layers


# ── MLP decomposition (1.5B path: model fails P1) ─────────────────────────

def decompose_layer(prompt_text: str, layer_idx: int, tokenizer, model_eager) -> dict:
    inputs = make_inputs_eager(prompt_text, tokenizer, model_eager)
    cache  = {}
    layer  = model_eager.model.layers[layer_idx - 1]

    def hook_pre(module, input):
        h = input[0] if isinstance(input, tuple) else input
        if isinstance(h, torch.Tensor):
            cache["h_before"] = h[0, -1, :].detach().clone()

    def hook_post_attn(module, input, output):
        h = input[0] if isinstance(input, tuple) else input
        if isinstance(h, torch.Tensor):
            cache["h_post_attn"] = h[0, -1, :].detach().clone()

    def hook_post_layer(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, torch.Tensor):
            cache["h_post_layer"] = h[0, -1, :].detach().clone()

    h1 = layer.register_forward_pre_hook(hook_pre)
    h2 = layer.post_attention_layernorm.register_forward_hook(hook_post_attn)
    h3 = layer.register_forward_hook(hook_post_layer)
    with torch.no_grad():
        model_eager(**inputs)
    h1.remove(); h2.remove(); h3.remove()

    return {
        name: {"top_digit": logit_lens_single(h, tokenizer, model_eager)[0],
               "top5"     : logit_lens_single(h, tokenizer, model_eager)[1]}
        for name, h in cache.items()
    }


def run_mlp_decomposition(cfg, tokenizer, model_eager,
                          all_results: dict, critical_layers: list[int]) -> tuple[dict, dict]:
    decomp_results = {}
    for phase_key in ["phase1_baseline", "phase3_control"]:
        wrong_answer = str(all_results[phase_key][0]["predicted"])
        correct      = PROMPTS[phase_key]["expected"]
        if wrong_answer == str(correct):
            wrong_answer = str(correct - 1)

        print(f"\n{'='*72}")
        print(f"MLP vs ATTENTION DECOMPOSITION — {phase_key}  wrong='{wrong_answer}'")
        print(f"{'='*72}")
        print(f"  {'Layer':>6}  {'Before':>10}  {'Post-attn':>10}  {'Post-MLP':>10}  Writer")
        print("  " + "-" * 58)

        for layer_idx in critical_layers:
            remove_all_hooks(model_eager)
            r         = decompose_layer(PROMPTS[phase_key]["text"], layer_idx, tokenizer, model_eager)
            before    = r.get("h_before",    {}).get("top_digit", "?")
            post_attn = r.get("h_post_attn", {}).get("top_digit", "?")
            post_mlp  = r.get("h_post_layer",{}).get("top_digit", "?")
            writer = (
                "ATTENTION" if before != wrong_answer and post_attn == wrong_answer else
                "MLP"       if post_attn != wrong_answer and post_mlp == wrong_answer else
                "(already)" if before == wrong_answer else "-"
            )
            print(f"  L{layer_idx:02d}    {before:>10}  {post_attn:>10}  {post_mlp:>10}  {writer}")
            decomp_results[f"{phase_key}_L{layer_idx:02d}"] = {
                "phase": phase_key, "layer": layer_idx,
                "h_before"    : r.get("h_before",    {}),
                "h_post_attn" : r.get("h_post_attn", {}),
                "h_post_layer": r.get("h_post_layer",{}),
            }

    lockin_layer = critical_layers[len(critical_layers) // 2]
    wrong_answer = str(all_results["phase1_baseline"][0]["predicted"])
    if wrong_answer == str(PROMPTS["phase1_baseline"]["expected"]):
        wrong_answer = str(PROMPTS["phase1_baseline"]["expected"] - 1)

    print(f"\nPer-n MLP decomposition at L{lockin_layer}  wrong='{wrong_answer}'")
    print(f"{'n':>4}  {'Before':>10}  {'Post-attn':>10}  {'Post-MLP':>10}  {'MLP wrote wrong?':>18}")
    print("-" * 60)

    pern_results = {}
    for n in [7, 8, 9, 10, 11, 12, 15]:
        prompt = (
            f'Count the number of times "apple" appears in this list: '
            + " ".join(["apple"] * n)
            + ". Respond only with the integer, nothing else."
        )
        remove_all_hooks(model_eager)
        r         = decompose_layer(prompt, lockin_layer, tokenizer, model_eager)
        before    = r.get("h_before",    {}).get("top_digit", "?")
        post_attn = r.get("h_post_attn", {}).get("top_digit", "?")
        post_mlp  = r.get("h_post_layer",{}).get("top_digit", "?")
        wrote_wrong = (
            "YES"     if post_attn != wrong_answer and post_mlp == wrong_answer else
            "already" if before == wrong_answer else "no"
        )
        print(f"{n:>4}  {before:>10}  {post_attn:>10}  {post_mlp:>10}  {wrote_wrong:>18}")
        pern_results[n] = {
            "n": n, "before": before, "post_attn": post_attn,
            "post_mlp": post_mlp, "wrote_wrong": wrote_wrong,
        }
    return decomp_results, pern_results


# ── Anomaly sweep (3B/7B path: model passes P1) ───────────────────────────

def run_anomaly_sweep(cfg, pipe, tokenizer) -> dict:
    print(f"\n{'='*65}")
    print(f"ANOMALY DETECTION SWEEP — {cfg.model_name}")
    print("=" * 65)

    def query(prompt: str) -> int | None:
        raw = pipe(
            [{"role": "user", "content": prompt}],
            max_new_tokens=8, temperature=0.0, do_sample=False,
            pad_token_id=tokenizer.eos_token_id, return_full_text=False,
        )[0]["generated_text"].strip()
        m = re.search(r'\b(\d+)\b', raw.strip())
        return int(m.group(1)) if m else None

    print("\n[Test 1] Vary banana position (1 banana, 9 apples)")
    print(f"  {'Position':>10}  {'Expected':>9}  {'Output':>8}  {'Detected?':>10}")
    pos_results = {}
    for pos in range(10):
        words = ["apple"] * 10
        words[pos] = "banana"
        pred = query(
            'Count the number of times "apple" appears in this list: '
            + " ".join(words)
            + ". Respond only with the integer, nothing else."
        )
        detected = pred == 9
        print(f"  {pos:>10}  {9:>9}  {str(pred):>8}  {'YES ✓' if detected else 'NO ✗':>10}")
        pos_results[pos] = {"predicted": pred, "detected": detected}

    print("\n[Test 2] Vary number of bananas (positions 0..n-1)")
    print(f"  {'N bananas':>10}  {'Expected':>9}  {'Output':>8}  {'Detected?':>10}")
    qty_results = {}
    for n_bananas in range(1, 6):
        words    = ["banana"] * n_bananas + ["apple"] * (10 - n_bananas)
        expected = 10 - n_bananas
        pred     = query(
            'Count the number of times "apple" appears in this list: '
            + " ".join(words)
            + ". Respond only with the integer, nothing else."
        )
        detected = pred == expected
        print(f"  {n_bananas:>10}  {expected:>9}  {str(pred):>8}  {'YES ✓' if detected else 'NO ✗':>10}")
        qty_results[n_bananas] = {"expected": expected, "predicted": pred, "detected": detected}

    threshold = next((k for k, v in qty_results.items() if v["detected"]), None)
    print(f"\nDetection threshold: {threshold} banana(s) needed")

    return {
        "position_results" : {str(k): v for k, v in pos_results.items()},
        "quantity_results" : {str(k): v for k, v in qty_results.items()},
        "detection_threshold": threshold,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg  = get_config(args.model)

    behavioral   = load_results(cfg, "behavioral")
    all_results  = behavioral["phases"]
    p1_attractor = str(all_results["phase1_baseline"][0]["predicted"])
    p1_correct   = PROMPTS["phase1_baseline"]["expected"]
    model_fails_p1 = p1_attractor != str(p1_correct)

    print(f"Model          : {cfg.model_name}")
    print(f"P1 attractor   : {p1_attractor}  correct: {p1_correct}  fails_p1: {model_fails_p1}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model_eager = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model_eager.eval()
    print(f"Eager model loaded. Layers: {model_eager.config.num_hidden_layers}")

    print("\nLayer submodule names:")
    for name, _ in model_eager.model.layers[0].named_children():
        print(f"  {name}")

    attn_summary  = run_attention_analysis(tokenizer, model_eager)
    probe_results = run_linear_probes(tokenizer, model_eager)
    lens_results, auto_critical = run_logit_lens(cfg, tokenizer, model_eager)

    # prefer config-set layers; fall back to auto-discovered
    critical_layers = cfg.critical_layers if cfg.critical_layers is not None else auto_critical

    output = {
        "model"          : cfg.model_name,
        "model_fails_p1" : model_fails_p1,
        "p1_attractor"   : p1_attractor,
        "critical_layers": critical_layers,
        "attention"      : attn_summary,
        "probe_results"  : probe_results,
        "logit_lens"     : lens_results,
    }

    if model_fails_p1:
        if critical_layers is None:
            LOGGER.warning("No critical layers found — skipping MLP decomposition. "
                           "Set critical_layers in config.py for %s.", cfg.key)
        else:
            decomp, pern = run_mlp_decomposition(cfg, tokenizer, model_eager,
                                                  all_results, critical_layers)
            output["mlp_decomposition"] = decomp
            output["per_n_decomp"]      = {str(k): v for k, v in pern.items()}
            decomp_path = output_path(cfg, "mlp_decomp_qwen.json")
            with open(decomp_path, "w") as f:
                json.dump({"decomposition": decomp,
                           "per_n": {str(k): v for k, v in pern.items()}}, f, indent=2)
            print(f"Saved: {decomp_path}")
    else:
        # load generation model for anomaly sweep
        model_gen = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        pipe = pipeline("text-generation", model=model_gen,
                        tokenizer=tokenizer, device_map="auto")
        anomaly = run_anomaly_sweep(cfg, pipe, tokenizer)
        output["anomaly_sweep"] = anomaly
        anomaly_path = output_path(cfg, "anomaly_sweep_qwen.json")
        with open(anomaly_path, "w") as f:
            json.dump(anomaly, f, indent=2)
        print(f"Saved: {anomaly_path}")

    save_path = output_path(cfg, "mechanistic_qwen.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

# mechanistic analysis for qwen: attention, linear probes, logit lens, mlp decomposition, and anomaly sweep.
# uv run -m src.experiments.qwen.mechanistic --model qwen-1.5b

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
from src.common.prompts import PARAPHRASES, PROMPTS
from src.common.utils import decompose_layer as decompose_layer_with_diff
from src.common.utils import get_writer_logit_diff, single_token_digit_proxy

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


def paraphrase_decomposition_qwen(
    cfg, tokenizer, model_eager, sites: list[int],
    correct_answer: int, wrong_answer: int,
) -> dict:
    """Decomposition at each writer site across paraphrased prompts, mirroring
    llama.logit_lens.paraphrase_decomposition. Tests whether the same
    input-state-matching account explains why no paraphrase bypasses the
    prior in Qwen-1.5B (uses the shared decompose_layer, which computes
    logit_diff; the local decompose_layer above only tracks top digit)."""
    print(f"\nParaphrase decomposition at sites {sites}")

    # qwen splits "10" into ["1","0"], so logit_diff needs a single-token proxy
    correct_proxy = single_token_digit_proxy(tokenizer, correct_answer)
    wrong_proxy = single_token_digit_proxy(tokenizer, wrong_answer)
    print(f"  correct_proxy={correct_proxy}  wrong_proxy={wrong_proxy}")

    results = {}
    for pname, template in PARAPHRASES.items():
        prompt = template.format(list=" ".join(["apple"] * 10))
        results[pname] = {}
        for site in sites:
            remove_all_hooks(model_eager)
            r = decompose_layer_with_diff(
                tokenizer, model_eager, prompt, site,
                correct_answer=correct_proxy, wrong_answer=wrong_proxy,
            )
            diff_before = r.get("h_before", {}).get("logit_diff", float("nan"))
            diff_post_mlp = r.get("h_post_layer", {}).get("logit_diff", float("nan"))
            writer_info = get_writer_logit_diff(
                diff_before,
                r.get("h_post_attn", {}).get("logit_diff", float("nan")),
                diff_post_mlp,
            )
            print(f"  {pname:<15}  L{site:02d}  diff_before={diff_before:>8.4f}  diff_post_mlp={diff_post_mlp:>8.4f}"
                  f"  mlp_contrib={writer_info['mlp_contribution']:>8.4f}")
            results[pname][f"L{site:02d}"] = {
                "diff_before": diff_before,
                "diff_post_mlp": diff_post_mlp,
                "writer": writer_info,
            }
    return results


# ── Anomaly sweep (3B/7B path: model passes P1) ───────────────────────────

INTRUDER_TOKENS = ["banana", "car", "seven", "xyz"]
BASE_NS = [8, 10, 12]


def is_single_token(tokenizer, word: str) -> bool:
    return len(tokenizer.encode(word, add_special_tokens=False)) == 1


def run_anomaly_sweep(cfg, pipe, tokenizer, intruder_token: str = "banana", base_n: int = 10) -> dict:
    print(f"\n{'='*65}")
    print(f"ANOMALY DETECTION SWEEP — {cfg.model_name}  intruder='{intruder_token}'  base_n={base_n}")
    print("=" * 65)

    def query(prompt: str) -> int | None:
        raw = pipe(
            [{"role": "user", "content": prompt}],
            max_new_tokens=8, temperature=0.0, do_sample=False,
            pad_token_id=tokenizer.eos_token_id, return_full_text=False,
        )[0]["generated_text"].strip()
        m = re.search(r'\b(\d+)\b', raw.strip())
        return int(m.group(1)) if m else None

    print(f"\n[Test 1] Vary {intruder_token} position (1 intruder, {base_n - 1} apples)")
    print(f"  {'Position':>10}  {'Expected':>9}  {'Output':>8}  {'Detected?':>10}")
    pos_results = {}
    for pos in range(base_n):
        words = ["apple"] * base_n
        words[pos] = intruder_token
        pred = query(
            'Count the number of times "apple" appears in this list: '
            + " ".join(words)
            + ". Respond only with the integer, nothing else."
        )
        detected = pred == base_n - 1
        print(f"  {pos:>10}  {base_n - 1:>9}  {str(pred):>8}  {'YES ✓' if detected else 'NO ✗':>10}")
        pos_results[pos] = {"predicted": pred, "detected": detected}

    max_intruders = min(5, base_n - 1)
    print(f"\n[Test 2] Vary number of {intruder_token}s (positions 0..n-1)")
    print(f"  {'N intruders':>12}  {'Expected':>9}  {'Output':>8}  {'Detected?':>10}")
    qty_results = {}
    for n_intruders in range(1, max_intruders + 1):
        words    = [intruder_token] * n_intruders + ["apple"] * (base_n - n_intruders)
        expected = base_n - n_intruders
        pred     = query(
            'Count the number of times "apple" appears in this list: '
            + " ".join(words)
            + ". Respond only with the integer, nothing else."
        )
        detected = pred == expected
        print(f"  {n_intruders:>12}  {expected:>9}  {str(pred):>8}  {'YES ✓' if detected else 'NO ✗':>10}")
        qty_results[n_intruders] = {"expected": expected, "predicted": pred, "detected": detected}

    threshold = next((k for k, v in qty_results.items() if v["detected"]), None)
    print(f"\nDetection threshold: {threshold} {intruder_token}(s) needed")

    return {
        "intruder_token"    : intruder_token,
        "base_n"            : base_n,
        "position_results"  : {str(k): v for k, v in pos_results.items()},
        "quantity_results"  : {str(k): v for k, v in qty_results.items()},
        "detection_threshold": threshold,
    }


def run_anomaly_variation(cfg, pipe, tokenizer) -> dict:
    """Vary the intruder token (base_n fixed at 10) and the base length
    (intruder fixed at 'banana') to test whether the detection threshold is a
    property of the model or an artifact of the one token/length tested."""
    valid_tokens = [t for t in INTRUDER_TOKENS if is_single_token(tokenizer, t)]
    skipped = [t for t in INTRUDER_TOKENS if t not in valid_tokens]
    if skipped:
        LOGGER.warning(f"Skipping intruder tokens not single-token for {cfg.model_name}: {skipped}")
    print(f"\nIntruder tokens used for variation sweep: {valid_tokens}")

    token_variation = {t: run_anomaly_sweep(cfg, pipe, tokenizer, intruder_token=t, base_n=10) for t in valid_tokens}
    length_variation = {n: run_anomaly_sweep(cfg, pipe, tokenizer, intruder_token="banana", base_n=n)
                         for n in BASE_NS if n != 10}

    thresholds_by_token = {t: r["detection_threshold"] for t, r in token_variation.items()}
    thresholds_by_length = {str(n): r["detection_threshold"] for n, r in length_variation.items()}
    thresholds_by_length["10"] = token_variation["banana"]["detection_threshold"]

    print(f"\nThreshold summary — {cfg.model_name}")
    print("  by intruder token (base_n=10)")
    for t, thr in thresholds_by_token.items():
        print(f"    {t:<10}  threshold={thr}")
    print("  by base length (intruder=banana)")
    for n, thr in sorted(thresholds_by_length.items(), key=lambda kv: int(kv[0])):
        print(f"    n={n:<8}  threshold={thr}")

    return {
        "token_variation" : token_variation,
        "length_variation": {str(k): v for k, v in length_variation.items()},
        "thresholds_by_token" : thresholds_by_token,
        "thresholds_by_length": thresholds_by_length,
    }


# ── Anomaly robustness follow-ups ──────────────────────────────────────────

LENGTH_SCALING_NS = [15, 20]


def run_length_scaling_sweep(cfg, pipe, tokenizer) -> dict:
    """Extends the base-length sweep past n=12 (banana intruder fixed) to test
    whether the tail-position detection bias is a fixed-window recency effect
    (hits stay at the same absolute positions) or scales proportionally with
    sequence length (hits track the last ~20-30% of the list)."""
    print(f"\nLength-scaling sweep (banana intruder) — {cfg.model_name}")
    results = {n: run_anomaly_sweep(cfg, pipe, tokenizer, intruder_token="banana", base_n=n)
               for n in LENGTH_SCALING_NS}

    print(f"\nTail-position summary — {cfg.model_name}")
    for n, r in results.items():
        hits = [pos for pos, v in r["position_results"].items() if v["detected"]]
        print(f"  n={n:<4}  hit_positions={hits}  last_position={n - 1}")

    return {str(k): v for k, v in results.items()}


TOKEN_PANEL = ["banana", "orange", "car", "chair", "the", "and", "xyz", "qzx", "seven", "five", "three", "nine"]


def run_token_panel_sweep(cfg, pipe, tokenizer, existing_results: dict | None = None) -> dict:
    """Runs the full position+quantity sweep (base_n=10) across a diverse panel
    of intruder tokens (food/object/function-word/nonsense/digit-word) to test
    whether detection is token-specific or a generic not-apple signal. Reuses
    tokens already computed by run_anomaly_variation instead of re-running them."""
    existing_results = existing_results or {}
    valid_tokens = [t for t in TOKEN_PANEL if is_single_token(tokenizer, t)]
    skipped = [t for t in TOKEN_PANEL if t not in valid_tokens]
    if skipped:
        LOGGER.warning(f"Skipping panel tokens not single-token for {cfg.model_name}: {skipped}")

    new_tokens = [t for t in valid_tokens if t not in existing_results]
    print(f"\nToken panel sweep, base_n=10 — {cfg.model_name}")
    print(f"Panel tokens: {valid_tokens}  "
          f"(reusing {len(valid_tokens) - len(new_tokens)} from the variation sweep, running {len(new_tokens)} new)")

    panel_results = {t: existing_results[t] for t in valid_tokens if t in existing_results}
    panel_results.update({t: run_anomaly_sweep(cfg, pipe, tokenizer, intruder_token=t, base_n=10) for t in new_tokens})

    thresholds = {t: panel_results[t]["detection_threshold"] for t in valid_tokens}
    position_signatures = {
        t: tuple(sorted(int(pos) for pos, v in panel_results[t]["position_results"].items() if v["detected"]))
        for t in valid_tokens
    }
    distinct_signatures = set(position_signatures.values())

    print(f"\nThreshold + position-signature summary — {cfg.model_name}")
    for t in valid_tokens:
        print(f"  {t:<10}  threshold={thresholds[t]}  hit_positions={position_signatures[t]}")
    print(f"  distinct position-hit signatures: {len(distinct_signatures)} across {len(valid_tokens)} tokens")

    return {
        "panel_results": panel_results,
        "thresholds_by_token": thresholds,
        "position_signatures": {t: list(sig) for t, sig in position_signatures.items()},
        "n_distinct_signatures": len(distinct_signatures),
    }


# ── Category attention probe (grounds the token-panel category split) ──────
# 3B splits the token panel cleanly by category: function/digit-words (the,
# and, seven, five) are detected almost immediately, nouns/nonsense (banana,
# orange, car, chair, xyz) need most of the list replaced. This probe tests
# whether that split shows up as a difference in how much the last token
# attends to the intruder, at a single fixed condition (intruder at word
# index 4, n=10) so every token is compared under identical conditions.

EASY_CATEGORY = ["the", "and", "seven", "five", "three", "nine"]
HARD_CATEGORY = ["banana", "orange", "car", "chair", "xyz"]


def find_intruder_position(tokens: list[str], intruder_token: str) -> int | None:
    """Locate the intruder's position within the word-list span (after the last
    colon, up to the closing period) — restricted to that span so a word like
    "the" that also appears in the instruction text isn't matched there instead."""
    colon_idx = None
    for i, t in enumerate(tokens):
        if t.strip() == ":":
            colon_idx = i
    if colon_idx is None:
        return None

    valid = {"apple", intruder_token}
    positions = []
    for i in range(colon_idx + 1, len(tokens)):
        if tokens[i].strip() in valid:
            positions.append(i)
        elif "." in tokens[i] and positions:
            break
    return positions[4] if len(positions) > 4 else None


def run_category_attention_probe(cfg, tokenizer, model_eager) -> dict:
    print(f"\nCategory attention probe (intruder at word index 4, n=10) — {cfg.model_name}")

    results = {}
    for token in TOKEN_PANEL:
        if not is_single_token(tokenizer, token):
            LOGGER.warning(f"Skipping {token!r} - not single-token for {cfg.model_name}")
            continue

        words = ["apple"] * 10
        words[4] = token
        prompt = (
            'Count the number of times "apple" appears in this list: '
            + " ".join(words)
            + ". Respond only with the integer, nothing else."
        )
        inputs = make_inputs_eager(prompt, tokenizer, model_eager)
        input_ids = inputs["input_ids"][0]
        tokens_decoded = [tokenizer.decode([t]) for t in input_ids]

        intruder_pos = find_intruder_position(tokens_decoded, token)
        if intruder_pos is None:
            LOGGER.warning(f"Could not locate {token!r} in the word-list span for {cfg.model_name}")
            continue

        with torch.no_grad():
            out = model_eager(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
            )
        attentions = torch.stack(out.attentions).squeeze(1).cpu().float()  # (layers, heads, seq, seq)
        attn_to_intruder = attentions[:, :, -1, intruder_pos]  # (layers, heads), from last token

        mean_per_layer = attn_to_intruder.mean(dim=1)
        overall_mean = float(attn_to_intruder.mean())
        overall_max = float(attn_to_intruder.max())
        peak_layer = int(mean_per_layer.argmax().item())

        print(f"  {token:<8}  overall_mean_attn={overall_mean:.4f}  overall_max_attn={overall_max:.4f}  peak_layer=L{peak_layer}")

        results[token] = {
            "intruder_pos": intruder_pos,
            "mean_attn_per_layer": [round(x, 4) for x in mean_per_layer.tolist()],
            "overall_mean_attn": round(overall_mean, 4),
            "overall_max_attn": round(overall_max, 4),
            "peak_layer": peak_layer,
        }

    easy_means = [results[t]["overall_mean_attn"] for t in EASY_CATEGORY if t in results]
    hard_means = [results[t]["overall_mean_attn"] for t in HARD_CATEGORY if t in results]
    easy_avg = round(sum(easy_means) / len(easy_means), 4) if easy_means else None
    hard_avg = round(sum(hard_means) / len(hard_means), 4) if hard_means else None

    print(f"\n  easy category (function/digit-words) avg attn: {easy_avg}")
    print(f"  hard category (nouns/nonsense) avg attn:        {hard_avg}")

    return {
        "per_token": results,
        "easy_category": EASY_CATEGORY,
        "hard_category": HARD_CATEGORY,
        "easy_avg_attn": easy_avg,
        "hard_avg_attn": hard_avg,
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

        # lowest priority / conditional: paraphrase decomposition at writer sites
        if cfg.writer_layers is not None:
            wrong_answer = p1_attractor if p1_attractor != str(p1_correct) else str(p1_correct - 1)
            paraphrase = paraphrase_decomposition_qwen(
                cfg, tokenizer, model_eager, cfg.writer_layers,
                p1_correct, int(wrong_answer),
            )
            output["paraphrase_decomposition"] = paraphrase
            paraphrase_path = output_path(cfg, "paraphrase_decomp_qwen.json")
            with open(paraphrase_path, "w") as f:
                json.dump(paraphrase, f, indent=2)
            print(f"Saved: {paraphrase_path}")
        else:
            LOGGER.warning("No writer_layers set for %s — skipping paraphrase decomposition.", cfg.key)
    else:
        # load generation model for anomaly sweep
        model_gen = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        pipe = pipeline("text-generation", model=model_gen,
                        tokenizer=tokenizer, device_map="auto")
        anomaly = run_anomaly_variation(cfg, pipe, tokenizer)
        output["anomaly_sweep"] = anomaly
        anomaly_path = output_path(cfg, "anomaly_sweep_qwen.json")
        with open(anomaly_path, "w") as f:
            json.dump(anomaly, f, indent=2)
        print(f"Saved: {anomaly_path}")

        length_scaling = run_length_scaling_sweep(cfg, pipe, tokenizer)
        token_panel = run_token_panel_sweep(cfg, pipe, tokenizer, existing_results=anomaly["token_variation"])
        robustness = {"length_scaling": length_scaling, "token_panel": token_panel}
        output["anomaly_robustness"] = robustness
        robustness_path = output_path(cfg, "anomaly_robustness_qwen.json")
        with open(robustness_path, "w") as f:
            json.dump(robustness, f, indent=2)
        print(f"Saved: {robustness_path}")

        # grounds the token-panel category split (function/digit-words vs nouns/nonsense)
        # in attention, using the eager model already loaded above
        category_attention = run_category_attention_probe(cfg, tokenizer, model_eager)
        output["category_attention"] = category_attention
        category_path = output_path(cfg, "category_attention_qwen.json")
        with open(category_path, "w") as f:
            json.dump(category_attention, f, indent=2)
        print(f"Saved: {category_path}")

    save_path = output_path(cfg, "mechanistic_qwen.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

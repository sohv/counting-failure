# For each interleaved-noise variant (space/comma baselines, newline/pipe x
# hint/no-hint), tests whether the count probe fit on ordinary space-separated
# prompts (n=3..13, matching mechanistic.py's own probe range) still decodes
# n=10 from the variant's hidden state (generalization, not cross-validation),
# and runs MLP/attention decomposition at the fixed writer layer (recomputed
# once from mechanistic_qwen.json via the same persistence backward-scan used
# in robustness.py, never re-detected per condition). "10" is multi-token for
# this tokenizer, so the graded logit-difference uses "1" (its leading digit,
# the same proxy mechanistic_qwen.json's own logit lens already reports as the
# correct-condition top digit) against the wrong digit "8", instead of "10"
# itself. Depends on behavioral.json and mechanistic_qwen.json.
# uv run -m src.experiments.qwen.interleaved_probe --model qwen-1.5b

import argparse
import json
import logging

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.config import add_model_arg, get_config, load_results, output_path
from src.common.prompts import INTERLEAVED_VARIANTS

LOGGER = logging.getLogger(__name__)

CORRECT_PROXY = "1"  # leading digit of "10", which tokenizes to multiple tokens here


def make_inputs_eager(prompt_text: str, tokenizer, model_eager):
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model_eager.device)


def get_hidden_states_last_token(prompt_text: str, tokenizer, model_eager) -> np.ndarray:
    inputs = make_inputs_eager(prompt_text, tokenizer, model_eager)
    with torch.no_grad():
        out = model_eager(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
    return torch.stack(out.hidden_states)[:, 0, -1, :].cpu().float().numpy()


def get_top_digit(logits_1d, tokenizer) -> str:
    candidates = {}
    for n in range(1, 21):
        ids = tokenizer.encode(str(n), add_special_tokens=False)
        if len(ids) == 1:
            candidates[str(n)] = logits_1d[ids[0]].item()
    return max(candidates, key=candidates.get) if candidates else "?"


def project_hidden(h, tokenizer, model_eager):
    with torch.no_grad():
        normed = model_eager.model.norm(h.unsqueeze(0).unsqueeze(0)).squeeze()
        logits = model_eager.lm_head.weight @ normed
    return logits


def single_token_logit_diff(h, tokenizer, model_eager, correct_token: str, wrong_token: str) -> float:
    logits = project_hidden(h, tokenizer, model_eager)
    correct_id = tokenizer.encode(correct_token, add_special_tokens=False)[0]
    wrong_id = tokenizer.encode(wrong_token, add_special_tokens=False)[0]
    return round((logits[wrong_id] - logits[correct_id]).item(), 4)


def find_writer(top_digits: list[str]) -> int | None:
    attractor = top_digits[-1]
    writer = None
    for i in range(len(top_digits) - 2, -1, -1):
        if top_digits[i] != attractor:
            writer = i + 1
            break
    return writer


def fit_count_probe(tokenizer, model_eager) -> tuple[list, list]:
    ns = list(range(3, 14))
    prompts = [
        'Count the number of times "apple" appears in this list: '
        + " ".join(["apple"] * n) + ". Respond only with the integer, nothing else."
        for n in ns
    ]
    labels = np.array(ns, dtype=float)
    hidden = np.stack([get_hidden_states_last_token(p, tokenizer, model_eager) for p in prompts])

    scalers, models = [], []
    for layer_idx in range(hidden.shape[1]):
        X = hidden[:, layer_idx, :]
        scaler = StandardScaler().fit(X)
        clf = Ridge(alpha=1.0).fit(scaler.transform(X), labels)
        scalers.append(scaler)
        models.append(clf)
    return scalers, models


def probe_generalization(tokenizer, model_eager, scalers, models) -> dict:
    print("\nCount-probe generalization to interleaved-noise variants (expected n=10)")
    print(f"  {'Variant':18s}  {'final-layer pred':>16}  {'final-layer |err|':>18}")

    results = {}
    for name, entry in INTERLEAVED_VARIANTS.items():
        h = get_hidden_states_last_token(entry["text"], tokenizer, model_eager)
        preds = []
        for layer_idx, (scaler, clf) in enumerate(zip(scalers, models)):
            pred = float(clf.predict(scaler.transform(h[layer_idx:layer_idx + 1]))[0])
            preds.append(round(pred, 4))
        final_pred = preds[-1]
        print(f"  {name:18s}  {final_pred:>16.4f}  {abs(final_pred - 10):>18.4f}")
        results[name] = {"predictions_per_layer": preds, "final_layer_prediction": final_pred,
                          "final_layer_abs_error": round(abs(final_pred - 10), 4)}
    return results


def decomposition_at_writer(tokenizer, model_eager, writer_layer: int, wrong_answer: str) -> dict:
    print(f"\nMLP/attention decomposition at fixed writer layer L{writer_layer} "
          f"(wrong='{wrong_answer}', correct proxy='{CORRECT_PROXY}')")
    print(f"  {'Variant':18s}  {'diff_before':>12}  {'diff_post_mlp':>14}  {'mlp_contrib':>12}  before->post_attn->post_mlp")

    results = {}
    layer = model_eager.model.layers[writer_layer - 1]
    for name, entry in INTERLEAVED_VARIANTS.items():
        cache = {}

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
        inputs = make_inputs_eager(entry["text"], tokenizer, model_eager)
        with torch.no_grad():
            model_eager(**inputs)
        h1.remove(); h2.remove(); h3.remove()

        diffs = {k: single_token_logit_diff(h, tokenizer, model_eager, CORRECT_PROXY, wrong_answer) for k, h in cache.items()}
        digits = {k: get_top_digit(project_hidden(h, tokenizer, model_eager), tokenizer) for k, h in cache.items()}

        diff_before = diffs.get("h_before", float("nan"))
        diff_post_attn = diffs.get("h_post_attn", float("nan"))
        diff_post_mlp = diffs.get("h_post_layer", float("nan"))
        attn_contrib = round(diff_post_attn - diff_before, 4)
        mlp_contrib = round(diff_post_mlp - diff_post_attn, 4)

        print(f"  {name:18s}  {diff_before:>12.4f}  {diff_post_mlp:>14.4f}  {mlp_contrib:>12.4f}  "
              f"{digits.get('h_before','?')}->{digits.get('h_post_attn','?')}->{digits.get('h_post_layer','?')}")

        results[name] = {
            "diff_before": diff_before, "diff_post_attn": diff_post_attn, "diff_post_mlp": diff_post_mlp,
            "attn_contribution": attn_contrib, "mlp_contribution": mlp_contrib,
            "top_digit_before": digits.get("h_before", "?"),
            "top_digit_post_attn": digits.get("h_post_attn", "?"),
            "top_digit_post_mlp": digits.get("h_post_layer", "?"),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    behavioral = load_results(cfg, "behavioral")
    wrong_answer = str(behavioral["phases"]["phase1_baseline"][0]["predicted"])
    mechanistic = load_results(cfg, "mechanistic_qwen")
    writer_layer = find_writer([e["top_digit"] for e in mechanistic["logit_lens"]["phase1_baseline"]])
    print(f"Model: {cfg.model_name}  wrong_answer={wrong_answer}  writer_layer=L{writer_layer}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model_eager = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager",
    )
    model_eager.eval()

    scalers, models = fit_count_probe(tokenizer, model_eager)
    generalization = probe_generalization(tokenizer, model_eager, scalers, models)
    decomposition = decomposition_at_writer(tokenizer, model_eager, writer_layer, wrong_answer)

    output = {
        "model": cfg.model_name,
        "wrong_answer": wrong_answer,
        "writer_layer": writer_layer,
        "correct_proxy": CORRECT_PROXY,
        "probe_generalization": generalization,
        "decomposition_at_writer": decomposition,
    }

    save_path = output_path(cfg, "interleaved_mechanistic_qwen.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

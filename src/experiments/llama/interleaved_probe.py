# tests whether the count probe and mlp/attention decomposition generalize to interleaved-noise variants.
# uv run -m src.experiments.llama.interleaved_probe --model llama-1b

import argparse
import json
import logging

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.common.config import add_model_arg, get_config, load_results, output_path
from src.common.prompts import INTERLEAVED_VARIANTS, make_prompt_repeated
from src.common.utils import (
    decompose_layer,
    get_hidden_states_last_token,
    get_writer_logit_diff,
    load_eager_model,
    remove_all_hooks,
)

LOGGER = logging.getLogger(__name__)


def fit_count_probe(tokenizer, model_eager) -> tuple[list, np.ndarray]:
    """Fit one Ridge probe per layer on all n=3..15 space-separated prompts (no
    held-out split - the interleaved variants are the out-of-distribution test
    points, so training uses the full in-distribution set)."""
    prompts = [make_prompt_repeated(n) for n in range(3, 16)]
    labels = np.array(range(3, 16), dtype=float)
    hidden = np.stack([get_hidden_states_last_token(tokenizer, model_eager, p) for p in prompts])

    n_layers = hidden.shape[1]
    scalers, models = [], []
    for layer_idx in range(n_layers):
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
        h = get_hidden_states_last_token(tokenizer, model_eager, entry["text"])
        preds = []
        for layer_idx, (scaler, clf) in enumerate(zip(scalers, models)):
            pred = float(clf.predict(scaler.transform(h[layer_idx:layer_idx + 1]))[0])
            preds.append(round(pred, 4))
        final_pred = preds[-1]
        print(f"  {name:18s}  {final_pred:>16.4f}  {abs(final_pred - 10):>18.4f}")
        results[name] = {"predictions_per_layer": preds, "final_layer_prediction": final_pred,
                          "final_layer_abs_error": round(abs(final_pred - 10), 4)}
    return results


def decomposition_at_lockin(tokenizer, model_eager, lockin_layer: int, wrong_answer: int) -> dict:
    print(f"\nMLP/attention decomposition at fixed lock-in layer L{lockin_layer} (wrong={wrong_answer})")
    print(f"  {'Variant':18s}  {'diff_before':>12}  {'diff_post_mlp':>14}  {'mlp_contrib':>12}")

    results = {}
    for name, entry in INTERLEAVED_VARIANTS.items():
        remove_all_hooks(model_eager)
        r = decompose_layer(
            tokenizer, model_eager, entry["text"], lockin_layer,
            correct_answer=entry["expected"], wrong_answer=wrong_answer,
        )
        diff_before = r.get("h_before", {}).get("logit_diff", float("nan"))
        diff_post_attn = r.get("h_post_attn", {}).get("logit_diff", float("nan"))
        diff_post_mlp = r.get("h_post_layer", {}).get("logit_diff", float("nan"))
        writer_info = get_writer_logit_diff(diff_before, diff_post_attn, diff_post_mlp)
        print(f"  {name:18s}  {diff_before:>12.4f}  {diff_post_mlp:>14.4f}  {writer_info['mlp_contribution']:>12.4f}")
        results[name] = {
            "diff_before": diff_before, "diff_post_attn": diff_post_attn, "diff_post_mlp": diff_post_mlp,
            "writer": writer_info,
            "top_digit_post_mlp": r.get("h_post_layer", {}).get("top_digit", "?"),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    behavioral = load_results(cfg, "behavioral")
    wrong_answer = int(behavioral["attractor"])
    logit_lens = load_results(cfg, "logit_lens")
    lockin_layer = logit_lens["summary"]["lockin_layer"]
    print(f"Model: {cfg.model_name}  wrong_answer={wrong_answer}  lockin_layer=L{lockin_layer}")

    tokenizer, model_eager = load_eager_model(cfg.model_name)

    scalers, models = fit_count_probe(tokenizer, model_eager)
    generalization = probe_generalization(tokenizer, model_eager, scalers, models)
    decomposition = decomposition_at_lockin(tokenizer, model_eager, lockin_layer, wrong_answer)

    output = {
        "model": cfg.model_name,
        "wrong_answer": wrong_answer,
        "lockin_layer": lockin_layer,
        "probe_generalization": generalization,
        "decomposition_at_lockin": decomposition,
    }

    save_path = output_path(cfg, "interleaved_mechanistic.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

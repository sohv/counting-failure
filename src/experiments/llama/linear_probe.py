"""
Ridge regression probe per layer on last-token residual stream to test whether
the correct count is linearly decodable, with leave-one-out cross-validation.
Separate probes for repeated vs unique token conditions.

Usage:
    uv run -m src.experiments.llama.linear_probe --model llama-1b
"""

import argparse
import json
import logging

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from src.common.config import add_model_arg, get_config, output_path
from src.common.prompts import make_prompt_repeated, make_prompt_unique
from src.common.utils import get_hidden_states_last_token, load_eager_model

LOGGER = logging.getLogger(__name__)


def run_loo_probe(hidden_array: np.ndarray, labels: np.ndarray) -> tuple[list[float], list[float]]:
    loo = LeaveOneOut()
    n_layers = hidden_array.shape[1]
    maes, r2s = [], []
    for layer_idx in range(n_layers):
        X = StandardScaler().fit_transform(hidden_array[:, layer_idx, :])
        preds = np.zeros(len(labels))
        for train_idx, test_idx in loo.split(X):
            clf = Ridge(alpha=1.0)
            clf.fit(X[train_idx], labels[train_idx])
            preds[test_idx] = clf.predict(X[test_idx])
        maes.append(round(mean_absolute_error(labels, preds), 4))
        r2s.append(round(r2_score(labels, preds), 4))
    return maes, r2s


def probe_count_encoding(cfg, tokenizer, model_eager) -> dict:
    probe_prompts, probe_labels = [], []
    for n in range(3, 16):
        probe_prompts.append(make_prompt_repeated(n))
        probe_labels.append(n)

    print(f"Probe dataset: {len(probe_prompts)} prompts, counts {probe_labels}")

    all_hidden = np.stack([
        get_hidden_states_last_token(tokenizer, model_eager, p) for p in probe_prompts
    ])
    labels = np.array(probe_labels, dtype=float)

    print(f"\nLayer-wise linear probe (Ridge, LOO cross-val)")
    print(f"  {'Layer':>6}  {'LOO MAE':>10}  {'LOO R2':>10}")

    layer_maes, layer_r2s = run_loo_probe(all_hidden, labels)
    for layer_idx, (mae, r2) in enumerate(zip(layer_maes, layer_r2s)):
        label = "embed" if layer_idx == 0 else f"L{layer_idx:02d}"
        print(f"  {label:>6}  {mae:>10.4f}  {r2:>10.4f}")

    return {"labels": probe_labels, "layer_maes": layer_maes, "layer_r2s": layer_r2s}


def probe_repeated_vs_unique(cfg, tokenizer, model_eager) -> dict:
    repeated_prompts, unique_prompts, probe_ns = [], [], []
    for n in range(3, 14):
        repeated_prompts.append(make_prompt_repeated(n))
        unique_prompts.append(make_prompt_unique(n))
        probe_ns.append(n)

    labels = np.array(probe_ns, dtype=float)

    print("\nCollecting activations for repeated and unique token probes")
    hidden_repeated = np.stack([
        get_hidden_states_last_token(tokenizer, model_eager, p) for p in repeated_prompts
    ])
    hidden_unique = np.stack([
        get_hidden_states_last_token(tokenizer, model_eager, p) for p in unique_prompts
    ])

    maes_rep, r2s_rep = run_loo_probe(hidden_repeated, labels)
    maes_uniq, r2s_uniq = run_loo_probe(hidden_unique, labels)

    print(f"\n{'Layer':>6}  {'MAE(repeat)':>12}  {'R2(repeat)':>11}  {'MAE(unique)':>12}  {'R2(unique)':>11}")
    for i, (mr, rr, mu, ru) in enumerate(zip(maes_rep, r2s_rep, maes_uniq, r2s_uniq)):
        label = "embed" if i == 0 else f"L{i:02d}"
        print(f"{label:>6}  {mr:>12.4f}  {rr:>11.4f}  {mu:>12.4f}  {ru:>11.4f}")

    return {
        "ns": probe_ns,
        "repeated": {"maes": maes_rep, "r2s": r2s_rep},
        "unique": {"maes": maes_uniq, "r2s": r2s_uniq},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    tokenizer, model_eager = load_eager_model(cfg.model_name)

    count_probe = probe_count_encoding(cfg, tokenizer, model_eager)
    comparison = probe_repeated_vs_unique(cfg, tokenizer, model_eager)

    output = {
        "model": cfg.model_name,
        "count_probe": count_probe,
        "repeated_vs_unique": comparison,
    }

    save_path = output_path(cfg, "linear_probe.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

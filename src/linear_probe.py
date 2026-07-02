"""
Odometer Pilot — Linear Probe.

Trains a per-layer Ridge regression (leave-one-out) on last-token residual
stream activations to see whether the correct count is linearly decodable
anywhere in the network, then repeats the probe separately for repeated vs.
unique token lists at matched lengths to see whether token identity affects
the *representation* (as opposed to the final output).

Usage:
    python linear_probe.py --model 1b
    python linear_probe.py --model 3b
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

import _paths  # noqa: F401  (adds config/ and data/ to sys.path)
from config import add_model_arg, get_config, output_path
from prompts import UNIQUE_VOCAB, make_prompt_repeated, make_prompt_unique
from utils import get_hidden_states_last_token, load_eager_model


def run_loo_probe(hidden_array: np.ndarray, labels: np.ndarray):
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
        maes.append(mean_absolute_error(labels, preds))
        r2s.append(r2_score(labels, preds))
    return maes, r2s


def probe_count_encoding(cfg, tokenizer, model_eager):
    probe_prompts, probe_labels = [], []
    for n in range(3, 16):
        probe_prompts.append(make_prompt_repeated(n))
        probe_labels.append(n)

    print(f"Probe dataset: {len(probe_prompts)} prompts, counts {probe_labels}")

    all_hidden = np.stack([
        get_hidden_states_last_token(tokenizer, model_eager, p) for p in probe_prompts
    ])
    labels = np.array(probe_labels, dtype=float)

    print("\nLayer-wise linear probe (Ridge, LOO cross-val):")
    print(f"  {'Layer':>6}  {'LOO MAE':>10}  {'LOO R²':>10}")
    print("  " + "-" * 32)

    layer_maes, layer_r2s = run_loo_probe(all_hidden, labels)
    for layer_idx, (mae, r2) in enumerate(zip(layer_maes, layer_r2s)):
        label = "embed" if layer_idx == 0 else f"L{layer_idx:02d}"
        print(f"  {label:>6}  {mae:>10.4f}  {r2:>10.4f}")

    save_path = output_path(cfg, "probe_results.json")
    with open(save_path, "w") as f:
        json.dump({"labels": probe_labels, "layer_maes": layer_maes, "layer_r2s": layer_r2s}, f, indent=2)
    print(f"\nSaved: {save_path}")


def probe_repeated_vs_unique(cfg, tokenizer, model_eager):
    repeated_prompts, unique_prompts, probe_ns = [], [], []
    for n in range(3, 14):
        repeated_prompts.append(make_prompt_repeated(n))
        unique_prompts.append(make_prompt_unique(n))
        probe_ns.append(n)

    labels = np.array(probe_ns, dtype=float)

    print("Collecting activations for repeated and unique token probes...")
    hidden_repeated = np.stack([
        get_hidden_states_last_token(tokenizer, model_eager, p) for p in repeated_prompts
    ])
    hidden_unique = np.stack([
        get_hidden_states_last_token(tokenizer, model_eager, p) for p in unique_prompts
    ])

    print("Running probes...")
    maes_rep, r2s_rep = run_loo_probe(hidden_repeated, labels)
    maes_uniq, r2s_uniq = run_loo_probe(hidden_unique, labels)

    print(f"\n{'Layer':>6}  {'MAE(repeat)':>12}  {'R²(repeat)':>11}  "
          f"{'MAE(unique)':>12}  {'R²(unique)':>11}  {'ΔMAE':>8}")
    print("-" * 70)
    for i, (mr, rr, mu, ru) in enumerate(zip(maes_rep, r2s_rep, maes_uniq, r2s_uniq)):
        label = "embed" if i == 0 else f"L{i:02d}"
        print(f"{label:>6}  {mr:>12.4f}  {rr:>11.4f}  {mu:>12.4f}  {ru:>11.4f}  {mr - mu:>+8.4f}")

    save_path = output_path(cfg, "probe_repeated_vs_unique.json")
    with open(save_path, "w") as f:
        json.dump({
            "ns": probe_ns, "labels": probe_ns,
            "repeated": {"maes": maes_rep, "r2s": r2s_rep},
            "unique": {"maes": maes_uniq, "r2s": r2s_uniq},
        }, f, indent=2)
    print(f"\nSaved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    tokenizer, model_eager = load_eager_model(cfg.model_name)

    probe_count_encoding(cfg, tokenizer, model_eager)
    probe_repeated_vs_unique(cfg, tokenizer, model_eager)


if __name__ == "__main__":
    main()

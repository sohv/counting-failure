"""
Odometer Pilot — Attention Sink Hypothesis.

Loads the model with eager attention (required for output_attentions=True),
visualizes mean last-token attention per layer for P1 (repeated tokens) vs
P3 (unique tokens), then extracts full per-head attention stats and
re-analyzes them restricted to just the word-list token span.

Usage:
    python attention_sink.py --model 1b
    python attention_sink.py --model 3b
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401  (adds config/ and data/ to sys.path)
from config import add_model_arg, get_config, output_path
from prompts import PROMPTS
from utils import load_eager_model, make_inputs_eager


def get_attentions(tokenizer, model_eager, prompt_text: str):
    inputs = make_inputs_eager(tokenizer, model_eager, prompt_text)
    with torch.no_grad():
        out = model_eager(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
        )
    attentions = torch.stack(out.attentions).squeeze(1)  # (n_layers, n_heads, seq_len, seq_len)
    return attentions, inputs["input_ids"][0]


def plot_attention_last_token(tokenizer, attentions, input_ids, title, ax_row):
    tokens = [tokenizer.decode([t]) for t in input_ids]
    last_tok_attn = attentions[:, :, -1, :].mean(dim=1).cpu().float().numpy()

    for layer_idx, ax in enumerate(ax_row):
        ax.bar(range(len(tokens)), last_tok_attn[layer_idx], color="steelblue")
        ax.set_title(f"L{layer_idx}", fontsize=7)
        ax.set_xticks([])
        ax.set_ylim(0, 1)
    ax_row[0].set_ylabel(title, fontsize=8, rotation=0, labelpad=60, va="center")


def visualize_attention(cfg, tokenizer, model_eager):
    step = max(cfg.n_layers // 4, 1)
    layers_to_plot = list(range(0, cfg.n_layers, step))[:4]

    fig, axes = plt.subplots(2, len(layers_to_plot), figsize=(14, 5), sharey=True)
    fig.suptitle("Mean attention from last token per layer\n"
                 "(P1: repeated tokens vs P3: unique tokens)", fontsize=10)

    for row_idx, phase_key in enumerate(["phase1_baseline", "phase3_control"]):
        attentions, input_ids = get_attentions(tokenizer, model_eager, PROMPTS[phase_key]["text"])
        selected = attentions[layers_to_plot]
        plot_attention_last_token(tokenizer, selected, input_ids, phase_key, axes[row_idx])

    plt.tight_layout()
    save_path = output_path(cfg, "attention_p1_vs_p3.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def get_attention_stats(tokenizer, model_eager, prompt_text: str, phase_key: str) -> dict:
    inputs = make_inputs_eager(tokenizer, model_eager, prompt_text)
    input_ids = inputs["input_ids"][0]
    tokens = [tokenizer.decode([t]) for t in input_ids]
    seq_len = len(tokens)

    with torch.no_grad():
        out = model_eager(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
        )

    attentions = torch.stack(out.attentions).squeeze(1).cpu().float()
    n_layers, n_heads, _, _ = attentions.shape
    last_tok = attentions[:, :, -1, :]  # (n_layers, n_heads, seq_len)

    per_head = last_tok.numpy().tolist()
    mean_over_heads = last_tok.mean(dim=1).numpy().tolist()

    entropies = []
    for layer_idx in range(n_layers):
        attn = last_tok[layer_idx].mean(dim=0)
        attn = attn / (attn.sum() + 1e-9)
        entropy = -(attn * (attn + 1e-9).log()).sum().item()
        entropies.append(entropy)

    argmax_positions = last_tok.mean(dim=1).argmax(dim=-1).numpy().tolist()
    argmax_tokens = [tokens[p] for p in argmax_positions]

    return {
        "phase": phase_key,
        "prompt": prompt_text,
        "tokens": tokens,
        "seq_len": seq_len,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "per_head_attn": per_head,
        "mean_attn": mean_over_heads,
        "entropy_per_layer": entropies,
        "argmax_pos": argmax_positions,
        "argmax_token": argmax_tokens,
    }


def extract_and_save_attention_data(cfg, tokenizer, model_eager) -> dict:
    attention_data = {}
    for phase_key in ["phase1_baseline", "phase2_anomaly", "phase3_control"]:
        print(f"Extracting attention: {phase_key}...")
        stats = get_attention_stats(tokenizer, model_eager, PROMPTS[phase_key]["text"], phase_key)
        attention_data[phase_key] = stats
        print(f"  seq_len={stats['seq_len']}  layers={stats['n_layers']}  heads={stats['n_heads']}")
        print(f"  entropy by layer: {[f'{e:.3f}' for e in stats['entropy_per_layer']]}")
        print(f"  argmax token by layer: {stats['argmax_token']}\n")

    save_path = output_path(cfg, "attention_data.json")
    with open(save_path, "w") as f:
        json.dump(attention_data, f, indent=2)
    print(f"Saved to {save_path}")
    return attention_data


def find_word_list_span_by_marker(tokens, phase_key: str):
    """Locate the word-list span by finding the first content-word token and
    the period that ends the list. Simpler than the colon-anchored version
    below, but doesn't guard against the prompt/BOS containing stray
    lookalike tokens — kept as an independent cross-check."""
    markers = {
        "phase1_baseline": "apple",
        "phase2_anomaly": "apple",
        "phase3_control": "dog",
    }
    marker = markers[phase_key]
    start = None
    for i, t in enumerate(tokens):
        if t.strip() == marker:
            start = i
            break

    end = len(tokens) - 1
    for i in range(start, len(tokens)):
        if "." in tokens[i]:
            end = i
            break

    return start, end


def analyze_word_span_masked(cfg, attention_data: dict):
    """Word-span attention analysis with the prompt/BOS tokens masked out
    (marker-anchored span, cell 34 in the original notebooks) — an earlier,
    simpler cut at the same question the colon-anchored version below
    answers more robustly."""
    print("=" * 70)
    print("ATTENTION ANALYSIS — word-list tokens only (BOS/prompt masked out)")
    print("=" * 70)

    summary = {}
    for phase_key, data in attention_data.items():
        tokens = data["tokens"]
        n_layers = data["n_layers"]

        start, end = find_word_list_span_by_marker(tokens, phase_key)
        word_tokens = tokens[start:end]
        print(f"\n[{phase_key}]")
        print(f"  Word-list span : positions {start}-{end - 1}")
        print(f"  Word tokens    : {word_tokens}")

        per_head = np.array(data["per_head_attn"])
        word_attn = per_head[:, :, start:end]
        word_attn_norm = word_attn / (word_attn.sum(axis=-1, keepdims=True) + 1e-9)
        mean_word_attn = word_attn_norm.mean(axis=1)

        entropies = []
        for l in range(n_layers):
            a = mean_word_attn[l]
            a = a / (a.sum() + 1e-9)
            h = -(a * np.log(a + 1e-9)).sum()
            entropies.append(round(float(h), 4))

        argmax_word_pos = mean_word_attn.argmax(axis=-1).tolist()
        argmax_word_tok = [word_tokens[p] for p in argmax_word_pos]
        uniformity = (mean_word_attn.min(axis=-1) / (mean_word_attn.max(axis=-1) + 1e-9)).mean()

        print(f"  Entropy by layer (word span): {entropies}")
        print(f"  Argmax word token by layer  : {argmax_word_tok}")
        print(f"  Mean uniformity score       : {uniformity:.4f}  (1=uniform, 0=spiked)")

        summary[phase_key] = {
            "word_span": [start, end],
            "word_tokens": word_tokens,
            "entropy_per_layer": entropies,
            "argmax_word_token": argmax_word_tok,
            "mean_uniformity": float(uniformity),
            "mean_word_attn": mean_word_attn.tolist(),
        }

    save_path = output_path(cfg, "attention_word_span_analysis.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {save_path}")

    print("\n" + "=" * 70)
    print("SUMMARY — mean entropy and uniformity over word-list span")
    print("=" * 70)
    print(f"  {'Phase':<25} {'Mean entropy':>14}  {'Uniformity':>12}")
    print("  " + "-" * 54)
    for phase_key, s in summary.items():
        me = np.mean(s["entropy_per_layer"])
        print(f"  {phase_key:<25} {me:>14.4f}  {s['mean_uniformity']:>12.4f}")


def find_word_list_span(tokens, phase_key: str):
    """Locate word-list token positions after the last ':' up to the period."""
    content_words = {
        "phase1_baseline": {"apple"},
        "phase2_anomaly": {"apple", "banana"},
        "phase3_control": {"dog", "cat", "car", "red", "blue",
                           "green", "house", "tree", "book", "pen"},
    }
    valid = content_words[phase_key]

    colon_idx = None
    for i, t in enumerate(tokens):
        if t.strip() == ":":
            colon_idx = i

    word_positions = []
    for i in range(colon_idx + 1, len(tokens)):
        if tokens[i].strip() in valid:
            word_positions.append(i)
        elif "." in tokens[i] and word_positions:
            break

    return word_positions


def get_word_attn(per_head_np, word_positions):
    word_attn = per_head_np[:, :, word_positions]
    return word_attn / (word_attn.sum(axis=-1, keepdims=True) + 1e-9)


def analyze_word_span(cfg, attention_data: dict):
    print("=" * 70)
    print("ATTENTION ANALYSIS — word-list tokens only (FIXED span)")
    print("=" * 70)

    summary_fixed = {}
    for phase_key, data in attention_data.items():
        tokens = data["tokens"]
        n_layers = data["n_layers"]

        word_positions = find_word_list_span(tokens, phase_key)
        word_tokens = [tokens[p] for p in word_positions]

        print(f"\n[{phase_key}]")
        print(f"  Word positions : {word_positions}")
        print(f"  Word tokens    : {word_tokens}")
        print(f"  Count          : {len(word_positions)}  (expected {data['seq_len']})")

        per_head = np.array(data["per_head_attn"])
        word_attn_norm = get_word_attn(per_head, word_positions)
        mean_word_attn = word_attn_norm.mean(axis=1)

        entropies = []
        for l in range(n_layers):
            a = mean_word_attn[l]
            a = a / (a.sum() + 1e-9)
            h = -(a * np.log(a + 1e-9)).sum()
            entropies.append(round(float(h), 4))

        argmax_pos = mean_word_attn.argmax(axis=-1).tolist()
        argmax_tok = [word_tokens[p] for p in argmax_pos]
        uniformity = (mean_word_attn.min(axis=-1) / (mean_word_attn.max(axis=-1) + 1e-9)).mean()

        print(f"  Entropy by layer: {entropies}")
        print(f"  Argmax token    : {argmax_tok}")
        print(f"  Uniformity      : {uniformity:.4f}")

        summary_fixed[phase_key] = {
            "word_positions": word_positions,
            "word_tokens": word_tokens,
            "entropy_per_layer": entropies,
            "argmax_word_token": argmax_tok,
            "mean_uniformity": float(uniformity),
            "mean_word_attn": mean_word_attn.tolist(),
        }

    save_path = output_path(cfg, "attention_word_span_fixed.json")
    with open(save_path, "w") as f:
        json.dump(summary_fixed, f, indent=2)
    print(f"\nSaved: {save_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Phase':<25} {'Mean entropy':>14}  {'Uniformity':>12}")
    print("  " + "-" * 54)
    for phase_key, s in summary_fixed.items():
        me = np.mean(s["entropy_per_layer"])
        print(f"  {phase_key:<25} {me:>14.4f}  {s['mean_uniformity']:>12.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    tokenizer, model_eager = load_eager_model(cfg.model_name)

    visualize_attention(cfg, tokenizer, model_eager)
    attention_data = extract_and_save_attention_data(cfg, tokenizer, model_eager)
    analyze_word_span_masked(cfg, attention_data)
    analyze_word_span(cfg, attention_data)


if __name__ == "__main__":
    main()

# word-span attention analysis: entropy and uniformity over the word-list span for p1/p2/p3.
# uv run -m src.experiments.llama.attention --model llama-1b

import argparse
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.config import add_model_arg, get_config, output_path
from src.common.prompts import PROMPTS
from src.common.utils import load_eager_model, make_inputs_eager

LOGGER = logging.getLogger(__name__)


def get_attentions(tokenizer, model_eager, prompt_text: str):
    inputs = make_inputs_eager(tokenizer, model_eager, prompt_text)
    with torch.no_grad():
        out = model_eager(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
        )
    attentions = torch.stack(out.attentions).squeeze(1)
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
    fig.suptitle("Mean attention from last token per layer\n(P1: repeated vs P3: unique)", fontsize=10)

    for row_idx, phase_key in enumerate(["phase1_baseline", "phase3_control"]):
        attentions, input_ids = get_attentions(tokenizer, model_eager, PROMPTS[phase_key]["text"])
        selected = attentions[layers_to_plot]
        plot_attention_last_token(tokenizer, selected, input_ids, phase_key, axes[row_idx])

    plt.tight_layout()
    save_path = output_path(cfg, "attention.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {save_path}")


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
    last_tok = attentions[:, :, -1, :]

    per_head = last_tok.numpy().tolist()
    mean_over_heads = last_tok.mean(dim=1).numpy().tolist()

    entropies = []
    for layer_idx in range(n_layers):
        attn = last_tok[layer_idx].mean(dim=0)
        attn = attn / (attn.sum() + 1e-9)
        entropy = -(attn * (attn + 1e-9).log()).sum().item()
        entropies.append(round(entropy, 4))

    argmax_positions = last_tok.mean(dim=1).argmax(dim=-1).numpy().tolist()
    argmax_tokens = [tokens[p] for p in argmax_positions]

    return {
        "phase": phase_key,
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


def extract_attention_data(cfg, tokenizer, model_eager) -> dict:
    attention_data = {}
    for phase_key in ["phase1_baseline", "phase2_anomaly", "phase3_control"]:
        print(f"Extracting attention: {phase_key}")
        stats = get_attention_stats(tokenizer, model_eager, PROMPTS[phase_key]["text"], phase_key)
        attention_data[phase_key] = stats
        print(f"  seq_len={stats['seq_len']}  layers={stats['n_layers']}  heads={stats['n_heads']}")
    return attention_data


def find_word_list_span(tokens: list[str], phase_key: str) -> list[int]:
    """Locate word-list token positions after the last colon up to the period."""
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

    if colon_idx is None:
        LOGGER.warning(f"No colon found in tokens for {phase_key}")
        return []

    word_positions = []
    for i in range(colon_idx + 1, len(tokens)):
        if tokens[i].strip() in valid:
            word_positions.append(i)
        elif "." in tokens[i] and word_positions:
            break

    return word_positions


def analyze_word_span(attention_data: dict) -> dict:
    print("\nWord-span attention analysis")
    summary = {}
    for phase_key, data in attention_data.items():
        tokens = data["tokens"]
        n_layers = data["n_layers"]

        word_positions = find_word_list_span(tokens, phase_key)
        word_tokens = [tokens[p] for p in word_positions]

        per_head = np.array(data["per_head_attn"])
        word_attn = per_head[:, :, word_positions]
        word_attn_norm = word_attn / (word_attn.sum(axis=-1, keepdims=True) + 1e-9)
        mean_word_attn = word_attn_norm.mean(axis=1)

        entropies = []
        for layer in range(n_layers):
            a = mean_word_attn[layer]
            a = a / (a.sum() + 1e-9)
            h = -(a * np.log(a + 1e-9)).sum()
            entropies.append(round(float(h), 4))

        uniformity = float((mean_word_attn.min(axis=-1) / (mean_word_attn.max(axis=-1) + 1e-9)).mean())

        print(f"  {phase_key:<25} mean_entropy={np.mean(entropies):.4f}  uniformity={uniformity:.4f}")

        summary[phase_key] = {
            "word_positions": word_positions,
            "word_tokens": word_tokens,
            "entropy_per_layer": entropies,
            "mean_uniformity": round(uniformity, 4),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    tokenizer, model_eager = load_eager_model(cfg.model_name)

    visualize_attention(cfg, tokenizer, model_eager)
    attention_data = extract_attention_data(cfg, tokenizer, model_eager)
    word_span = analyze_word_span(attention_data)

    output = {
        "model": cfg.model_name,
        "attention_data": attention_data,
        "word_span_analysis": word_span,
    }

    save_path = output_path(cfg, "attention.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

from typing import List, Tuple
import torch
import matplotlib.pyplot as plt


def get_attentions(model, tokenizer, prompt_text: str):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True
        )
    
    attentions = outputs.attentions
    return attentions, inputs


def find_word_list_span(tokens: List[str]) -> Tuple[int, int]:
    start_idx = next((i for i, t in enumerate(tokens) if "apple" in t or "dog" in t), 0)
    end_idx = len(tokens) - 1
    return start_idx, end_idx


def get_word_attn(attention_matrix, word_positions: List[int], head_idx: int = None):
    if head_idx is not None:
        attn = attention_matrix[head_idx]
    else:
        attn = attention_matrix.mean(dim=0)
    
    word_attn = attn[:, word_positions]
    return word_attn


def get_attention_stats(attentions, tokens: List[str]):
    stats = {
        "n_layers": len(attentions),
        "n_heads": attentions[0].shape[1],
        "seq_length": attentions[0].shape[2],
    }
    return stats


def plot_attention_last_token(attention_matrix, tokens: List[str], title: str = ""):
    attn = attention_matrix[-1].mean(dim=0)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(attn.unsqueeze(0).cpu().numpy(), aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_title(title)
    plt.tight_layout()
    return fig

"""
Consolidated experiment implementations extracted from all notebooks.
Contains all 20+ functions organized by experiment type.
"""
import re
import random
from typing import Optional, List, Tuple, Dict
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from utils import extract_count, set_seed, remove_all_hooks, PhaseSummary

def run_single(
    phase_key: str,
    prompts: dict,
    pipe,
    config,
    model,
    tokenizer,
    seed: int = 0
) -> dict:
    """Run single phase evaluation with fixed seed."""
    entry = prompts[phase_key]
    set_seed(seed)
    
    raw = pipe(
        [{"role": "user", "content": entry["text"]}],
        max_new_tokens=config.max_new_tokens,
        do_sample=False,
        temperature=None,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        max_length=None,
    )[0]["generated_text"].strip()
    
    predicted = extract_count(raw)
    correct = (predicted == entry["expected"]) if predicted is not None else False
    
    return {
        "seed": seed,
        "raw": raw,
        "predicted": predicted,
        "expected": entry["expected"],
        "correct": correct,
    }

def run_phase(
    phase_key: str,
    prompts: dict,
    pipe,
    config,
    model,
    tokenizer,
    verbose: bool = True
) -> Tuple[list, PhaseSummary]:
    """Run full phase across N_RUNS seeds."""
    entry = prompts[phase_key]
    seeds = list(range(config.n_runs))
    
    if verbose:
        print(f"\n{chr(9472)*60}")
        print(f"  {phase_key.upper():<25} | {entry['description']}")
        print(f"{chr(9472)*60}")
    
    phase_results = []
    for seed in seeds:
        r = run_single(phase_key, prompts, pipe, config, model, tokenizer, seed)
        status = "✓" if r["correct"] else "✗"
        
        if verbose:
            raw_preview = repr(r['raw'][:50])
            print(f"  seed={seed:02d}  pred={str(r['predicted']):>4}  exp={r['expected']}  {status}  {raw_preview}")
        
        phase_results.append(r)
    
    summary = PhaseSummary(
        phase=phase_key,
        description=entry['description'],
        expected=entry['expected'],
        n_runs=config.n_runs,
    )
    for r in phase_results:
        summary.add(r)
    
    return phase_results, summary

def n_sweep_analysis(
    phase_key: str,
    prompts: dict,
    pipe,
    config,
    model,
    tokenizer,
    n_range: List[int] = None,
    unique_vocab: List[str] = None,
    n_trials_per_n: int = 3,
) -> dict:
    """Sweep count from n_min to n_max."""
    if n_range is None:
        n_range = range(5, 21)
    
    print(f"\nN-sweep for {phase_key} (n_range={min(n_range)}-{max(n_range)})...")
    
    results = {}
    for n in n_range:
        if unique_vocab:
            words = unique_vocab[:n]
            text = f"Count the number of words in this list: {' '.join(words)}. " \
                   "Respond only with the integer, nothing else."
        else:
            word = "apple"
            text = f'Count the number of times "{word}" appears in this list: ' \
                   f"{' '.join([word]*n)}. Respond only with the integer, nothing else."
        
        predictions = []
        for trial in range(n_trials_per_n):
            raw = pipe(
                [{"role": "user", "content": text}],
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False,
            )[0]["generated_text"].strip()
            
            pred = extract_count(raw)
            predictions.append(pred if pred is not None else -1)
        
        results[n] = predictions
        print(f"  n={n:2d}: {predictions}")
    
    return results

def get_attentions(model, tokenizer, prompt_text: str):
    """Extract attention weights from all layers."""
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
    """Find start and end indices of the word list."""
    start_idx = next((i for i, t in enumerate(tokens) if "apple" in t or "dog" in t), 0)
    end_idx = len(tokens) - 1
    return start_idx, end_idx

def get_word_attn(attention_matrix, word_positions: List[int], head_idx: int = None):
    """Get attention to word positions."""
    if head_idx is not None:
        attn = attention_matrix[head_idx]
    else:
        attn = attention_matrix.mean(dim=0)
    
    word_attn = attn[:, word_positions]
    return word_attn

def get_attention_stats(attentions, tokens: List[str]):
    """Compute attention statistics."""
    stats = {
        "n_layers": len(attentions),
        "n_heads": attentions[0].shape[1],
        "seq_length": attentions[0].shape[2],
    }
    return stats

def plot_attention_last_token(attention_matrix, tokens: List[str], title: str = ""):
    """Visualize attention from last token."""
    attn = attention_matrix[-1].mean(dim=0)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(attn.unsqueeze(0).cpu().numpy(), aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_title(title)
    plt.tight_layout()
    return fig

def get_hidden_states(model, tokenizer, prompt_text: str, layer_idx: int = -1):
    """Extract hidden states from a specific layer."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    hidden_states = outputs.hidden_states[layer_idx]
    return hidden_states

def get_hidden_states_from_text(model, tokenizer, texts: List[str], layer_idx: int = -1):
    """Get hidden states for multiple texts."""
    states = []
    for text in texts:
        hs = get_hidden_states(model, tokenizer, text, layer_idx)
        states.append(hs)
    return states

def run_loo_probe(
    model, tokenizer, prompts: dict, phase_keys: List[str] = None
) -> dict:
    """Leave-one-out linear probe for counting."""
    if phase_keys is None:
        phase_keys = ["phase1_baseline", "phase3_control"]
    
    results = {}
    
    for test_phase in phase_keys:
        train_phases = [p for p in phase_keys if p != test_phase]
        
        X_train, y_train = [], []
        for phase_key in train_phases:
            for seed in range(10):
                set_seed(seed)
                hs = get_hidden_states(
                    model, tokenizer, prompts[phase_key]["text"], layer_idx=-1
                )
                X_train.append(hs[0, -1, :].cpu().numpy())
                y_train.append(prompts[phase_key]["expected"])
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        X_test, y_test = [], []
        for seed in range(10):
            set_seed(seed)
            hs = get_hidden_states(
                model, tokenizer, prompts[test_phase]["text"], layer_idx=-1
            )
            X_test.append(hs[0, -1, :].cpu().numpy())
            y_test.append(prompts[test_phase]["expected"])
        
        accuracy = clf.score(X_test, y_test)
        results[test_phase] = {"accuracy": accuracy, "classifier": clf}
    
    return results

def logit_lens(
    model, tokenizer, prompt_text: str,
    target_token: int = None,
    max_layers: int = None
) -> dict:
    """Run logit lens: decode top logits at each layer."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    if target_token is None:
        target_token = inputs.input_ids.shape[1] - 1
    
    results = {}
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    hidden_states = outputs.hidden_states
    n_layers = len(hidden_states) if max_layers is None else max_layers
    
    for layer_idx in range(n_layers):
        hs = hidden_states[layer_idx]
        hs_target = hs[0, target_token, :].unsqueeze(0)
        
        with torch.no_grad():
            lm_head_logits = model.lm_head(hs_target)
        
        top_tokens = lm_head_logits.topk(5)[1][0]
        top_texts = [tokenizer.decode([t]) for t in top_tokens]
        
        results[layer_idx] = {
            "top_tokens": top_tokens.cpu().tolist(),
            "top_texts": top_texts,
            "logits": lm_head_logits[0].cpu().numpy(),
        }
    
    return results

def logit_lens_single(model, tokenizer, prompt_text: str, layer_idx: int) -> dict:
    """Logit lens for a single layer."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    hs = outputs.hidden_states[layer_idx][0, -1, :].unsqueeze(0)
    
    with torch.no_grad():
        logits = model.lm_head(hs)
    
    return {
        "logits": logits[0].cpu().numpy(),
        "layer": layer_idx,
    }

def decompose_layer(
    model, tokenizer, prompt_text: str, layer_idx: int
) -> dict:
    """Decompose layer into attention and MLP contributions."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    attn_outputs = []
    mlp_outputs = []
    
    def hook_post_attn(module, input, output):
        attn_outputs.append(output[0].detach())
    
    def hook_post_layer(module, input, output):
        mlp_outputs.append(output[0].detach())
    
    layer = model.model.layers[layer_idx]
    attn_hook = layer.self_attn.register_forward_hook(hook_post_attn)
    mlp_hook = layer.mlp.register_forward_hook(hook_post_layer)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    attn_hook.remove()
    mlp_hook.remove()
    
    return {
        "layer": layer_idx,
        "attn_output_shape": attn_outputs[0].shape if attn_outputs else None,
        "mlp_output_shape": mlp_outputs[0].shape if mlp_outputs else None,
    }

def save_h(model, layer_idx: int, prompt_text: str, tokenizer) -> torch.Tensor:
    """Save hidden state at layer for patching."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    return outputs.hidden_states[layer_idx].detach()

def patch_h(model, layer_idx: int, h_source: torch.Tensor, prompt_text: str, tokenizer):
    """Patch in source hidden state."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    original_forward = model.model.layers[layer_idx].forward
    
    def patched_forward(x, *args, **kwargs):
        result = original_forward(x, *args, **kwargs)
        result[0][:, :, :] = h_source
        return result
    
    model.model.layers[layer_idx].forward = patched_forward
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    model.model.layers[layer_idx].forward = original_forward
    return outputs

def get_top_digit(logits_tensor, tokenizer) -> Optional[int]:
    """Get top digit (0-9) from logits."""
    if logits_tensor is None:
        return None
    
    digit_ids = {}
    for i in range(10):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        if tokens:
            digit_ids[i] = tokens[0]
    
    digit_logits = {}
    for digit, token_id in digit_ids.items():
        if token_id < logits_tensor.shape[-1]:
            digit_logits[digit] = logits_tensor[token_id].item()
    
    return max(digit_logits, key=digit_logits.get) if digit_logits else None

def make_inputs(tokenizer, text: str) -> dict:
    """Create input dict for model."""
    return tokenizer(text, return_tensors="pt")

def make_inputs_eager(tokenizer, text: str) -> dict:
    """Create inputs for eager execution (debugging)."""
    return tokenizer(text, return_tensors="pt")

def make_prompt_repeated(n: int) -> str:
    """Create prompt with repeated words."""
    return f'Count: {" ".join(["apple"]*n)}. Answer: '

def make_prompt_unique(words: List[str]) -> str:
    """Create prompt with unique words."""
    return f'Count: {" ".join(words)}. Answer: '

def make_prompt_numbered(n: int) -> str:
    """Create prompt with numbered list."""
    return f'Count the numbers 1 to {n}: {", ".join(str(i) for i in range(1, n+1))}. Total: '

def zero_ablate_mlp(model, layer_idx: int, prompt_text: str, tokenizer):
    """Zero-ablate MLP at layer."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    def zero_mlp_hook(module, input, output):
        return (torch.zeros_like(output[0]),) + output[1:]
    
    layer = model.model.layers[layer_idx]
    hook = layer.mlp.register_forward_hook(zero_mlp_hook)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    hook.remove()
    return outputs

def inspect_hook(module, input, output):
    """Generic hook for inspection."""
    return output

def hook_pre_layer(module, input, stored_tensors: dict = None):
    """Hook before layer execution to capture inputs."""
    if stored_tensors is not None:
        stored_tensors['pre_input'] = input[0].detach() if isinstance(input, tuple) else input.detach()
    return None

def hook_post_layer(module, input, output, stored_tensors: dict = None):
    """Hook after layer execution to capture outputs."""
    if stored_tensors is not None:
        stored_tensors['post_output'] = output[0].detach() if isinstance(output, tuple) else output.detach()
    return output

def hook_post_attn(module, input, output, stored_tensors: dict = None):
    """Hook after attention layer."""
    if stored_tensors is not None:
        if isinstance(output, tuple):
            stored_tensors['attn_output'] = output[0].detach()
        else:
            stored_tensors['attn_output'] = output.detach()
    return output

def steer_hook(module, input, output, steering_vector: torch.Tensor = None, scale: float = 1.0):
    """Hook to apply steering vector to model activations."""
    if steering_vector is not None:
        if isinstance(output, tuple):
            output = (output[0] + scale * steering_vector.to(output[0].device),) + output[1:]
        else:
            output = output + scale * steering_vector.to(output.device)
    return output

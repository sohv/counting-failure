"""Shared helpers: model loading, generation, and mechanistic-interp plumbing."""

import random
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from huggingface_hub import login as hf_login, whoami as hf_whoami
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

warnings.filterwarnings("ignore")


def ensure_hf_login():
    """meta-llama/Llama-3.2-* are gated repos. Both source notebooks call
    huggingface_hub.login() up front; here we only prompt if there isn't
    already a cached token, so run_all.py doesn't ask 6 times in a row."""
    try:
        hf_whoami()
    except Exception:
        hf_login()


# ── Generation-side loading ────────────────────────────────────────────────
def load_generation_model(model_name: str):
    """Load tokenizer + causal LM + text-generation pipeline (matches the
    original notebooks' section 3)."""
    ensure_hf_login()
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    print("Model loaded successfully.")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return tokenizer, model, pipe


def load_eager_model(model_name: str, tokenizer=None):
    """Load a second copy of the model with eager attention so
    output_attentions=True / hooks work (matches the Attention Sink section)."""
    ensure_hf_login()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_eager = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model_eager.eval()
    model_eager.config.output_hidden_states = True
    print("Eager model loaded.")
    print(f"Attention impl: {model_eager.config._attn_implementation}")
    return tokenizer, model_eager


# ── Answer extraction ───────────────────────────────────────────────────────
def extract_count(raw_output: str) -> Optional[int]:
    """Grab the first standalone integer from the model's answer."""
    match = re.search(r"\b(\d+)\b", raw_output.strip())
    return int(match.group(1)) if match else None


def generate(pipe, tokenizer, prompt_text: str, max_new_tokens: int = 8,
             temperature: float = 0.0) -> str:
    return pipe(
        [{"role": "user", "content": prompt_text}],
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=(temperature if temperature > 0 else None),
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        max_length=None,
    )[0]["generated_text"].strip()


def run_single(pipe, tokenizer, phase_key: str, entry: dict, seed: int,
               max_new_tokens: int, temperature: float) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    raw = generate(pipe, tokenizer, entry["text"],
                   max_new_tokens=max_new_tokens, temperature=temperature)
    predicted = extract_count(raw)
    correct = (predicted == entry["expected"]) if predicted is not None else False

    return {
        "seed": seed,
        "raw": raw,
        "predicted": predicted,
        "expected": entry["expected"],
        "correct": correct,
    }


def get_known_attractor(pipe, tokenizer, prompts: dict, seed: int = 0) -> str:
    """Cheap single-seed baseline read, used by downstream scripts to
    discover 'what wrong digit does this model lock onto' without redoing
    the full 10-seed sweep from the behavioral experiment script."""
    r = run_single(pipe, tokenizer, "phase1_baseline", prompts["phase1_baseline"],
                   seed=seed, max_new_tokens=16, temperature=0.0)
    return str(r["predicted"])


@dataclass
class PhaseSummary:
    phase: str
    description: str
    expected: int
    n_runs: int
    predictions: list = field(default_factory=list)
    n_correct: int = 0

    def add(self, r):
        self.predictions.append(r["predicted"])
        if r["correct"]:
            self.n_correct += 1

    def accuracy(self) -> float:
        return self.n_correct / self.n_runs

    def dist(self) -> dict:
        return dict(Counter(str(p) for p in self.predictions))


# ── Mechanistic-interp plumbing (hooks) ────────────────────────────────────
def remove_all_hooks(model):
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()


def make_inputs_eager(tokenizer, model_eager, prompt_text: str):
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model_eager.device)


def _digit_candidates(tokenizer, logits_1d, max_digit: int = 20) -> dict:
    candidates = {}
    for n in range(1, max_digit):
        ids = tokenizer.encode(str(n), add_special_tokens=False)
        if len(ids) == 1:
            candidates[str(n)] = logits_1d[ids[0]].item()
    return candidates


def get_top_digit(tokenizer, logits_1d, max_digit: int = 20) -> str:
    candidates = _digit_candidates(tokenizer, logits_1d, max_digit)
    return max(candidates, key=candidates.get) if candidates else "?"


def get_top_digit_and_margin(tokenizer, logits_1d, max_digit: int = 20):
    """Top digit token plus the logit gap to the runner-up digit — a small
    margin means the model was on a knife's edge between two answers, and a
    top-1 pick there is fragile to numerical noise (bf16 rounding, GPU/library
    version) rather than a robust finding."""
    candidates = _digit_candidates(tokenizer, logits_1d, max_digit)
    if not candidates:
        return "?", None
    ranked = sorted(candidates.values(), reverse=True)
    top_digit = max(candidates, key=candidates.get)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else None
    return top_digit, (round(margin, 4) if margin is not None else None)


def logit_lens_single(tokenizer, model_eager, hidden_state):
    """Project a single residual-stream vector through the final layernorm
    and unembedding matrix; return (top_digit, top5_tokens, margin)."""
    norm = model_eager.model.norm
    unembed = model_eager.lm_head.weight
    with torch.no_grad():
        normed = norm(hidden_state.unsqueeze(0).unsqueeze(0)).squeeze()
        logits = unembed @ normed
    top_digit, margin = get_top_digit_and_margin(tokenizer, logits)
    top5 = [tokenizer.decode([i]) for i in logits.topk(5).indices]
    return top_digit, top5, margin


def decompose_layer(tokenizer, model_eager, prompt_text: str, layer_idx: int) -> dict:
    """Capture the residual stream entering a decoder layer, entering its
    MLP (i.e. after attention), and leaving the layer, then logit-lens each
    one. `layer_idx` is 1-indexed to match the notebooks' "L14" convention.
    """
    inputs = make_inputs_eager(tokenizer, model_eager, prompt_text)
    cache = {}
    layer = model_eager.model.layers[layer_idx - 1]

    def hook_pre_layer(module, input):
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

    h1 = layer.register_forward_pre_hook(hook_pre_layer)
    h2 = layer.post_attention_layernorm.register_forward_hook(hook_post_attn)
    h3 = layer.register_forward_hook(hook_post_layer)

    with torch.no_grad():
        model_eager(**inputs)

    h1.remove()
    h2.remove()
    h3.remove()

    if len(cache) < 3:
        print(f"  Warning L{layer_idx}: only captured {list(cache.keys())}")

    results = {}
    for name, h in cache.items():
        top_digit, top5, margin = logit_lens_single(tokenizer, model_eager, h)
        results[name] = {"top_digit": top_digit, "top5": top5, "margin": margin}
    return results


def get_writer(before, post_attn, post_mlp, target) -> str:
    """Determine which sublayer introduced or removed the target digit."""
    if before != target and post_attn == target:
        return "ATTENTION"
    elif post_attn != target and post_mlp == target:
        return "MLP"
    elif before == target and post_attn != target:
        return "ATTN ERASES"
    elif post_attn == target and post_mlp != target:
        return "MLP ERASES"
    elif before == target and post_attn == target and post_mlp == target:
        return "(stable)"
    else:
        return "-"


def get_hidden_states_last_token(tokenizer, model_eager, prompt_text: str):
    """Return (n_layers+1, hidden_dim) residual-stream activations at the
    last token, one row per layer (row 0 = embedding output)."""
    inputs = make_inputs_eager(tokenizer, model_eager, prompt_text)
    with torch.no_grad():
        out = model_eager(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
    hidden = torch.stack(out.hidden_states)  # (n_layers+1, 1, seq_len, hidden_dim)
    return hidden[:, 0, -1, :].cpu().float().numpy()

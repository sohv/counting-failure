"""Shared helpers: model loading, generation, and mechanistic-interp plumbing."""

import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from huggingface_hub import login as hf_login, whoami as hf_whoami
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

LOGGER = logging.getLogger(__name__)


def ensure_hf_login():
    try:
        hf_whoami()
    except Exception:
        hf_login()


# -- Model loading -----------------------------------------------------------

def load_generation_model(model_name: str):
    ensure_hf_login()
    print(f"Loading {model_name}")
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
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return tokenizer, model, pipe


def load_eager_model(model_name: str, tokenizer=None, dtype: torch.dtype = torch.bfloat16):
    ensure_hf_login()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_eager = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model_eager.eval()
    model_eager.config.output_hidden_states = True
    print(f"Eager model loaded (dtype={dtype})")
    return tokenizer, model_eager


# -- Answer extraction -------------------------------------------------------

def extract_count(raw_output: str) -> int | None:
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

    def add(self, r: dict):
        self.predictions.append(r["predicted"])
        if r["correct"]:
            self.n_correct += 1

    def accuracy(self) -> float:
        return self.n_correct / self.n_runs

    def dist(self) -> dict:
        return dict(Counter(str(p) for p in self.predictions))


# -- Hook plumbing -----------------------------------------------------------

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


# -- Digit logit extraction --------------------------------------------------

def single_token_digit_proxy(tokenizer, n: int) -> int:
    """Return n if str(n) is single-token for this tokenizer, else the integer
    value of its leading-digit token (e.g. Qwen splits "10" into ["1", "0"],
    so callers that need a single-token correct/wrong answer for logit_diff
    should use this instead of the raw multi-digit number)."""
    ids = tokenizer.encode(str(n), add_special_tokens=False)
    if len(ids) == 1:
        return n
    return int(tokenizer.decode([ids[0]]))


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


def get_top_digit_and_margin(tokenizer, logits_1d, max_digit: int = 20) -> tuple[str, float | None]:
    candidates = _digit_candidates(tokenizer, logits_1d, max_digit)
    if not candidates:
        return "?", None
    ranked = sorted(candidates.values(), reverse=True)
    top_digit = max(candidates, key=candidates.get)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else None
    return top_digit, (round(margin, 4) if margin is not None else None)


# -- Logit difference (wrong - correct) -------------------------------------

def logit_difference(tokenizer, logits_1d, correct_answer: int, wrong_answer: int) -> float:
    """Wrong-minus-correct logit difference. Positive means wrong answer dominates."""
    correct_ids = tokenizer.encode(str(correct_answer), add_special_tokens=False)
    wrong_ids = tokenizer.encode(str(wrong_answer), add_special_tokens=False)
    if len(correct_ids) != 1 or len(wrong_ids) != 1:
        LOGGER.warning(f"Multi-token answer: correct={correct_answer}, wrong={wrong_answer}")
        return float("nan")
    return round((logits_1d[wrong_ids[0]] - logits_1d[correct_ids[0]]).item(), 4)


# -- Logit lens --------------------------------------------------------------

def logit_lens_single(
    tokenizer,
    model_eager,
    hidden_state,
    correct_answer: int | None = None,
    wrong_answer: int | None = None,
) -> dict:
    """Project a residual-stream vector through final layernorm + unembedding.

    Returns dict with top_digit, top5, margin, and optionally logit_diff.
    """
    norm = model_eager.model.norm
    unembed = model_eager.lm_head.weight
    with torch.no_grad():
        normed = norm(hidden_state.unsqueeze(0).unsqueeze(0)).squeeze()
        logits = unembed @ normed
    top_digit, margin = get_top_digit_and_margin(tokenizer, logits)
    top5 = [tokenizer.decode([i]) for i in logits.topk(5).indices]

    result = {"top_digit": top_digit, "top5": top5, "margin": margin}
    if correct_answer is not None and wrong_answer is not None:
        result["logit_diff"] = logit_difference(tokenizer, logits, correct_answer, wrong_answer)
    return result


# -- Layer decomposition -----------------------------------------------------

def _get_post_attn_layernorm(layer):
    """Get the post-attention layernorm, compatible with Llama and Qwen."""
    if hasattr(layer, "post_attention_layernorm"):
        return layer.post_attention_layernorm
    for name, child in layer.named_children():
        if "post" in name.lower() and "norm" in name.lower():
            return child
    LOGGER.warning("Could not find post-attention layernorm, falling back to input_layernorm")
    return layer.input_layernorm


def decompose_layer(
    tokenizer,
    model_eager,
    prompt_text: str,
    layer_idx: int,
    correct_answer: int | None = None,
    wrong_answer: int | None = None,
) -> dict:
    """Capture residual stream entering a layer, after attention, and after MLP.

    layer_idx is 1-indexed to match 'L14' convention.
    Returns logit lens results at each of the three states.
    """
    inputs = make_inputs_eager(tokenizer, model_eager, prompt_text)
    cache = {}
    layer = model_eager.model.layers[layer_idx - 1]
    post_attn_norm = _get_post_attn_layernorm(layer)

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
    h2 = post_attn_norm.register_forward_hook(hook_post_attn)
    h3 = layer.register_forward_hook(hook_post_layer)

    with torch.no_grad():
        model_eager(**inputs)

    h1.remove()
    h2.remove()
    h3.remove()

    if len(cache) < 3:
        LOGGER.warning(f"L{layer_idx}: only captured {list(cache.keys())}")

    results = {}
    for name, h in cache.items():
        results[name] = logit_lens_single(
            tokenizer, model_eager, h,
            correct_answer=correct_answer, wrong_answer=wrong_answer,
        )
    return results


def get_writer_logit_diff(
    diff_before: float, diff_post_attn: float, diff_post_mlp: float,
) -> dict:
    """Attribute logit-difference movement to attention vs MLP."""
    return {
        "attn_contribution": round(diff_post_attn - diff_before, 4),
        "mlp_contribution": round(diff_post_mlp - diff_post_attn, 4),
        "total_change": round(diff_post_mlp - diff_before, 4),
    }


def get_writer_argmax(before: str, post_attn: str, post_mlp: str, target: str) -> str:
    """Determine which sublayer introduced or removed the target digit (argmax-based)."""
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


# -- Ablation and patching (multi-layer) -------------------------------------
# Each takes a list of 1-indexed layer numbers so a single site reproduces the
# original single-layer behavior; a multi-site list ablates/patches all of
# them simultaneously (joint intervention).

def zero_ablate_mlp(
    tokenizer, model_eager, prompt: str, layer_indices: list[int],
    correct: int, wrong: int,
) -> dict:
    remove_all_hooks(model_eager)

    def zero_hook(module, input, output):
        return torch.zeros_like(output)

    handles = [model_eager.model.layers[i - 1].mlp.register_forward_hook(zero_hook) for i in layer_indices]
    inputs = make_inputs_eager(tokenizer, model_eager, prompt)
    with torch.no_grad():
        out = model_eager(**inputs)
    for h in handles:
        h.remove()

    logits = out.logits[0, -1, :]
    return {
        "top_digit": get_top_digit(tokenizer, logits),
        "logit_diff": logit_difference(tokenizer, logits, correct, wrong),
        "top5": [tokenizer.decode([i]) for i in logits.topk(5).indices],
    }


def mean_ablate_mlp(
    tokenizer, model_eager, target_prompt: str,
    reference_prompts: list[str], layer_indices: list[int],
    correct: int, wrong: int,
) -> dict:
    """Replace writer MLP output(s) with the mean MLP output from reference prompts."""
    means = {}
    for layer_idx in layer_indices:
        outs = []
        for ref_prompt in reference_prompts:
            remove_all_hooks(model_eager)
            cache = {}

            def capture(module, input, output, _cache=cache):
                _cache["out"] = output.detach().clone() if isinstance(output, torch.Tensor) else output[0].detach().clone()

            handle = model_eager.model.layers[layer_idx - 1].mlp.register_forward_hook(capture)
            with torch.no_grad():
                model_eager(**make_inputs_eager(tokenizer, model_eager, ref_prompt))
            handle.remove()
            outs.append(cache["out"][0, -1, :] if cache["out"].dim() > 1 else cache["out"])
        means[layer_idx] = torch.stack(outs).mean(dim=0)

    remove_all_hooks(model_eager)

    def make_replace_hook(mean_vec):
        def replace_hook(module, input, output, _mean=mean_vec):
            if isinstance(output, torch.Tensor):
                patched = output.clone()
                patched[0, -1, :] = _mean
                return patched
            else:
                patched = output[0].clone()
                patched[0, -1, :] = _mean
                return (patched,) + output[1:]
        return replace_hook

    handles = [
        model_eager.model.layers[layer_idx - 1].mlp.register_forward_hook(make_replace_hook(means[layer_idx]))
        for layer_idx in layer_indices
    ]
    inputs = make_inputs_eager(tokenizer, model_eager, target_prompt)
    with torch.no_grad():
        out = model_eager(**inputs)
    for h in handles:
        h.remove()

    logits = out.logits[0, -1, :]
    return {
        "top_digit": get_top_digit(tokenizer, logits),
        "logit_diff": logit_difference(tokenizer, logits, correct, wrong),
    }


def denoising_patch_mlp(
    tokenizer, model_eager, source_prompt: str, target_prompt: str,
    layer_indices: list[int], correct: int, wrong: int, norm_match: bool = True,
) -> dict:
    """Inject source MLP output(s) into target forward pass, norm-matched per site."""
    source_mlps = {}
    for layer_idx in layer_indices:
        remove_all_hooks(model_eager)
        source_cache = {}

        def capture_source(module, input, output, _cache=source_cache):
            o = output if isinstance(output, torch.Tensor) else output[0]
            _cache["out"] = o[0, -1, :].detach().clone()

        handle = model_eager.model.layers[layer_idx - 1].mlp.register_forward_hook(capture_source)
        with torch.no_grad():
            model_eager(**make_inputs_eager(tokenizer, model_eager, source_prompt))
        handle.remove()
        source_mlp = source_cache["out"]

        if norm_match:
            remove_all_hooks(model_eager)
            target_cache = {}

            def capture_target(module, input, output, _cache=target_cache):
                o = output if isinstance(output, torch.Tensor) else output[0]
                _cache["out"] = o[0, -1, :].detach().clone()

            handle = model_eager.model.layers[layer_idx - 1].mlp.register_forward_hook(capture_target)
            with torch.no_grad():
                model_eager(**make_inputs_eager(tokenizer, model_eager, target_prompt))
            handle.remove()
            target_norm = target_cache["out"].norm()
            source_mlp = source_mlp * (target_norm / (source_mlp.norm() + 1e-8))

        source_mlps[layer_idx] = source_mlp

    remove_all_hooks(model_eager)

    def make_inject_hook(src_vec):
        def inject_hook(module, input, output, _src=src_vec):
            if isinstance(output, torch.Tensor):
                patched = output.clone()
                patched[0, -1, :] = _src
                return patched
            else:
                patched = output[0].clone()
                patched[0, -1, :] = _src
                return (patched,) + output[1:]
        return inject_hook

    handles = [
        model_eager.model.layers[layer_idx - 1].mlp.register_forward_hook(make_inject_hook(source_mlps[layer_idx]))
        for layer_idx in layer_indices
    ]
    with torch.no_grad():
        out = model_eager(**make_inputs_eager(tokenizer, model_eager, target_prompt))
    for h in handles:
        h.remove()

    logits = out.logits[0, -1, :]
    return {
        "top_digit": get_top_digit(tokenizer, logits),
        "logit_diff": logit_difference(tokenizer, logits, correct, wrong),
    }


# -- Hidden state extraction -------------------------------------------------

def get_hidden_states_last_token(tokenizer, model_eager, prompt_text: str):
    """Return (n_layers+1, hidden_dim) residual-stream activations at the last token."""
    inputs = make_inputs_eager(tokenizer, model_eager, prompt_text)
    with torch.no_grad():
        out = model_eager(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
    hidden = torch.stack(out.hidden_states)
    return hidden[:, 0, -1, :].cpu().float().numpy()

# Re-runs the phase1_baseline logit lens in bfloat16 and float32 across 3 seeds
# and checks whether the auto-discovered writer layer (persistence backward-scan,
# same criterion as qwen.mechanistic.run_logit_lens) is stable. Only meaningful
# for models with a counting-writer to find (model_fails_p1=True); 3B/7B solve
# P1 and are skipped.
# uv run -m src.experiments.qwen.robustness --model qwen-1.5b

import argparse
import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.config import add_model_arg, get_config, load_results, output_path
from src.common.prompts import PROMPTS

LOGGER = logging.getLogger(__name__)

ROBUSTNESS_SEEDS = [42, 123, 456]


def get_top_digit(logits_1d, tokenizer) -> str:
    candidates = {}
    for n in range(1, 21):
        ids = tokenizer.encode(str(n), add_special_tokens=False)
        if len(ids) == 1:
            candidates[str(n)] = logits_1d[ids[0]].item()
    return max(candidates, key=candidates.get) if candidates else "?"


def make_inputs_eager(prompt_text: str, tokenizer, model_eager):
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model_eager.device)


def logit_lens_single(h, tokenizer, model_eager) -> str:
    with torch.no_grad():
        normed = model_eager.model.norm(h.unsqueeze(0).unsqueeze(0)).squeeze()
        logits = model_eager.lm_head.weight @ normed
    return get_top_digit(logits, tokenizer)


def find_writer(top_digits: list[str]) -> int | None:
    """Earliest layer such that every layer after it (through the last) keeps
    the same top digit as the final layer - same persistence criterion as
    qwen.mechanistic.run_logit_lens's backward scan."""
    attractor = top_digits[-1]
    writer = None
    for i in range(len(top_digits) - 2, -1, -1):
        if top_digits[i] != attractor:
            writer = i + 1
            break
    return writer


def run_single_condition(model_name: str, dtype: torch.dtype, seed: int) -> dict:
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_eager = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager",
    )
    model_eager.eval()

    inputs = make_inputs_eager(PROMPTS["phase1_baseline"]["text"], tokenizer, model_eager)
    with torch.no_grad():
        out = model_eager(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
    top_digits = [logit_lens_single(h[0, -1, :], tokenizer, model_eager) for h in out.hidden_states]
    writer = find_writer(top_digits)

    del model_eager
    torch.cuda.empty_cache()

    return {
        "dtype": str(dtype),
        "seed": seed,
        "attractor": top_digits[-1],
        "writer_layer": writer,
        "top_digit_trajectory": top_digits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    behavioral = load_results(cfg, "behavioral")
    p1_attractor = str(behavioral["phases"]["phase1_baseline"][0]["predicted"])
    p1_correct = PROMPTS["phase1_baseline"]["expected"]
    model_fails_p1 = p1_attractor != str(p1_correct)

    print(f"Model          : {cfg.model_name}")
    print(f"P1 attractor   : {p1_attractor}  correct: {p1_correct}  fails_p1: {model_fails_p1}")

    if not model_fails_p1:
        print("Model solves P1 - no counting-writer to test for stability. Skipping.")
        return

    mechanistic = load_results(cfg, "mechanistic_qwen")
    baseline_writer = find_writer([e["top_digit"] for e in mechanistic["logit_lens"]["phase1_baseline"]])
    print(f"Baseline writer layer from mechanistic_qwen.json: L{baseline_writer}")

    conditions = []
    for dtype in [torch.bfloat16, torch.float32]:
        for seed in ROBUSTNESS_SEEDS:
            label = f"{dtype}_seed{seed}"
            print(f"\nRunning condition: {label}")
            result = run_single_condition(cfg.model_name, dtype, seed)
            result["label"] = label
            conditions.append(result)
            print(f"  Writer layer: L{result['writer_layer']}  attractor: {result['attractor']}")

    writer_layers = [c["writer_layer"] for c in conditions]
    bf16_writers = [c["writer_layer"] for c in conditions if "bfloat16" in c["dtype"]]
    fp32_writers = [c["writer_layer"] for c in conditions if "float32" in c["dtype"]]

    is_stable = len(set(writer_layers)) == 1
    bf16_stable = len(set(bf16_writers)) == 1
    fp32_stable = len(set(fp32_writers)) == 1
    cross_precision_stable = bf16_stable and fp32_stable and set(bf16_writers) == set(fp32_writers)

    stability = {
        "all_writer_layers": writer_layers,
        "is_stable": is_stable,
        "bf16_stable": bf16_stable,
        "fp32_stable": fp32_stable,
        "cross_precision_stable": cross_precision_stable,
        "baseline_writer": baseline_writer,
    }

    print("\nStability analysis")
    print(f"  Writer layers across conditions: {writer_layers}")
    print(f"  All stable: {is_stable}")
    print(f"  Cross-precision stable: {cross_precision_stable}")

    output = {
        "model": cfg.model_name,
        "baseline_writer": baseline_writer,
        "conditions": conditions,
        "stability": stability,
    }

    save_path = output_path(cfg, "robustness_qwen.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

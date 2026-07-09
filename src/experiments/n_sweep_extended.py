# fills in gap n-values around behavioral.py's n-sweep and extends past n=20 to test attractor persistence.
# uv run -m src.experiments.n_sweep_extended --model qwen-3b

import argparse
import json
import logging

from src.common.config import add_model_arg, get_config, load_results, output_path
from src.common.prompts import make_prompt_repeated
from src.common.utils import extract_count, generate, load_generation_model

LOGGER = logging.getLogger(__name__)

EXTRA_NS = [13, 14, 16, 17, 18, 19, 25, 30, 35, 40]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    args = parser.parse_args()
    cfg = get_config(args.model)

    behavioral = load_results(cfg, "behavioral")
    existing_sweep = behavioral["n_sweep"]

    tokenizer, model, pipe = load_generation_model(cfg.model_name)

    print(f"\nExtended n-sweep ({cfg.key}), P1 repeated-token condition")
    print(f"  {'n':>4}  {'output':>8}  {'correct':>8}  {'tok guard':>10}")

    extra = {}
    for n in EXTRA_NS:
        payload = " ".join(["apple"] * n)
        tok_count = len(tokenizer.encode(payload, add_special_tokens=False))
        tok_ok = tok_count == n

        raw = generate(pipe, tokenizer, make_prompt_repeated(n), max_new_tokens=8)
        pred = extract_count(raw)
        correct = pred == n
        tok_str = "ok" if tok_ok else f"MISMATCH({tok_count})"

        print(f"  {n:>4}  {str(pred):>8}  {'yes' if correct else 'no':>8}  {tok_str:>10}")

        extra[n] = {"n": n, "p1_output": pred, "p1_correct": correct,
                    "tok_count": tok_count, "tok_guard_pass": tok_ok}

    combined = {str(n): v for n, v in extra.items()}
    for n_str, v in existing_sweep.items():
        combined[n_str] = {"n": v["n"], "p1_output": v["p1_output"], "p1_correct": v["p1_correct"],
                            "tok_count": v["tok_count_p1"], "tok_guard_pass": v["tok_guard_pass"]}

    combined_sorted = dict(sorted(combined.items(), key=lambda kv: int(kv[0])))

    print(f"\nFull combined sweep, n=5..20")
    print(f"  {'n':>4}  {'output':>8}  {'correct':>8}")
    for n_str, v in combined_sorted.items():
        print(f"  {v['n']:>4}  {str(v['p1_output']):>8}  {'yes' if v['p1_correct'] else 'no':>8}")

    output = {
        "model": cfg.model_name,
        "extra_ns": extra,
        "combined_sweep": combined_sorted,
    }

    save_path = output_path(cfg, "n_sweep_extended.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

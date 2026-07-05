"""Run every experiment for one model, in order.

Each stage runs as a subprocess so a crash in one stage does not take down
earlier results. Results land in results/<model short name>/.

Usage:
    uv run -m src.experiments.llama.run_all --model llama-1b
    uv run -m src.experiments.llama.run_all --model qwen-3b --only behavioral linear_probe
"""

import argparse
import subprocess
import sys

from src.common.config import add_model_arg, get_config

STAGES = [
    "behavioral",
    "linear_probe",
    "attention",
    "logit_lens",
    "causal",
    "robustness",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    parser.add_argument(
        "--only", nargs="+", choices=STAGES, default=None,
        help="Run only these stages (default: all, in order).",
    )
    args = parser.parse_args()
    cfg = get_config(args.model)

    stages = args.only or STAGES

    print(f"Running {len(stages)} stage(s) for {cfg.key} ({cfg.model_name})\n")

    for stage in stages:
        prefix = "src.experiments" if stage == "behavioral" else "src.experiments.llama"
        module = f"{prefix}.{stage}"
        print(f"\n{stage}")
        result = subprocess.run([sys.executable, "-m", module, "--model", args.model])
        if result.returncode != 0:
            print(f"Stage '{stage}' failed with exit code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

    print("\nAll stages completed.")


if __name__ == "__main__":
    main()

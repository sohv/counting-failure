"""
Odometer Pilot — run every experiment for one model, in order.

Each stage is a standalone script (own model load), run as a subprocess so a
crash in one stage doesn't take down the ones before it. Results land in
results/<model short name>/ (e.g. results/Llama-3.2-1B-Instruct/).

Usage:
    python run_all.py --model 1b
    python run_all.py --model 3b
    python run_all.py --model 1b --only behavioral_experiment attention_sink
"""

import argparse
import subprocess
import sys
from pathlib import Path

import src._paths as _paths  # noqa: F401  (adds config/ and data/ to sys.path)
from config import add_model_arg, get_config

STAGES = [
    "behavioral_experiment",
    "attention_sink",
    "linear_probe",
    "logit_lens_decomposition",
    "activation_patching",
    "additional_experiments",
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
    here = Path(__file__).parent

    print(f"Running {len(stages)} stage(s) for model={cfg.key} ({cfg.model_name})\n")

    for stage in stages:
        script = here / f"{stage}.py"
        print("\n" + "#" * 70)
        print(f"# {stage}")
        print("#" * 70)
        result = subprocess.run([sys.executable, str(script), "--model", args.model])
        if result.returncode != 0:
            print(f"\n[run_all] Stage '{stage}' failed with exit code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

    print("\nAll stages completed successfully.")


if __name__ == "__main__":
    main()

# runs every common and qwen-specific experiment for one qwen model, in order, each stage as a subprocess.
# uv run -m src.experiments.qwen.run_all --model qwen-1.5b

import argparse
import subprocess
import sys

from src.common.config import add_model_arg, get_config

STAGES = {
    "behavioral": "src.experiments.behavioral",
    "n_sweep_extended": "src.experiments.n_sweep_extended",
    "token_generality": "src.experiments.token_generality",
    "interleaved_noise": "src.experiments.interleaved_noise",
    "mechanistic": "src.experiments.qwen.mechanistic",
    "diagnostics": "src.experiments.qwen.diagnostics",
    "causal": "src.experiments.qwen.causal",
    "robustness": "src.experiments.qwen.robustness",
    "interleaved_probe": "src.experiments.qwen.interleaved_probe",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arg(parser)
    parser.add_argument(
        "--only", nargs="+", choices=list(STAGES), default=None,
        help="Run only these stages (default: all, in order).",
    )
    args = parser.parse_args()
    cfg = get_config(args.model)

    stage_names = args.only or list(STAGES)

    print(f"Running {len(stage_names)} stage(s) for {cfg.key} ({cfg.model_name})\n")

    for name in stage_names:
        module = STAGES[name]
        print(f"\n{name}")
        result = subprocess.run([sys.executable, "-m", module, "--model", args.model])
        if result.returncode != 0:
            print(f"Stage '{name}' failed with exit code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

    print("\nAll stages completed.")


if __name__ == "__main__":
    main()

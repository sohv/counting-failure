"""
Model registry and shared run configuration for the odometer experiments.

Everything that differs between the 1B and 3B pilots (model name, layer
indices where the wrong-answer "attractor" locks in, etc.) lives here so the
experiment scripts themselves stay identical across model sizes and are
driven purely by `--model {1b,3b}`.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    key: str                     # "1b" or "3b"
    model_name: str               # HF hub id
    n_layers: int                 # transformer block count
    critical_layers: list         # layers to scan in the MLP/attention decomposition
    lockin_layer: int             # layer where the attractor is believed to lock in
    gap_layers: list              # extra layers worth a per-n decomposition sweep
    steer_layer: int              # layer used for the steering-vector experiment
    # The two source notebooks diverged in which follow-up investigations
    # they actually ran. These flags gate those sections per model instead
    # of silently extrapolating an experiment to a model it was never run
    # against.
    runs_mechanistic_deep_dive: bool    # full layer sweep + mechanistic summary + paraphrase decomposition (3B notebook only)
    runs_additional_experiments: bool   # language robustness / symbols / CoT / steering vector (1B notebook only)

    @property
    def short_name(self) -> str:
        """e.g. 'Llama-3.2-1B-Instruct' — used as the results/ subfolder name."""
        return self.model_name.split("/")[-1]


MODEL_CONFIGS = {
    "1b": ModelConfig(
        key="1b",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        n_layers=16,
        critical_layers=[11, 12, 13, 14, 15, 16],
        lockin_layer=14,
        gap_layers=[14],
        steer_layer=14,
        runs_mechanistic_deep_dive=False,
        runs_additional_experiments=True,
    ),
    "3b": ModelConfig(
        key="3b",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        n_layers=28,
        critical_layers=[22, 23, 24, 25, 26, 27, 28],
        lockin_layer=26,
        gap_layers=[26, 23, 22],
        steer_layer=26,
        runs_mechanistic_deep_dive=True,
        runs_additional_experiments=False,
    ),
}

# ── Shared run parameters (identical across both original notebooks) ──────
N_RUNS = 10
SEEDS = list(range(N_RUNS))
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 16

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")


def add_model_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        required=True,
        help="Which pilot model to run the experiment against.",
    )
    return parser


def get_config(model_key: str) -> ModelConfig:
    return MODEL_CONFIGS[model_key]


def output_path(cfg: ModelConfig, filename: str) -> str:
    """Every artifact (json/png) lands in results/<model short name>/, e.g.
    results/Llama-3.2-1B-Instruct/odometer_results.json, so 1B and 3B runs
    never collide."""
    model_dir = os.path.join(RESULTS_ROOT, cfg.short_name)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, filename)

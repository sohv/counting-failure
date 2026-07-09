"""Model registry and shared run configuration for the counting-failure experiments."""

import argparse
import json
import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    key: str
    model_name: str
    n_layers: int
    critical_layers: list[int] | None = None
    lockin_layer: int | None = None
    gap_layers: list[int] | None = None
    steer_layer: int | None = None
    writer_layers: list[int] | None = None

    @property
    def short_name(self) -> str:
        return self.model_name.split("/")[-1]

    @property
    def ablation_sites(self) -> list[int]:
        """Sites for the individual/joint mean-ablation and patching comparison."""
        if self.writer_layers is not None:
            return self.writer_layers
        return [self.lockin_layer]


MODEL_CONFIGS = {
    "llama-1b": ModelConfig(
        key="llama-1b",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        n_layers=16,
        critical_layers=[11, 12, 13, 14, 15, 16],
        lockin_layer=14,
        gap_layers=[14],
        steer_layer=14,
    ),
    "llama-3b": ModelConfig(
        key="llama-3b",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        n_layers=28,
        critical_layers=[22, 23, 24, 25, 26, 27, 28],
        lockin_layer=26,
        gap_layers=[26, 23, 22],
        steer_layer=26,
        writer_layers=[22, 26],
    ),
    "qwen-1.5b": ModelConfig(
        key="qwen-1.5b",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        n_layers=28,
        writer_layers=[22, 24],
    ),
    "qwen-3b": ModelConfig(
        key="qwen-3b",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        n_layers=36,
    ),
    "qwen-7b": ModelConfig(
        key="qwen-7b",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        n_layers=28,
    ),
}

N_RUNS = 10
SEEDS = list(range(N_RUNS))
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 16

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")


def add_model_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        required=True,
        help="Which model to run the experiment against.",
    )
    return parser


def get_config(model_key: str) -> ModelConfig:
    return MODEL_CONFIGS[model_key]


def output_path(cfg: ModelConfig, filename: str) -> str:
    model_dir = os.path.join(RESULTS_ROOT, cfg.short_name)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, filename)


def load_results(cfg: ModelConfig, name: str) -> dict:
    """Load JSON results from a previous experiment stage."""
    path = output_path(cfg, f"{name}.json")
    with open(path) as f:
        return json.load(f)

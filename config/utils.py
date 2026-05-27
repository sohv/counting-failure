import re
import json
import random
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

CONFIG_DIR = Path(__file__).parent


def load_config_from_json(config_file: str = None) -> dict:
    if config_file is None:
        config_file = CONFIG_DIR / "config.json"
    else:
        config_file = Path(config_file)
        if not config_file.is_absolute():
            config_file = CONFIG_DIR / config_file
    
    if not config_file.exists():
        return {}
    with open(config_file, 'r') as f:
        return json.load(f)


def get_llm_config(llm_key: str, config_file: str = None) -> dict:
    config = load_config_from_json(config_file)
    llm_configs = config.get("llm_config", {})
    return llm_configs.get(llm_key, {})


@dataclass
class Config:
    model_name: str
    n_runs: int = 10
    temperature: float = 0.0
    max_new_tokens: int = 16
    device: str = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_name: str):
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
    
    return model, tokenizer, pipe

def extract_count(raw_output: str) -> Optional[int]:
    match = re.search(r'\b(\d+)\b', raw_output.strip())
    return int(match.group(1)) if match else None

def get_top_digit(logits_tensor) -> Optional[int]:
    if logits_tensor is None:
        return None
    digit_ids = {str(i): tokenizer.encode(str(i), add_special_tokens=False)[0] 
                 for i in range(10)}
    digit_logits = {i: logits_tensor[digit_ids[str(i)]].item() 
                    for i in range(10)}
    return max(digit_logits, key=digit_logits.get)

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
        return self.n_correct / self.n_runs if self.n_runs > 0 else 0.0
    
    def dist(self) -> dict:
        return dict(Counter(str(p) for p in self.predictions))

def load_prompts(prompts_file: str = None) -> dict:
    if prompts_file is None:
        prompts_file = CONFIG_DIR / "prompts.json"
    else:
        prompts_file = Path(prompts_file)
        if not prompts_file.is_absolute():
            prompts_file = CONFIG_DIR / prompts_file
    
    with open(prompts_file, 'r') as f:
        return json.load(f)

def remove_all_hooks(model):
    for name, module in model.named_modules():
        module._forward_hooks.clear()

def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

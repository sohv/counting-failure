"""
Shared utilities: config, model loading, extraction, etc.
"""
import re
import json
import random
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

@dataclass
class Config:
    """Experiment configuration."""
    model_name: str
    n_runs: int = 10
    temperature: float = 0.0
    max_new_tokens: int = 16
    device: str = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer with proper device mapping."""
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
    """Extract first integer from model output."""
    match = re.search(r'\b(\d+)\b', raw_output.strip())
    return int(match.group(1)) if match else None

def get_top_digit(logits_tensor) -> Optional[int]:
    """Get top digit (0-9) from logits."""
    if logits_tensor is None:
        return None
    digit_ids = {str(i): tokenizer.encode(str(i), add_special_tokens=False)[0] 
                 for i in range(10)}
    digit_logits = {i: logits_tensor[digit_ids[str(i)]].item() 
                    for i in range(10)}
    return max(digit_logits, key=digit_logits.get)

@dataclass
class PhaseSummary:
    """Summary statistics for one phase."""
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

def load_prompts(prompts_file: str = "prompts.json") -> dict:
    """Load prompts from JSON file."""
    with open(prompts_file, 'r') as f:
        return json.load(f)

def remove_all_hooks(model):
    """Remove all registered forward hooks from model."""
    for name, module in model.named_modules():
        module._forward_hooks.clear()

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

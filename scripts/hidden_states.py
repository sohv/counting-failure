from typing import List
import torch


def get_hidden_states(model, tokenizer, prompt_text: str, layer_idx: int = -1):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    hidden_states = outputs.hidden_states[layer_idx]
    return hidden_states


def get_hidden_states_from_text(model, tokenizer, texts: List[str], layer_idx: int = -1):
    states = []
    for text in texts:
        hs = get_hidden_states(model, tokenizer, text, layer_idx)
        states.append(hs)
    return states

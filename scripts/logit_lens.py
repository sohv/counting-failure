import torch
from typing import Optional


def logit_lens(
    model, tokenizer, prompt_text: str,
    target_token: int = None,
    max_layers: int = None
) -> dict:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    if target_token is None:
        target_token = inputs.input_ids.shape[1] - 1
    
    results = {}
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    hidden_states = outputs.hidden_states
    n_layers = len(hidden_states) if max_layers is None else max_layers
    
    for layer_idx in range(n_layers):
        hs = hidden_states[layer_idx]
        hs_target = hs[0, target_token, :].unsqueeze(0)
        
        with torch.no_grad():
            lm_head_logits = model.lm_head(hs_target)
        
        top_tokens = lm_head_logits.topk(5)[1][0]
        top_texts = [tokenizer.decode([t]) for t in top_tokens]
        
        results[layer_idx] = {
            "top_tokens": top_tokens.cpu().tolist(),
            "top_texts": top_texts,
            "logits": lm_head_logits[0].cpu().numpy(),
        }
    
    return results


def logit_lens_single(model, tokenizer, prompt_text: str, layer_idx: int) -> dict:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    hs = outputs.hidden_states[layer_idx][0, -1, :].unsqueeze(0)
    
    with torch.no_grad():
        logits = model.lm_head(hs)
    
    return {
        "logits": logits[0].cpu().numpy(),
        "layer": layer_idx,
    }


def decompose_layer(
    model, tokenizer, prompt_text: str, layer_idx: int
) -> dict:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    attn_outputs = []
    mlp_outputs = []
    
    def hook_post_attn(module, input, output):
        attn_outputs.append(output[0].detach())
    
    def hook_post_layer(module, input, output):
        mlp_outputs.append(output[0].detach())
    
    layer = model.model.layers[layer_idx]
    attn_hook = layer.self_attn.register_forward_hook(hook_post_attn)
    mlp_hook = layer.mlp.register_forward_hook(hook_post_layer)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    attn_hook.remove()
    mlp_hook.remove()
    
    return {
        "layer": layer_idx,
        "attn_output_shape": attn_outputs[0].shape if attn_outputs else None,
        "mlp_output_shape": mlp_outputs[0].shape if mlp_outputs else None,
    }

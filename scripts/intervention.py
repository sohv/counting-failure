import torch


def save_h(model, layer_idx: int, prompt_text: str, tokenizer) -> torch.Tensor:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    return outputs.hidden_states[layer_idx].detach()


def patch_h(model, layer_idx: int, h_source: torch.Tensor, prompt_text: str, tokenizer):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    original_forward = model.model.layers[layer_idx].forward
    
    def patched_forward(x, *args, **kwargs):
        result = original_forward(x, *args, **kwargs)
        result[0][:, :, :] = h_source
        return result
    
    model.model.layers[layer_idx].forward = patched_forward
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    model.model.layers[layer_idx].forward = original_forward
    return outputs


def zero_ablate_mlp(model, layer_idx: int, prompt_text: str, tokenizer):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    def zero_mlp_hook(module, input, output):
        return (torch.zeros_like(output[0]),) + output[1:]
    
    layer = model.model.layers[layer_idx]
    hook = layer.mlp.register_forward_hook(zero_mlp_hook)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    hook.remove()
    return outputs

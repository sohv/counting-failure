import torch
from typing import Optional


def inspect_hook(module, input, output):
    return output


def hook_pre_layer(module, input, stored_tensors: dict = None):
    if stored_tensors is not None:
        stored_tensors['pre_input'] = input[0].detach() if isinstance(input, tuple) else input.detach()
    return None


def hook_post_layer(module, input, output, stored_tensors: dict = None):
    if stored_tensors is not None:
        stored_tensors['post_output'] = output[0].detach() if isinstance(output, tuple) else output.detach()
    return output


def hook_post_attn(module, input, output, stored_tensors: dict = None):
    if stored_tensors is not None:
        if isinstance(output, tuple):
            stored_tensors['attn_output'] = output[0].detach()
        else:
            stored_tensors['attn_output'] = output.detach()
    return output


def steer_hook(module, input, output, steering_vector: torch.Tensor = None, scale: float = 1.0):
    if steering_vector is not None:
        if isinstance(output, tuple):
            output = (output[0] + scale * steering_vector.to(output[0].device),) + output[1:]
        else:
            output = output + scale * steering_vector.to(output.device)
    return output

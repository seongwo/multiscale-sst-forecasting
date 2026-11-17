"""Activation checkpointing/chunking without new weights or changed tensor maps."""
import torch
from torch.utils.checkpoint import checkpoint

def install(model, node_chunk=32, checkpoint_enabled=True):
    attention=model.core.intra_patch_attention
    original_attention=attention.forward
    def attention_forward(*args):
        if checkpoint_enabled and attention.training and torch.is_grad_enabled():
            return checkpoint(original_attention,*args,use_reentrant=False,preserve_rng_state=True)
        return original_attention(*args)
    attention.forward=attention_forward
    feedforward=model.core.ff
    original_feedforward=feedforward.forward
    def feedforward_forward(x):
        output=[]
        for part in x.split(node_chunk,dim=1):
            if checkpoint_enabled and feedforward.training and torch.is_grad_enabled():
                output.append(checkpoint(original_feedforward,part,use_reentrant=False,preserve_rng_state=True))
            else:
                output.append(original_feedforward(part))
        return torch.cat(output,dim=1)
    feedforward.forward=feedforward_forward
    return model

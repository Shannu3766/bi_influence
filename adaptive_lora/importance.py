import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .utils import get_lora_layers
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def compute_bi_scores(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Computes Block Influence (BI) scores for each LoRA layer based on
    relative change in activation magnitude (not just cosine similarity).
    
    Formula:
        BI_i = ||output|| / (||input|| + eps)
    """

    model.eval()
    lora_layers = get_lora_layers(model)
    if not lora_layers:
        logger.warning("No LoRA layers found in model. Returning empty scores.")
        return {}

    activations = {name: {'in': [], 'out': []} for name in lora_layers}

    def make_hook(name):
        def hook(module, inp, out):
            x_in = inp[0].detach().cpu().float()
            x_out = out.detach().cpu().float()
            activations[name]['in'].append(x_in)
            activations[name]['out'].append(x_out)
        return hook

    hooks = []
    for name, layer in lora_layers.items():
        hooks.append(layer.register_forward_hook(make_hook(name)))

    try:
        num_batches = min(3, len(dataloader))
        iterator = iter(dataloader)
        for _ in range(num_batches):
            batch = next(iterator)
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                model(**batch)
    except Exception as e:
        logger.error(f"Error during BI score computation: {e}")
    finally:
        for h in hooks:
            h.remove()

    bi_scores = {}
    eps = 1e-6
    for name in lora_layers:
        if not activations[name]['in'] or not activations[name]['out']:
            continue
        x_in = torch.cat(activations[name]['in'])
        x_out = torch.cat(activations[name]['out'])

        # Compute mean activation norms
        in_norm = x_in.norm(dim=-1).mean().item() + eps
        out_norm = x_out.norm(dim=-1).mean().item() + eps

        # Influence = relative output norm
        bi = out_norm / in_norm
        bi_scores[name] = bi

    # Normalize scores across layers for stability
    s = torch.tensor(list(bi_scores.values()))
    s_norm = (s - s.min()) / (s.max() - s.min() + eps)
    bi_scores = {k: float(v) for k, v in zip(bi_scores.keys(), s_norm)}

    model.train()
    return bi_scores

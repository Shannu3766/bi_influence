import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .utils import get_lora_layers
import logging
from typing import Dict

logger = logging.getLogger(__name__)
import torch
import torch.nn.functional as F

def compute_bi_scores(model, dataloader, device):
    model.eval()
    lora_layers = get_lora_layers(model)
    activations = {name: {'in': [], 'out': []} for name in lora_layers}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name]['in'].append(inp[0].detach().cpu().float())
            activations[name]['out'].append(out.detach().cpu().float())
        return hook

    hooks = [layer.register_forward_hook(make_hook(name)) for name, layer in lora_layers.items()]

    try:
        batch = next(iter(dataloader))
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            model(**batch)
    finally:
        for h in hooks:
            h.remove()

    bi_scores = {}
    for name in lora_layers:
        if not activations[name]['in'] or not activations[name]['out']:
            continue

        x_in = torch.cat(activations[name]['in'])
        x_out = torch.cat(activations[name]['out'])

        # Flatten to [N, D]
        x_in = x_in.view(-1, x_in.size(-1))
        x_out = x_out.view(-1, x_out.size(-1))

        # Compute mean cosine similarity ρ_i
        rho = F.cosine_similarity(x_in, x_out, dim=1).mean().item()

        # Compute BI score s_i = 1 - ρ_i
        bi_scores[name] = 1.0 - rho

    # Normalize to [0, 1]
    s = torch.tensor(list(bi_scores.values()))
    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return {k: float(v) for k, v in zip(bi_scores.keys(), s_norm)}

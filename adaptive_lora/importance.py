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
    average cosine similarity between input/output activations.

    s_i = 1 - mean(cos_sim(x_in, x_out))
    """
    model.eval()
    lora_layers = get_lora_layers(model)

    if not lora_layers:
        logger.warning("No LoRA layers found in the model. Returning empty scores.")
        return {}

    activations = {name: {'in': [], 'out': []} for name in lora_layers}

    def get_hook(name):
        def hook(module, input_act, output_act):
            activations[name]['in'].append(input_act[0].detach().cpu().float())
            activations[name]['out'].append(output_act.detach().cpu().float())
        return hook

    # Register hooks
    hooks = []
    for name, layer in lora_layers.items():
        hooks.append(layer.register_forward_hook(get_hook(name)))

    try:
        # 🔁 Collect multiple batches (not just one!)
        num_batches = min(3, len(dataloader))  # take up to 3 small batches
        iterator = iter(dataloader)

        for _ in range(num_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break

            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                model(**batch)

    except Exception as e:
        logger.error(f"Failed during BI score computation: {e}")

    finally:
        for hook in hooks:
            hook.remove()

    # 🧮 Compute BI scores
    bi_scores = {}
    for name in lora_layers:
        if not activations[name]['in'] or not activations[name]['out']:
            logger.warning(f"No activations captured for {name}. Skipping.")
            continue

        in_acts = torch.cat(activations[name]['in'])
        out_acts = torch.cat(activations[name]['out'])

        in_flat = in_acts.view(-1, in_acts.size(-1))
        out_flat = out_acts.view(-1, out_acts.size(-1))

        # compute cosine similarity
        cos_sim = F.cosine_similarity(in_flat, out_flat, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # avoid >1 or <−1 numerical issues

        avg_cos_sim = cos_sim.mean().item()
        bi_scores[name] = 1.0 - avg_cos_sim  # higher → more influential

    model.train()
    return bi_scores

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from typing import Dict
# import logging

# logger = logging.getLogger(__name__)

# def compute_bi_scores(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, num_batches: int = 2) -> Dict[str, float]:
#     """
#     Computes the original Block Influence (BI) importance score for each LoRA layer.
#     BI_i = 1 - ρ_i, where ρ_i is the average cosine similarity between input and output activations.

#     Based on Algorithm 1 from the Adaptive LoRA paper.
#     """
#     model.eval()

#     # 1️⃣ Identify LoRA layers
#     from adaptive_lora.utils import get_lora_layers
#     lora_layers = get_lora_layers(model)
#     if not lora_layers:
#         logger.warning("No LoRA layers found in the model. Returning empty scores.")
#         return {}

#     # 2️⃣ Store activations
#     activations = {name: {'in': [], 'out': []} for name in lora_layers}
#     hooks = []

#     def hook_factory(name):
#         """Hook function to capture input/output activations."""
#         def hook(module, input_act, output_act):
#             try:
#                 if input_act is None or output_act is None:
#                     return
#                 x_in = input_act[0].detach().to("cpu", torch.float32)
#                 x_out = output_act.detach().to("cpu", torch.float32)
#                 activations[name]['in'].append(x_in)
#                 activations[name]['out'].append(x_out)
#             except Exception as e:
#                 logger.warning(f"Hook failed for {name}: {e}")
#         return hook

#     # 3️⃣ Register hooks
#     for name, layer in lora_layers.items():
#         try:
#             hooks.append(layer.register_forward_hook(hook_factory(name)))
#         except Exception as e:
#             logger.warning(f"Failed to register hook for {name}: {e}")

#     # 4️⃣ Forward pass over small validation subset
#     try:
#         iterator = iter(dataloader)
#         for i in range(min(num_batches, len(dataloader))):
#             batch = next(iterator)
#             batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
#             with torch.no_grad():
#                 model(**batch)
#     except Exception as e:
#         logger.error(f"Error during BI forward pass: {e}")
#     finally:
#         for h in hooks:
#             h.remove()

#     # 5️⃣ Compute cosine similarity and BI scores
#     bi_scores = {}
#     eps = 1e-8
#     for name in lora_layers:
#         if not activations[name]['in'] or not activations[name]['out']:
#             bi_scores[name] = 0.0
#             continue

#         try:
#             in_acts = torch.cat(activations[name]['in'])
#             out_acts = torch.cat(activations[name]['out'])

#             # Flatten [batch, seq, dim] -> [batch*seq, dim]
#             in_flat = in_acts.view(-1, in_acts.size(-1))
#             out_flat = out_acts.view(-1, out_acts.size(-1))

#             # Compute per-token cosine similarity ρ_i
#             cos_sim = F.cosine_similarity(in_flat, out_flat, dim=1)
#             cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # numerical safety

#             rho_i = cos_sim.mean().item()

#             # Compute BI score s_i = 1 - ρ_i
#             s_i = 1.0 - rho_i

#             # Handle NaNs or infs
#             if not torch.isfinite(torch.tensor(s_i)):
#                 s_i = 0.0

#             bi_scores[name] = s_i
#         except Exception as e:
#             logger.warning(f"Failed computing BI for {name}: {e}")
#             bi_scores[name] = 0.0

#     s = torch.tensor(list(bi_scores.values()), dtype=torch.float32)
#     if torch.isnan(s).any() or (s.max() - s.min()) < eps:
#         bi_scores = {k: 1.0 for k in bi_scores}
#     else:
#         s = (s - s.min()) / (s.max() - s.min() + eps)
#         bi_scores = {k: float(v) for k, v in zip(bi_scores.keys(), s)}

#     model.train()
#     return bi_scores


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
import logging
from adaptive_lora.utils import get_lora_layers

logger = logging.getLogger(__name__)

def compute_bi_scores(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, num_batches: int = 4) -> Dict[str, float]:
    """
    Computes the original (unnormalized) Block Influence (BI) score for each LoRA layer:
        s_i = 1 - ρ_i
    where ρ_i is the mean cosine similarity between input and output activations.
    """
    model.eval()
    lora_layers = get_lora_layers(model)
    if not lora_layers:
        logger.warning("No LoRA layers found in the model. Returning empty scores.")
        return {}

    activations = {name: {'in': [], 'out': []} for name in lora_layers}
    hooks = []

    def hook_factory(name):
        def hook(module, input_act, output_act):
            try:
                if input_act is None or output_act is None:
                    return
                x_in = input_act[0].detach().to("cpu", torch.float32)
                x_out = output_act.detach().to("cpu", torch.float32)
                activations[name]['in'].append(x_in)
                activations[name]['out'].append(x_out)
            except Exception as e:
                logger.warning(f"Hook failed for {name}: {e}")
        return hook

    # Register hooks
    for name, layer in lora_layers.items():
        try:
            hooks.append(layer.register_forward_hook(hook_factory(name)))
        except Exception as e:
            logger.warning(f"Failed to register hook for {name}: {e}")

    # Run small subset
    try:
        iterator = iter(dataloader)
        for i in range(min(num_batches, len(dataloader))):
            batch = next(iterator)
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                model(**batch)
    except Exception as e:
        logger.error(f"Error during BI forward pass: {e}")
    finally:
        for h in hooks:
            h.remove()

    bi_scores = {}
    eps = 1e-8
    for name in lora_layers:
        if not activations[name]['in'] or not activations[name]['out']:
            bi_scores[name] = 0.0
            continue

        try:
            in_acts = torch.cat(activations[name]['in'])
            out_acts = torch.cat(activations[name]['out'])

            in_flat = in_acts.view(-1, in_acts.size(-1))
            out_flat = out_acts.view(-1, out_acts.size(-1))

            cos_sim = F.cosine_similarity(in_flat, out_flat, dim=1)
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

            rho_i = cos_sim.mean().item()
            s_i = 1.0 - rho_i
            if not torch.isfinite(torch.tensor(s_i)):
                s_i = 0.0
            bi_scores[name] = s_i
        except Exception as e:
            logger.warning(f"Failed computing BI for {name}: {e}")
            bi_scores[name] = 0.0

    model.train()
    return bi_scores

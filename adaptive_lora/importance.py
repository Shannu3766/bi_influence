# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from .utils import get_lora_layers
# import logging
# from typing import Dict

# logger = logging.getLogger(__name__)

# def compute_bi_scores(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
#     """
#     Computes Block Influence (BI) scores for each LoRA layer using
#     cosine similarity between mean input/output representations.

#     Uses dimension-safe pooling to handle layers with different in/out shapes.
#     """
#     model.eval()
#     lora_layers = get_lora_layers(model)
#     if not lora_layers:
#         logger.warning("No LoRA layers found in model. Returning empty scores.")
#         return {}

#     activations = {name: {'in': [], 'out': []} for name in lora_layers}

#     def make_hook(name):
#         def hook(module, inp, out):
#             x_in = inp[0].detach().to("cpu", torch.float32)
#             x_out = out.detach().to("cpu", torch.float32)

#             # Flatten batch & sequence, keep feature dim last
#             x_in = x_in.view(-1, x_in.shape[-1])
#             x_out = x_out.view(-1, x_out.shape[-1])

#             # Mean-pool to handle mismatched dims safely
#             in_mean = x_in.mean(dim=0)
#             out_mean = x_out.mean(dim=0)

#             # Store mean representations (safe for cosine)
#             activations[name]['in'].append(in_mean)
#             activations[name]['out'].append(out_mean)
#         return hook

#     hooks = [layer.register_forward_hook(make_hook(name)) for name, layer in lora_layers.items()]

#     try:
#         num_batches = min(3, len(dataloader))
#         iterator = iter(dataloader)
#         for _ in range(num_batches):
#             batch = next(iterator)
#             batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
#             with torch.no_grad():
#                 model(**batch)
#     except Exception as e:
#         logger.error(f"Error during BI score computation: {e}")
#     finally:
#         for h in hooks:
#             h.remove()

#     bi_scores = {}
#     eps = 1e-8
#     for name in lora_layers:
#         if not activations[name]['in'] or not activations[name]['out']:
#             continue

#         # Average across batches
#         x_in = torch.stack(activations[name]['in']).mean(dim=0)
#         x_out = torch.stack(activations[name]['out']).mean(dim=0)

#         # Match dimensions by truncation or padding
#         if x_in.shape[0] != x_out.shape[0]:
#             min_dim = min(x_in.shape[0], x_out.shape[0])
#             x_in = x_in[:min_dim]
#             x_out = x_out[:min_dim]

#         # Cosine similarity
#         rho = F.cosine_similarity(x_in.unsqueeze(0), x_out.unsqueeze(0)).mean().item()
#         bi = 1.0 - rho  # BI = 1 - cosine similarity

#         bi_scores[name] = bi

#     # Normalize scores across layers
#     s = torch.tensor(list(bi_scores.values()))
#     s_norm = (s - s.min()) / (s.max() - s.min() + eps)
#     bi_scores = {k: float(v) for k, v in zip(bi_scores.keys(), s_norm)}

#     model.train()
#     return bi_scores


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .utils import get_lora_layers
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# def compute_bi_scores(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
#     """
#     Universal Block Influence (BI) computation.
#     Compatible with classification, QA, and generative tasks.
#     Uses mean cosine decorrelation between layer inputs and outputs.
#     """

#     model.eval()
#     lora_layers = get_lora_layers(model)
#     if not lora_layers:
#         logger.warning("No LoRA layers found in model. Returning empty scores.")
#         return {}

#     activations = {name: {'in': [], 'out': []} for name in lora_layers}

#     def make_hook(name):
#         def hook(module, inp, out):
#             x_in = inp[0].detach().to("cpu", torch.float32)
#             x_out = out.detach().to("cpu", torch.float32)

#             # Flatten any extra dimensions (e.g., seq_len, heads)
#             x_in = x_in.view(-1, x_in.shape[-1])
#             x_out = x_out.view(-1, x_out.shape[-1])

#             # Mean pool over sequence/batch to avoid dim mismatch
#             in_mean = x_in.mean(dim=0)
#             out_mean = x_out.mean(dim=0)

#             activations[name]['in'].append(in_mean)
#             activations[name]['out'].append(out_mean)
#         return hook

#     hooks = [layer.register_forward_hook(make_hook(name)) for name, layer in lora_layers.items()]

#     try:
#         iterator = iter(dataloader)
#         for _ in range(min(3, len(dataloader))):
#             batch = next(iterator)
#             # Move batch tensors to device automatically
#             batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
#             with torch.no_grad():
#                 # ✅ Works for LM, CLS, QA, etc.
#                 model(**batch)
#     except Exception as e:
#         logger.error(f"Error during BI computation: {e}")
#     finally:
#         for h in hooks:
#             h.remove()

#     bi_scores = {}
#     eps = 1e-8
#     for name in lora_layers:
#         if not activations[name]['in'] or not activations[name]['out']:
#             continue

#         x_in = torch.stack(activations[name]['in']).mean(dim=0)
#         x_out = torch.stack(activations[name]['out']).mean(dim=0)

#         # Match dims by truncation or padding
#         if x_in.shape[0] != x_out.shape[0]:
#             min_dim = min(x_in.shape[0], x_out.shape[0])
#             x_in = x_in[:min_dim]
#             x_out = x_out[:min_dim]

#         rho = F.cosine_similarity(x_in.unsqueeze(0), x_out.unsqueeze(0)).mean().item()
#         bi = 1.0 - rho  # decorrelation = importance
#         bi_scores[name] = bi

#     # Normalize across layers
#     s = torch.tensor(list(bi_scores.values()))
#     s_norm = (s - s.min()) / (s.max() - s.min() + eps)
#     bi_scores = {k: float(v) for k, v in zip(bi_scores.keys(), s_norm)}

#     model.train()
#     return bi_scores


# def compute_bi_scores(model, dataloader, device, num_batches=2):
#     model.eval()
#     lora_layers = get_lora_layers(model)
#     if not lora_layers:
#         return {}

#     activations = {name: {'in': [], 'out': []} for name in lora_layers}

#     def hook_factory(name):
#         def hook(module, inp, out):
#             if inp is None or out is None:
#                 return
#             try:
#                 x_in = inp[0].detach().to("cpu", torch.float32)
#                 x_out = out.detach().to("cpu", torch.float32)
#                 activations[name]['in'].append(x_in)
#                 activations[name]['out'].append(x_out)
#             except Exception:
#                 pass
#         return hook

#     hooks = [layer.register_forward_hook(hook_factory(name)) for name, layer in lora_layers.items()]

#     try:
#         it = iter(dataloader)
#         for _ in range(min(num_batches, len(dataloader))):
#             batch = next(it)
#             batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
#             with torch.no_grad():
#                 model(**batch)
#     finally:
#         for h in hooks:
#             h.remove()

#     bi_scores = {}
#     eps = 1e-8
#     for name in lora_layers:
#         ins, outs = activations[name]['in'], activations[name]['out']
#         if not ins or not outs:
#             bi_scores[name] = 0.0
#             continue
#         x_in = torch.cat(ins)
#         x_out = torch.cat(outs)
#         in_norm = x_in.norm(dim=-1).mean().item()
#         out_norm = x_out.norm(dim=-1).mean().item()
#         bi = out_norm / (in_norm + eps)
#         if not torch.isfinite(torch.tensor(bi)):
#             bi = 0.0
#         bi_scores[name] = bi

#     s = torch.tensor(list(bi_scores.values()))
#     if s.isnan().any() or (s.max() - s.min()) < eps:
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

logger = logging.getLogger(__name__)

def compute_bi_scores(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, num_batches: int = 2) -> Dict[str, float]:
    """
    Computes the original Block Influence (BI) importance score for each LoRA layer.
    BI_i = 1 - ρ_i, where ρ_i is the average cosine similarity between input and output activations.

    Based on Algorithm 1 from the Adaptive LoRA paper.
    """
    model.eval()

    # 1️⃣ Identify LoRA layers
    from adaptive_lora.utils import get_lora_layers
    lora_layers = get_lora_layers(model)
    if not lora_layers:
        logger.warning("No LoRA layers found in the model. Returning empty scores.")
        return {}

    # 2️⃣ Store activations
    activations = {name: {'in': [], 'out': []} for name in lora_layers}
    hooks = []

    def hook_factory(name):
        """Hook function to capture input/output activations."""
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

    # 3️⃣ Register hooks
    for name, layer in lora_layers.items():
        try:
            hooks.append(layer.register_forward_hook(hook_factory(name)))
        except Exception as e:
            logger.warning(f"Failed to register hook for {name}: {e}")

    # 4️⃣ Forward pass over small validation subset
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

    # 5️⃣ Compute cosine similarity and BI scores
    bi_scores = {}
    eps = 1e-8
    for name in lora_layers:
        if not activations[name]['in'] or not activations[name]['out']:
            bi_scores[name] = 0.0
            continue

        try:
            in_acts = torch.cat(activations[name]['in'])
            out_acts = torch.cat(activations[name]['out'])

            # Flatten [batch, seq, dim] -> [batch*seq, dim]
            in_flat = in_acts.view(-1, in_acts.size(-1))
            out_flat = out_acts.view(-1, out_acts.size(-1))

            # Compute per-token cosine similarity ρ_i
            cos_sim = F.cosine_similarity(in_flat, out_flat, dim=1)
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # numerical safety

            rho_i = cos_sim.mean().item()

            # Compute BI score s_i = 1 - ρ_i
            s_i = 1.0 - rho_i

            # Handle NaNs or infs
            if not torch.isfinite(torch.tensor(s_i)):
                s_i = 0.0

            bi_scores[name] = s_i
        except Exception as e:
            logger.warning(f"Failed computing BI for {name}: {e}")
            bi_scores[name] = 0.0

    # 6️⃣ Normalize across layers
    s = torch.tensor(list(bi_scores.values()), dtype=torch.float32)
    if torch.isnan(s).any() or (s.max() - s.min()) < eps:
        bi_scores = {k: 1.0 for k in bi_scores}
    else:
        s = (s - s.min()) / (s.max() - s.min() + eps)
        bi_scores = {k: float(v) for k, v in zip(bi_scores.keys(), s)}

    model.train()
    return bi_scores

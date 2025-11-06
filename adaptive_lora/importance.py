import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .utils import get_lora_layers
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_bi_scores(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Computes the Block Influence (BI) importance score for each LoRA layer.

    The BI score is defined as s_i = 1 - ρ_i, where ρ_i is the average
    cosine similarity between the layer's input and output activations.

    Args:
        model: The PEFT model.
        dataloader: A DataLoader for a (small) subset of the validation data.
        device: The device to run computations on.

    Returns:
        A dictionary mapping layer names to their BI importance score.
    """
    model.eval()
    lora_layers = get_lora_layers(model)
    
    if not lora_layers:
        logger.warning("No LoRA layers found in the model. Returning empty scores.")
        return {}

    # Stores {layer_name: {'in': [tensor, ...], 'out': [tensor, ...]}}
    activations = {name: {'in': [], 'out': []} for name in lora_layers}
    hooks = []

    def get_hook(name: str):
        """Creates a hook function to capture activations."""
        def hook(module, input_act, output_act):
            # We detach, move to CPU, and convert to float to save GPU memory
            # The input is a tuple, we typically want the first element
            activations[name]['in'].append(input_act[0].detach().cpu().float())
            activations[name]['out'].append(output_act.detach().cpu().float())
        return hook

    # Register hooks
    for name, layer in lora_layers.items():
        hooks.append(layer.register_forward_hook(get_hook(name)))

    # --- Run a single batch from the dataloader ---
    try:
        batch = next(iter(dataloader))
        # Move batch to the correct device
        batch = {
            k: v.to(device) 
            for k, v in batch.items() 
            if isinstance(v, torch.Tensor)
        }
        
        with torch.no_grad():
            model(**batch)
            
    except Exception as e:
        logger.error(f"Failed during model forward pass for BI score computation: {e}")
    finally:
        # --- Always remove hooks ---
        for hook in hooks:
            hook.remove()

    # --- Compute BI scores ---
    bi_scores = {}
    for name in lora_layers:
        if not activations[name]['in'] or not activations[name]['out']:
            logger.warning(f"No activations captured for layer {name}. Skipping.")
            continue

        # Concatenate all activations from the batch
        try:
            in_acts = torch.cat(activations[name]['in'])
            out_acts = torch.cat(activations[name]['out'])

            # Flatten [Batch, Seq_Len, Hidden_Dim] -> [Batch * Seq_Len, Hidden_Dim]
            # Or [Batch, Hidden_Dim] -> [Batch, Hidden_Dim]
            in_flat = in_acts.view(-1, in_acts.size(-1))
            out_flat = out_acts.view(-1, out_acts.size(-1))

            # Compute cosine similarity for each token/item in the batch
            # ρ_i = (x_in ⋅ x_out) / (||x_in|| * ||x_out||)
            cos_sim = F.cosine_similarity(in_flat, out_flat, dim=1)
            
            # Average over the batch
            avg_cos_sim = cos_sim.mean().item()
            
            # BI score s_i = 1 - ρ_i
            bi_scores[name] = 1.0 - avg_cos_sim
            
        except Exception as e:
            logger.error(f"Error computing BI score for layer {name}: {e}")
            bi_scores[name] = 0.0 # Default to 0 if computation fails

    model.train() # Set model back to training mode
    return bi_scores
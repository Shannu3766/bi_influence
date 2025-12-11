import os
import csv
from typing import Dict, Any
import torch
from peft.tuners.lora import LoraLayer
import logging

logger = logging.getLogger(__name__)

def get_lora_layers(model: torch.nn.Module) -> Dict[str, LoraLayer]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoraLayer)
    }

def save_epoch_log(
    log_file: str, 
    epoch: int, 
    ranks: Dict[str, int], 
    scores: Dict[str, float]
):
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    fieldnames = ['epoch', 'layer_name', 'importance_score', 'allocated_rank']
    
    file_exists = os.path.isfile(log_file)
    
    try:
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            for layer_name in ranks.keys():
                writer.writerow({
                    'epoch': epoch,
                    'layer_name': layer_name,
                    'importance_score': scores.get(layer_name, 0.0),
                    'allocated_rank': ranks.get(layer_name, 0)
                })
    except IOError as e:
        logger.error(f"Failed to write to log file {log_file}: {e}")
import os
import logging
import torch
from peft.tuners.lora import LoraLayer
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from .importance import compute_bi_scores
from .allocation import allocate_ranks_bi
from .utils import get_lora_layers, save_epoch_log

logger = logging.getLogger(__name__)

# ============================================================
# 🔧 HELPER: SVD-Based Layer Resizing with Padding
# ============================================================
def resize_lora_layer_svd(
    layer: LoraLayer, 
    new_rank: int, 
    lora_alpha: int, 
    adapter_name: str = "default",
    **kwargs
):
    """
    Resizes a LoRA layer using SVD to preserve learned weights.
    - Decreasing Rank: Uses SVD truncation (Compression).
    - Increasing Rank: Uses SVD + Zero Padding (Growth).
    """
    # 1. Capture current weights and scaling before modification
    with torch.no_grad():
        if adapter_name not in layer.lora_A:
            return
            
        old_r = layer.r[adapter_name]
        
        # Handle edge case if rank is 0
        if old_r == 0: 
             layer.update_layer(adapter_name, new_rank, lora_alpha=lora_alpha, init_lora_weights=True, **kwargs)
             return

        old_alpha = layer.lora_alpha[adapter_name]
        old_scaling = old_alpha / old_r
        
        # Get current weights
        A_old = layer.lora_A[adapter_name].weight
        B_old = layer.lora_B[adapter_name].weight
        
        # Compute the full effective delta weight: W = B @ A * scaling
        W_delta = (B_old @ A_old) * old_scaling
        
        # 2. Perform SVD
        # Process on the same device to prevent overhead/errors
        dtype = A_old.dtype
        U, S, Vh = torch.linalg.svd(W_delta.float(), full_matrices=False)
        
        # 3. Determine how many components to keep
        k = new_rank
        k = min(k, S.size(0)) # Can't keep more components than exist
        
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]
        
        # 4. Calculate new matrices A' and B'
        # Distribute S via sqrt to keep A and B balanced
        sqrt_S = torch.diag(torch.sqrt(S_k))
        B_new = (U_k @ sqrt_S).to(dtype)
        A_new = (sqrt_S @ Vh_k).to(dtype)
        
        # 5. Scaling Correction
        # We need (B_new @ A_new) * new_scaling approx= W_delta
        if new_rank > 0:
            new_scaling = lora_alpha / new_rank
            # Multiply weights by inverse sqrt of new scaling
            scale_correction = 1.0 / (new_scaling ** 0.5)
            B_new *= scale_correction
            A_new *= scale_correction

    # 6. Update Layer Architecture (This resets weights to random/zero)
    if 'init_lora_weights' in kwargs:
        kwargs.pop('init_lora_weights')

    layer.update_layer(
        adapter_name=adapter_name,
        r=new_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True, # Allow init, we overwrite immediately
        **kwargs
    )
    
    # 7. Write Saved Weights Back
    with torch.no_grad():
        device = layer.lora_A[adapter_name].weight.device
        
        if k < new_rank:
             # Case: Rank Increased (Growth)
             # Zero out the new random weights first
             layer.lora_A[adapter_name].weight.data.zero_()
             layer.lora_B[adapter_name].weight.data.zero_()
             
             # Copy preserved weights into the top-left corner (Padding)
             layer.lora_A[adapter_name].weight.data[:k, :] = A_new.to(device)
             layer.lora_B[adapter_name].weight.data[:, :k] = B_new.to(device)
        else:
             # Case: Rank Decreased or Same (Compression)
             layer.lora_A[adapter_name].weight.data = A_new.to(device)
             layer.lora_B[adapter_name].weight.data = B_new.to(device)
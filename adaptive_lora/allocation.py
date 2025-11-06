import torch
from typing import Dict
import logging

# Get a logger
logger = logging.getLogger(__name__)

def allocate_ranks_bi(
    scores: Dict[str, float], 
    total_rank: int, 
    tau: float = 1.0, 
    min_rank: int = 1
) -> Dict[str, int]:
    """
    Allocates ranks to layers based on their BI importance scores using
    a softmax function with temperature.
    
    r_i = R_remaining * Softmax(s_i / τ) + r_min

    Args:
        scores: Dictionary of {layer_name: importance_score}.
        total_rank: The total rank budget R to distribute.
        tau: Temperature parameter τ to sharpen or soften the distribution.
             (τ < 1 sharpens, τ > 1 softens).
        min_rank: The *desired* minimum rank to assign to each layer.

    Returns:
        A dictionary of {layer_name: allocated_rank}.
    """
    if not scores:
        return {}

    num_layers = len(scores)
    
    # --- NEW ROBUSTNESS CHECK ---
    required_min_budget = num_layers * min_rank
    
    if total_rank < required_min_budget:
        # The requested min_rank is impossible with the total_rank budget.
        # We must lower the min_rank to what is possible.
        
        # Calculate the highest possible integer min_rank
        effective_min_rank = max(0, int(torch.floor(torch.tensor(total_rank / num_layers)).item()))
        
        logger.warning(
            f"Total rank {total_rank} is less than the required minimum budget "
            f"({required_min_budget} = {num_layers} layers * {min_rank} min_rank). "
            f"Forcing a lower effective min_rank of {effective_min_rank} for all layers."
        )
        
        # Overwrite the user's requested min_rank with the possible one
        min_rank = effective_min_rank 
        
        if min_rank == 0:
             logger.warning(
                f"Total rank {total_rank} is less than the number of layers ({num_layers}). "
                f"Some layers will be assigned a rank of 0."
             )
    # --- END NEW CHECK ---


    if total_rank < num_layers * min_rank:
        # This check is now redundant but kept as a final safeguard.
        # It should never be hit if the logic above is correct.
        raise ValueError(
            f"Total rank {total_rank} is less than the minimum required "
            f"({num_layers} layers * {min_rank} min_rank = {num_layers * min_rank})."
        )

    layer_names = list(scores.keys())
    s = torch.tensor([scores[name] for name in layer_names], dtype=torch.float32)

    # Calculate remaining budget after assigning min_rank to all
    remaining_budget = total_rank - (num_layers * min_rank)

    if remaining_budget == 0:
        # No budget left, just return min_rank for all
        return {name: min_rank for name in layer_names}

    # --- Softmax allocation for the remaining budget ---
    # r_i = R_remaining * Softmax(s_i / τ)
    s_temp = s / tau
    probs = torch.softmax(s_temp, dim=0)
    
    # Get raw rank allocations for the remaining budget
    raw_ranks_to_add = probs * remaining_budget

    # --- Largest Remainder Method for integer rounding ---
    # This ensures the sum is exactly equal to remaining_budget
    int_ranks_to_add = torch.floor(raw_ranks_to_add).int()
    remainder = remaining_budget - int_ranks_to_add.sum()

    if remainder > 0:
        residuals = raw_ranks_to_add - int_ranks_to_add
        # Get indices of the k largest residuals, where k = remainder
        _, top_indices = torch.topk(residuals, k=int(remainder.item()))
        int_ranks_to_add[top_indices] += 1

    # Add the base min_rank back to the allocated ranks
    final_ranks = int_ranks_to_add + min_rank

    # Ensure sum is correct (should be, but as a safeguard)
    if final_ranks.sum() != total_rank:
        # Simple correction if floating point errors caused a mismatch
        diff = total_rank - final_ranks.sum()
        final_ranks[final_ranks.argmax()] += diff
        
    return {name: rank.item() for name, rank in zip(layer_names, final_ranks)}
from peft import get_peft_model, LoraConfig, TaskType
import copy

def _get_base_model(model):
    # If model is a peft wrapper, try to access .base_model or .model
    try:
        base = getattr(model, 'base_model', None) or getattr(model, 'model', None)
        if base is not None:
            return base
    except Exception:
        pass
    return model

def apply_adaptive_lora(model, rank_allocation, target_modules=None, lora_alpha=32, lora_dropout=0.1, task_type=TaskType.SEQ_CLS):
    """Re-initialize LoRA adapters on the base model according to rank_allocation.
    This *recreates* adapters fresh (Option B). Returns the new peft-wrapped model.
    """
    base = _get_base_model(model)
    # create a single r to put into LoraConfig (peft expects scalar r) but we'll keep per-layer info on model._adaptive_lora_ranks
    if not rank_allocation:
        r = max(1,1)
    else:
        r = max(1, min(rank_allocation.values()))
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias='none',
        target_modules=target_modules or ['q_proj','k_proj','v_proj','o_proj','dense'],
        task_type=task_type,
    )
    # Create a fresh PEFT model wrapper
    new_model = get_peft_model(copy.deepcopy(base), lora_config)
    # attach the allocation map for bookkeeping
    new_model._adaptive_lora_ranks = rank_allocation
    return new_model

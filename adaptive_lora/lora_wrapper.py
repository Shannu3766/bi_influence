from peft import get_peft_model, LoraConfig, TaskType
import copy

def apply_adaptive_lora(model, rank_allocation, target_modules=None, lora_alpha=32, lora_dropout=0.1, task_type=TaskType.SEQ_CLS):
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'dense']
    base_model = getattr(model, 'base_model', model)
    r = max(1, min(rank_allocation.values()) if rank_allocation else 1)
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias='none',
        target_modules=target_modules,
        task_type=task_type,
    )
    new_model = get_peft_model(copy.deepcopy(base_model), config)
    new_model._adaptive_lora_ranks = rank_allocation
    return new_model

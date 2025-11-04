from peft import get_peft_model, LoraConfig, TaskType

def apply_adaptive_lora(model, rank_allocation, target_modules=None, lora_alpha=32, lora_dropout=0.1, task_type=TaskType.SEQ_CLS):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "dense"]
    base_r = max(1, min(rank_allocation.values()))
    lora_config = LoraConfig(
        r=base_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=task_type,
    )
    model = get_peft_model(model, lora_config)
    model._adaptive_lora_ranks = rank_allocation
    return model

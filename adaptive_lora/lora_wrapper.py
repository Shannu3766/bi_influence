# Placeholder: LoRA application logic (reuse from previous version)

"""
Integrate adaptive ranks with PEFT LoRA configuration.
"""
from peft import get_peft_model, LoraConfig, TaskType


def apply_adaptive_lora(model, rank_allocation, target_modules=None, lora_alpha=32,
                        lora_dropout=0.1, task_type=TaskType.SEQ_CLS):
    """
    Apply LoRA modules with adaptive ranks.

    Args:
        model: transformer model
        rank_allocation: dict {layer_name: rank}
        target_modules: list[str] of module name substrings
        lora_alpha, lora_dropout: LoRA hyperparams
        task_type: e.g., TaskType.SEQ_CLS, TaskType.CAUSAL_LM

    Returns:
        model wrapped with LoRA adapters
    """
    if target_modules is None:
        # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "dense", "ff", "wi", "wo"]
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "dense"]

    # use smallest rank as base (PEFT expects uniform r)
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
    model._adaptive_lora_ranks = rank_allocation  # for reference
    return model

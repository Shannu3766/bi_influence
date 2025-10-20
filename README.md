adalora_bi
===========
BI-based Adaptive LoRA (Algorithm 2) with dynamic per-epoch reallocation.

Usage:
 - compute BI scores: compute_bi_importance_from_dataloader(model, dataloader, device)
 - allocate ranks: bi_allocate_ranks(scores, R, tau)
 - inject adapters: inject_adaptive_lora(model, module_names, ranks)
 - dynamic fine-tuning: fine_tune_lora_dynamic(...)

Example:
  python adalora_bi/examples/finetune_bi_demo.py

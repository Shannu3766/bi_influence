"""
End-to-end demo:

1) Load a HF model & tokenizer (DistilBERT used as example).
2) Create small train/val dataloaders from datasets.
3) Compute BI scores using compute_bi_importance_from_dataloader (on val_loader).
4) Allocate ranks using bi_allocate_ranks.
5) Inject LoRA per-module using inject_adaptive_lora.
6) Fine-tune only LoRA parameters using fine_tune_lora.

Run:
    python examples/finetune_bi_demo.py
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from adalora_bi import (
    compute_bi_importance_from_dataloader,
    bi_allocate_ranks,
    inject_adaptive_lora,
    fine_tune_lora
)

def collate_fn(batch, tokenizer, max_length=128):
    texts = [b["text"] for b in batch]
    labels = [b["label"] for b in batch]
    toks = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    toks["labels"] = torch.tensor(labels, dtype=torch.long)
    return toks

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    # Prepare dataset small subsets for demo
    ds_train = load_dataset("ag_news", split="train[:0.5%]")
    ds_val = load_dataset("ag_news", split="test[:0.5%]")

    train_loader = DataLoader(ds_train, batch_size=8, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(ds_val, batch_size=8, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

    # Step 1: compute BI scores on val set (cheap: max_batches small)
    print("Computing BI importance scores (this may take time depending on model/dataloaders)...")
    module_names, scores = compute_bi_importance_from_dataloader(
        model, val_loader, device=device, target_module_name_substrings=None, max_batches=16
    )
    print("Monitored modules (first 20):", module_names[:20])
    print("Raw BI scores (first 20):", scores[:20])

    # Step 2: allocate ranks
    total_R = 64
    tau = 0.5
    ranks = bi_allocate_ranks(scores, total_R, tau=tau)
    # Pair and show
    for n, r in zip(module_names, ranks):
        print(f"{n} -> r={r}")

    # Step 3: inject LoRA
    patched = inject_adaptive_lora(model, module_names, ranks, alpha=16, dropout=0.0)
    print("Patched modules count:", len(patched))

    # Step 4: fine-tune only LoRA params
    print("Starting fine-tuning LoRA adapters...")
    fine_tune_lora(model, train_loader, val_loader=val_loader, device=device, epochs=2, lr=5e-4, save_path="adalora_bi_checkpoint.pt")

    print("Done.")

if __name__ == "__main__":
    main()

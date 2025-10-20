import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from .importance import compute_bi_importance_from_dataloader
from .allocation import bi_allocate_ranks
from .lora_injector import inject_adaptive_lora

def freeze_base_model(model: nn.Module):
    for n, p in model.named_parameters():
        p.requires_grad = False

def enable_lora_params(model: nn.Module):
    for module in model.modules():
        if module.__class__.__name__ == "LoRALinear":
            if hasattr(module, "A") and module.A is not None:
                module.A.requires_grad = True
            if hasattr(module, "B") and module.B is not None:
                module.B.requires_grad = True

def get_lora_parameters(model: nn.Module):
    params = []
    for n, p in model.named_parameters():
        # heuristic: LoRA weights A and B are trainable parameters inside LoRALinear
        if p.requires_grad:
            params.append(p)
    return params

def evaluate(model: nn.Module, dataloader, device: str):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            loss_sum += loss.item() * labels.size(0)
    return {"loss": loss_sum / total if total>0 else 0.0, "accuracy": correct / total if total>0 else 0.0}

def fine_tune_lora_dynamic(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str = "cuda",
    total_R: int = 64,
    tau: float = 0.5,
    epochs: int = 3,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    alpha: int = 16,
    dropout: float = 0.0,
    max_batches_for_bi: int = 8,
    log_every: int = 10,
    save_path: Optional[str] = None
):
    """
    Dynamic training loop. Recomputes BI and reallocates ranks at the start of every epoch.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # freeze base and ensure LoRA params are trainable when injected
    freeze_base_model(model)

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        # 1) Recompute BI importance and allocate ranks
        print("Computing BI importance (using current model state)...")
        module_names, scores = compute_bi_importance_from_dataloader(
            model, val_loader, device=device, target_module_name_substrings=None, max_batches=max_batches_for_bi
        )
        print("BI scores collected for", len(module_names), "modules")
        ranks = bi_allocate_ranks(scores, total_R, tau=tau)
        print("Allocated ranks (sample first 20):")
        for n, r in zip(module_names[:20], ranks[:20]):
            print(f"  {n} -> r={r}")

        # 2) Inject or re-inject LoRA wrappers with new ranks
        patched = inject_adaptive_lora(model, module_names, ranks, alpha=alpha, dropout=dropout)
        print(f"Patched {len(patched)} modules with LoRA adapters.")

        # 3) enable LoRA params
        enable_lora_params(model)
        lora_params = get_lora_parameters(model)
        if len(lora_params) == 0:
            raise RuntimeError("No LoRA parameters found to train; check injection step.")
        optimizer = optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

        # 4) Train one epoch (only LoRA params)
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % log_every == 0:
                avg = running_loss / log_every
                print(f"Epoch {epoch+1} step {step+1} avg loss {avg:.4f}")
                running_loss = 0.0

        # 5) Validate
        stats = evaluate(model, val_loader, device)
        print(f"After epoch {epoch+1} validation: loss={stats['loss']:.4f}, acc={stats['accuracy']:.4f}")

        # 6) Optionally save
        if save_path:
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    return model

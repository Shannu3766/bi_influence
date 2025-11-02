import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from .importance import compute_bi_importance_from_dataloader
from .allocation import bi_allocate_ranks
from .lora_injector import inject_adaptive_lora
from tqdm import tqdm
import torch.nn.functional as F


def freeze_base_model(model: nn.Module):
    for _, p in model.named_parameters():
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
    for _, p in model.named_parameters():
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
            # Universal loss handler: works for classification + causal LM
            if logits.dim() == 3:
            # e.g. GPT-2, T5, DeepSeek
                loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=100
                )
            else:
                # e.g. DistilBERT classification
                loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=-1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            loss_sum += loss.item() * labels.size(0)
    return {
        "loss": loss_sum / total if total > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
    }


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
    max_batches_for_bi: int = 4,
    log_every: int = 10,
    save_path: Optional[str] = None,
    recompute_every: int = 1,
    fast_mode: bool = True,
):
    """
    Dynamic fine-tuning of LoRA with BI-based rank allocation.

    Fast mode limits BI computation to fewer layers and fewer batches.
    recompute_every controls how often to recompute BI (every N epochs).
    """

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    freeze_base_model(model)

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        # 1️⃣ Recompute BI importance & allocate ranks
        if epoch % recompute_every == 0:
            print("Computing BI importance...")
            start_time = time.time()
            module_names, scores = compute_bi_importance_from_dataloader(
                model,
                val_loader,
                device=device,
                target_module_name_substrings=None,
                max_batches=max_batches_for_bi,
                fast_mode=fast_mode,
            )
            elapsed = time.time() - start_time
            print(
                f"BI scores collected for {len(module_names)} modules in {elapsed:.2f}s"
            )

            ranks = bi_allocate_ranks(scores, total_R, tau=tau)
            print("Allocated ranks (first few):")
            for n, r in list(zip(module_names, ranks))[:10]:
                print(f"  {n} -> r={r}")

            patched = inject_adaptive_lora(
                model, module_names, ranks, alpha=alpha, dropout=dropout
            )
            print(f"Patched {len(patched)} modules with LoRA adapters.")
            # Ensure LoRA adapters are moved to same device
            model.to(device)   

        # 2️⃣ Enable LoRA params
        enable_lora_params(model)
        lora_params = get_lora_parameters(model)
        optimizer = optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

        # 3️⃣ Train one epoch
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch+1}")):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if (step + 1) % log_every == 0:
        avg_loss = running_loss / log_every
        print(f"Epoch {epoch+1} step {step+1} avg loss {avg_loss:.4f}")
        running_loss = 0.0

        # 4️⃣ Validation
        stats = evaluate(model, val_loader, device)
        print(
            f"After epoch {epoch+1}: val loss={stats['loss']:.4f}, acc={stats['accuracy']:.4f}"
        )

        # 5️⃣ Optional save
        if save_path:
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    return model

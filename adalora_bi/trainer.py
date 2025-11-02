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


# ---------------------- Utility functions ----------------------

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
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------- Evaluation ----------------------

def evaluate(model: nn.Module, dataloader, device: str):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**{k: v for k, v in batch.items() if k not in ["labels", "start_positions", "end_positions"]})

            # === Classification ===
            if "labels" in batch:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = criterion(logits, batch["labels"])
                preds = logits.argmax(dim=-1)
                total += batch["labels"].size(0)
                correct += (preds == batch["labels"]).sum().item()

            # === QA ===
            elif "start_positions" in batch and "end_positions" in batch:
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                start_loss = criterion(start_logits, batch["start_positions"])
                end_loss = criterion(end_logits, batch["end_positions"])
                loss = (start_loss + end_loss) / 2
                total += batch["start_positions"].size(0)
                correct += 0  # accuracy doesn’t apply to QA

            # === Causal LM ===
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                labels = batch.get("labels", None)
                if labels is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=100
                    )
                else:
                    loss = torch.tensor(0.0)

            loss_sum += loss.item() * next(iter(batch.values())).size(0)

    return {
        "loss": loss_sum / total if total > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
    }


# ---------------------- Main Training ----------------------

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
    Now supports both Classification and Question Answering tasks.
    """

    model.to(device)
    freeze_base_model(model)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        # ----------------- 1️⃣ Recompute BI Importance -----------------
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
            print(f"BI scores collected for {len(module_names)} modules in {elapsed:.2f}s")

            ranks = bi_allocate_ranks(scores, total_R, tau=tau)
            print("Allocated ranks (first few):")
            for n, r in list(zip(module_names, ranks))[:10]:
                print(f"  {n} -> r={r}")

            patched = inject_adaptive_lora(model, module_names, ranks, alpha=alpha, dropout=dropout)
            print(f"Patched {len(patched)} modules with LoRA adapters.")
            model.to(device)

        # ----------------- 2️⃣ Enable LoRA Params -----------------
        enable_lora_params(model)
        optimizer = optim.AdamW(get_lora_parameters(model), lr=lr, weight_decay=weight_decay)

        # ----------------- 3️⃣ Train -----------------
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**{k: v for k, v in batch.items() if k not in ["labels", "start_positions", "end_positions"]})

            # --- Classification ---
            if "labels" in batch:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = criterion(logits, batch["labels"])

            # --- Question Answering ---
            elif "start_positions" in batch and "end_positions" in batch:
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                start_loss = criterion(start_logits, batch["start_positions"])
                end_loss = criterion(end_logits, batch["end_positions"])
                loss = (start_loss + end_loss) / 2

            # --- Causal LM ---
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
                labels = batch.get("labels", None)
                if labels is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=100
                    )
                else:
                    continue
            else:
                raise ValueError("Batch missing required labels for loss computation.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (step + 1) % log_every == 0:
                print(f"Step {step+1}: avg_loss = {running_loss / log_every:.4f}")
                running_loss = 0.0

        # ----------------- 4️⃣ Validation -----------------
        stats = evaluate(model, val_loader, device)
        print(f"After epoch {epoch+1}: val_loss={stats['loss']:.4f}, val_acc={stats['accuracy']:.4f}")

        # ----------------- 5️⃣ Save -----------------
        if save_path:
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    return model

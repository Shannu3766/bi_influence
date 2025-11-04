from transformers import TrainerCallback
from torch.utils.data import DataLoader
import torch
from .bi_score import compute_bi_scores

class AdaptiveLoRACallback(TrainerCallback):
    def __init__(self, r_min=1, tau=0.5, total_rank=None, val_subset_size=50, val_batch_size=2):
        self.r_min = r_min
        self.tau = tau
        self.total_rank = total_rank
        self.val_subset_size = val_subset_size
        self.val_batch_size = val_batch_size

    def on_train_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None or not trainer.is_world_process_zero():
            return control

        print("\n[Adaptive LoRA] Forced BI computation at training end...")

        eval_dataset = trainer.eval_dataset
        try:
            if hasattr(eval_dataset, "select"):
                small = eval_dataset.select(range(min(self.val_subset_size, len(eval_dataset))))
            else:
                small = eval_dataset[:min(self.val_subset_size, len(eval_dataset))]
        except Exception:
            small = eval_dataset

        val_loader = DataLoader(small, batch_size=self.val_batch_size)
        device = trainer.args.device if hasattr(trainer.args, "device") else ("cuda" if torch.cuda.is_available() else "cpu")

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        _ = compute_bi_scores(trainer.model, dataloader=val_loader, device=device, total_rank=self.total_rank or 64, tau=self.tau, r_min=self.r_min)

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        print("[Adaptive LoRA] ✅ BI Scores and Ranks computed and displayed successfully.\n")
        return control

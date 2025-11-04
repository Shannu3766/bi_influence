from transformers import TrainerCallback
from torch.utils.data import DataLoader
import copy, torch
from .bi_score import compute_bi_scores
from .rank_allocator import allocate_ranks_softmax
from .lora_wrapper import apply_adaptive_lora


class AdaptiveLoRACallback(TrainerCallback):
    """
    Callback to recompute BI scores and apply adaptive LoRA ranks during training.

    Supports configurable recompute intervals, smoothing, and one-time computation.
    """

    def __init__(
        self,
        r_min=1,
        tau=0.5,
        total_rank=None,
        recompute_interval=1,
        val_subset_size=50,
        val_batch_size=2,
        smoothing_alpha=0.8,
        compute_once=False,
    ):
        self.r_min = r_min
        self.tau = tau
        self.total_rank = total_rank
        self.recompute_interval = recompute_interval
        self.smoothing_alpha = smoothing_alpha
        self._prev_bi = None
        self.val_subset_size = val_subset_size
        self.val_batch_size = val_batch_size
        self.compute_once = compute_once
        self._has_computed_once = False

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return control

        # Only main process
        if not trainer.is_world_process_zero():
            return control

        epoch = int(state.epoch or 0)

        # Skip recompute if not due yet
        if self.compute_once and self._has_computed_once:
            return control
        if epoch % self.recompute_interval != 0:
            return control

        print(
            f"\\n[Adaptive LoRA] Rank recompute triggered at epoch {epoch} "
            f"(interval={self.recompute_interval}, compute_once={self.compute_once})"
        )

        # Subset evaluation dataset to limit memory
        eval_dataset = trainer.eval_dataset
        try:
            if hasattr(eval_dataset, "select"):
                small = eval_dataset.select(range(min(self.val_subset_size, len(eval_dataset))))
            else:
                small = eval_dataset[:min(self.val_subset_size, len(eval_dataset))]
        except Exception:
            small = eval_dataset

        val_loader = DataLoader(small, batch_size=self.val_batch_size)
        model = trainer.model
        device = (
            trainer.args.device if hasattr(trainer.args, "device")
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Compute new BI scores
        new_bi = compute_bi_scores(model, tokenizer=trainer.tokenizer, dataloader=val_loader, device=device)

        # Smooth scores for stability
        if self._prev_bi is not None:
            for k in new_bi:
                new_bi[k] = self.smoothing_alpha * self._prev_bi.get(k, new_bi[k]) + (1 - self.smoothing_alpha) * new_bi[k]

        self._prev_bi = copy.deepcopy(new_bi)

        # Allocate ranks using softmax
        ranks = allocate_ranks_softmax(
            new_bi,
            total_rank=self.total_rank or max(64, len(new_bi)),
            tau=self.tau,
            r_min=self.r_min,
        )

        # Apply LoRA adapters
        trainer.model = apply_adaptive_lora(trainer.model, ranks)

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        print(
            f"[Adaptive LoRA] Applied ranks for {len(ranks)} layers "
            f"(r_min={self.r_min}, tau={self.tau}, total_rank={self.total_rank or max(64, len(new_bi))})"
        )

        if self.compute_once:
            self._has_computed_once = True

        return control

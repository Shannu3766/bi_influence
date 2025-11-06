from transformers import TrainerCallback
from torch.utils.data import DataLoader
import csv, os, torch, copy, time
from .bi_score import compute_bi_scores
from .rank_allocator import allocate_ranks_softmax
from .lora_wrapper import apply_adaptive_lora

class AdaptiveLoRACallback(TrainerCallback):
    def __init__(self, r_min=1, tau=0.5, total_rank=64, recompute_interval=1, val_subset_size=50, val_batch_size=2,
                 smoothing_alpha=0.8, target_modules=None, log_path='adaptive_lora_epoch_logs.csv'):
        self.r_min = r_min
        self.tau = tau
        self.total_rank = total_rank
        self.recompute_interval = recompute_interval
        self.val_subset_size = val_subset_size
        self.val_batch_size = val_batch_size
        self.smoothing_alpha = smoothing_alpha
        self.target_modules = target_modules or ['q_proj','k_proj','v_proj','o_proj','dense']
        self._prev_bi = None
        self.log_path = log_path

        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'layer', 'BI', 'Rank'])

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get('trainer', None)
        if trainer is None or not trainer.is_world_process_zero():
            return control
        epoch = int(state.epoch or 0)
        total_epochs = int(getattr(trainer.args, 'num_train_epochs', 0))
        if epoch % self.recompute_interval != 0:
            return control
        print(f"\n[Adaptive LoRA] Epoch {epoch}/{total_epochs}: computing BI, reallocating ranks, and logging CSV...")

        eval_dataset = trainer.eval_dataset
        try:
            if hasattr(eval_dataset, 'select'):
                small = eval_dataset.select(range(min(self.val_subset_size, len(eval_dataset))))
            else:
                small = eval_dataset[:min(self.val_subset_size, len(eval_dataset))]
        except Exception:
            small = eval_dataset
        val_loader = DataLoader(small, batch_size=self.val_batch_size)

        device = trainer.args.device if hasattr(trainer.args, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

        start = time.time()
        bi = compute_bi_scores(trainer.model, dataloader=val_loader, device=device)
        print(f"[Adaptive LoRA] BI computed in {time.time()-start:.1f}s for {len(bi)} layers.")

        if self._prev_bi:
            for k in bi:
                bi[k] = self.smoothing_alpha * self._prev_bi.get(k, bi[k]) + (1-self.smoothing_alpha) * bi[k]
        self._prev_bi = copy.deepcopy(bi)

        ranks = allocate_ranks_softmax(bi, total_rank=self.total_rank, tau=self.tau, r_min=self.r_min)

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for k,v in ranks.items():
                writer.writerow([epoch, k, round(bi[k],6), v])

        print('[Adaptive LoRA] Reinitializing LoRA adapters fresh for next epoch...')
        new_model = apply_adaptive_lora(trainer.model, ranks, target_modules=self.target_modules)
        trainer.model = new_model
        trainer.optimizer = None
        trainer.lr_scheduler = None

        if device.startswith('cuda'):
            torch.cuda.empty_cache()

        print(f"[Adaptive LoRA] ✅ Epoch {epoch} complete. BI + Ranks logged to {self.log_path}.\n")
        return control

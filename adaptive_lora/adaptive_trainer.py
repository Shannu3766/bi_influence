from transformers import Trainer
from torch.utils.data import DataLoader
from .bi_score import compute_bi_scores
from .rank_allocator import allocate_ranks_softmax
from .lora_wrapper import apply_adaptive_lora
import copy

class AdaptiveTrainer(Trainer):
    def __init__(self, *args, r_min=1, tau=0.5, total_rank=None, recompute_interval=1, smoothing_alpha=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_min = r_min
        self.tau = tau
        self.total_rank = total_rank
        self.recompute_interval = recompute_interval
        self.smoothing_alpha = smoothing_alpha
        self._prev_bi_scores = None

    def _recompute_adaptive_ranks(self, epoch):
        print(f"[Adaptive LoRA] Recomputing ranks after epoch {epoch}...")
        val_loader = DataLoader(self.eval_dataset, batch_size=4)
        new_bi = compute_bi_scores(self.model, tokenizer=self.tokenizer, dataloader=val_loader, device=self.args.device)
        if self._prev_bi_scores is not None:
            for k in new_bi:
                new_bi[k] = self.smoothing_alpha * self._prev_bi_scores.get(k, new_bi[k]) + (1 - self.smoothing_alpha) * new_bi[k]
        self._prev_bi_scores = copy.deepcopy(new_bi)
        ranks = allocate_ranks_softmax(new_bi, total_rank=self.total_rank, tau=self.tau, r_min=self.r_min)
        self.model = apply_adaptive_lora(self.model, ranks)
        print(f"[Adaptive LoRA] Updated ranks for {len(ranks)} layers.")
        return ranks

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if self.state.epoch and int(self.state.epoch) % self.recompute_interval == 0:
            self._recompute_adaptive_ranks(int(self.state.epoch))
        return loss

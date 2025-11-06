import os
import logging
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from torch.utils.data import DataLoader
from .importance import compute_bi_scores
from .allocation import allocate_ranks_bi
from .utils import get_lora_layers, save_epoch_log

logger = logging.getLogger(__name__)

class AdaptiveLoRACallback(TrainerCallback):
    """
    A Hugging Face TrainerCallback that implements per-epoch adaptive
    LoRA rank allocation based on Block Influence (BI) scores (Algorithm 2).
    """

    def __init__(
        self,
        total_rank: int,
        val_dataloader: DataLoader,
        tau: float = 1.0,
        log_path: str = "./logs",
        verbose: bool = True
    ):
        """
        Args:
            total_rank: The total rank budget R to distribute.
            val_dataloader: A DataLoader for a (small) subset of the validation data
                            used to compute BI scores.
            tau: Temperature for softmax allocation.
            log_path: Directory to save the CSV logs.
            verbose: If True, prints a summary of rank changes each epoch.
        """
        self.total_rank = total_rank
        self.val_dataloader = val_dataloader
        self.tau = tau
        self.log_file = os.path.join(log_path, "adaptive_lora_epoch_logs.csv")
        self.verbose = verbose

        # Ensure log directory exists
        if log_path and not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

    # ============================================================
    # 🔁 EPOCH-END HOOK
    # ============================================================
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,  # This 'model' is the PEFT model
        **kwargs
    ):
        """
        Called at the end of each epoch to perform rank adaptation.
        """
        epoch = int(state.epoch)
        if self.verbose:
            print(f"\n--- AdaptiveLoRA: Starting rank update for Epoch {epoch} ---")

        device = next(model.parameters()).device

        # ============================================================
        # 1️⃣ Compute BI Scores (Algorithm 2, lines 4–11)
        # ============================================================
        if self.verbose:
            print("Computing BI importance scores...")

        scores = compute_bi_scores(model, self.val_dataloader, device)
        if not scores:
            if self.verbose:
                print("No LoRA layers found or BI scores computed. Skipping update.")
            return

        # ============================================================
        # 2️⃣ Allocate Ranks (Algorithm 2, lines 12–14)
        # ============================================================
        if self.verbose:
            print("Allocating new ranks...")

        new_ranks = allocate_ranks_bi(scores, self.total_rank, self.tau)

        # ============================================================
        # 3️⃣ Update LoRA Adapter Modules
        # ============================================================
        if self.verbose:
            print("Applying new ranks to LoRA modules...")

        lora_layers = get_lora_layers(model)

        # Retrieve PEFT configuration
        config = model.peft_config.get('default')
        if not config:
            logger.error("Could not find 'default' PEFT config. Skipping update.")
            return

        # Gather LoRA configuration parameters
        init_lora_weights = getattr(config, 'init_lora_weights', True)
        use_rslora = getattr(config, 'use_rslora', False)
        use_dora = getattr(config, 'use_dora', False)
        use_qalora = getattr(config, 'use_qalora', False)
        lora_bias = getattr(config, 'bias', 'none')
        qalora_group_size = getattr(config, 'qalora_group_size', 64)

        # Iterate over LoRA layers and update their ranks
        for name, layer in lora_layers.items():
            new_rank = new_ranks.get(name)
            if new_rank is None:
                logger.warning(f"No new rank allocated for layer {name}. Skipping.")
                continue

            current_rank = layer.r.get('default', 0)

            if current_rank != new_rank:
                if self.verbose:
                    print(f"  - {name}: r={current_rank} -> {new_rank} "
                          f"(Score: {scores.get(name, 0):.4f})")

                # =====================================================
                # Handle rank = 0 (disable LoRA)
                # =====================================================
                if new_rank == 0:
                    # Set the rank to zero; LoRA will be bypassed automatically
                    layer.r['default'] = 0

                    # Optionally, zero out LoRA weights and freeze them
                    try:
                        for p_name, p in layer.named_parameters(recurse=False):
                            if 'lora' in p_name or 'A' in p_name or 'B' in p_name:
                                p.data.zero_()
                                p.requires_grad = False
                    except Exception as e:
                        logger.warning(f"Warning: Failed to zero LoRA params in {name}: {e}")

                # =====================================================
                # Handle rank > 0 (enable/update LoRA)
                # =====================================================
                else:
                    # Determine dropout probability if available
                    lora_dropout_p = 0.0
                    if hasattr(layer, 'lora_dropout') and 'default' in layer.lora_dropout:
                        lora_dropout_p = layer.lora_dropout['default'].p

                    # Update layer with new rank configuration
                    if hasattr(layer, 'update_layer'):
                        layer.update_layer(
                            adapter_name='default',
                            r=new_rank,
                            lora_alpha=layer.lora_alpha.get('default', 1),
                            lora_dropout=lora_dropout_p,
                            init_lora_weights=init_lora_weights,
                            use_rslora=use_rslora,
                            use_dora=use_dora,
                            use_qalora=use_qalora,
                            lora_bias=lora_bias,
                            qalora_group_size=qalora_group_size
                        )
                    else:
                        # Fallback if update_layer doesn't exist
                        layer.r['default'] = new_rank
            else:
                if self.verbose:
                    print(f"  - {name}: r={new_rank} (Unchanged)")

        # ============================================================
        # 4️⃣ Log Results
        # ============================================================
        save_epoch_log(self.log_file, epoch, new_ranks, scores)

        if self.verbose:
            print(f"--- AdaptiveLoRA: Update complete. Logs saved to {self.log_file} ---")

# import os
# import logging
# from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
# from torch.utils.data import DataLoader
# from .importance import compute_bi_scores
# from .allocation import allocate_ranks_bi
# from .utils import get_lora_layers, save_epoch_log

# logger = logging.getLogger(__name__)

# class AdaptiveLoRACallback(TrainerCallback):
#     """
#     A Hugging Face TrainerCallback that implements per-epoch adaptive
#     LoRA rank allocation based on Block Influence (BI) scores.
#     """

#     def __init__(
#         self,
#         total_rank: int,
#         val_dataloader: DataLoader,
#         tau: float = 1.0,
#         log_path: str = "./logs",
#         verbose: bool = True
#     ):
#         self.total_rank = total_rank
#         self.val_dataloader = val_dataloader
#         self.tau = tau
#         self.log_file = os.path.join(log_path, "adaptive_lora_epoch_logs.csv")
#         self.verbose = verbose

#         if log_path and not os.path.exists(log_path):
#             os.makedirs(log_path, exist_ok=True)

#     # ============================================================
#     # 🔁 EPOCH-END HOOK
#     # ============================================================
#     def on_epoch_end(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         model,
#         **kwargs
#     ):
#         epoch = int(state.epoch)
#         if self.verbose:
#             print(f"\n--- AdaptiveLoRA: Starting rank update for Epoch {epoch} ---")

#         device = next(model.parameters()).device

#         # 1️⃣ Compute BI scores
#         if self.verbose:
#             print("Computing BI importance scores...")
#         scores = compute_bi_scores(model, self.val_dataloader, device)
#         if not scores:
#             if self.verbose:
#                 print("No LoRA layers found or BI scores computed. Skipping update.")
#             return

#         # 2️⃣ Allocate ranks
#         if self.verbose:
#             print("Allocating new ranks...")
#         new_ranks = allocate_ranks_bi(scores, self.total_rank, self.tau)

#         # 3️⃣ Update LoRA modules
#         if self.verbose:
#             print("Applying new ranks to LoRA modules...")

#         lora_layers = get_lora_layers(model)
#         config = model.peft_config.get("default")
#         if not config:
#             logger.error("Could not find 'default' PEFT config. Skipping update.")
#             return

#         init_lora_weights = getattr(config, "init_lora_weights", True)
#         use_rslora = getattr(config, "use_rslora", False)
#         use_dora = getattr(config, "use_dora", False)
#         use_qalora = getattr(config, "use_qalora", False)
#         lora_bias = getattr(config, "bias", "none")
#         qalora_group_size = getattr(config, "qalora_group_size", 64)

#         for name, layer in lora_layers.items():
#             new_rank = new_ranks.get(name)
#             if new_rank is None:
#                 logger.warning(f"No new rank allocated for layer {name}. Skipping.")
#                 continue

#             current_rank = layer.r.get("default", 0)

#             if current_rank != new_rank:
#                 if self.verbose:
#                     print(
#                         f"  - {name}: r={current_rank} -> {new_rank} "
#                         f"(Score: {scores.get(name, 0):.4f})"
#                     )

#                 # Handle rank = 0  ➜ effectively disable LoRA
#                 if new_rank == 0:
#                     layer.r["default"] = 0
#                     try:
#                         for p_name, p in layer.named_parameters(recurse=False):
#                             if "lora" in p_name or "A" in p_name or "B" in p_name:
#                                 p.data.zero_()
#                                 p.requires_grad = False
#                     except Exception as e:
#                         logger.warning(
#                             f"Warning: Failed to zero LoRA params in {name}: {e}"
#                         )

#                 # Handle rank > 0  ➜ enable/update LoRA
#                 else:
#                     lora_dropout_p = 0.0
#                     if hasattr(layer, "lora_dropout") and "default" in layer.lora_dropout:
#                         lora_dropout_p = layer.lora_dropout["default"].p

#                     if hasattr(layer, "update_layer"):
#                         layer.update_layer(
#                             adapter_name="default",
#                             r=new_rank,
#                             lora_alpha=layer.lora_alpha.get("default", 1),
#                             lora_dropout=lora_dropout_p,
#                             init_lora_weights=init_lora_weights,
#                             use_rslora=use_rslora,
#                             use_dora=use_dora,
#                             use_qalora=use_qalora,
#                             lora_bias=lora_bias,
#                             qalora_group_size=qalora_group_size,
#                         )
#                     else:
#                         layer.r["default"] = new_rank
#             else:
#                 if self.verbose:
#                     print(
#                         f"  - {name}: r={new_rank} (rank Unchanged, Score: {scores.get(name, 0):.4f})"
#                     )

#         # 4️⃣ Log epoch summary
#         save_epoch_log(self.log_file, epoch, new_ranks, scores)
#         if self.verbose:
#             print(
#                 f"--- AdaptiveLoRA: Update complete. Logs saved to {self.log_file} ---"
#             )


import os
import logging
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from .importance import compute_bi_scores
from .allocation import allocate_ranks_bi
from .utils import get_lora_layers, save_epoch_log

logger = logging.getLogger(__name__)

class AdaptiveLoRACallback(TrainerCallback):
    """
    Adaptive LoRA that computes BI scores *before each epoch*,
    allocates new ranks, and trains with updated LoRA parameters.
    """

    def __init__(
        self,
        total_rank: int,
        val_dataloader,
        tau: float = 1.0,
        log_path: str = "./logs",
        verbose: bool = True,
    ):
        self.total_rank = total_rank
        self.val_dataloader = val_dataloader
        self.tau = tau
        self.verbose = verbose
        self.log_file = os.path.join(log_path, "adaptive_lora_epoch_logs.csv")

        os.makedirs(log_path, exist_ok=True)

    # ============================================================
    # 🔁 EPOCH-BEGIN: compute & apply ranks before training
    # ============================================================
    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs
    ):
        epoch = int(state.epoch) + 1  # epoch starts at 0 before training
        if self.verbose:
            print(f"\n--- AdaptiveLoRA: Preparing ranks for Epoch {epoch} ---")

        device = next(model.parameters()).device

        # 1️⃣ Compute BI scores BEFORE training
        if self.verbose:
            print("Computing BI importance scores (pre-training)...")
        scores = compute_bi_scores(model, self.val_dataloader, device)
        if not scores:
            if self.verbose:
                print("No LoRA layers or scores found. Skipping update.")
            return

        # 2️⃣ Allocate new ranks
        if self.verbose:
            print("Allocating ranks based on BI scores...")
        new_ranks = allocate_ranks_bi(scores, self.total_rank, self.tau)

        # 3️⃣ Apply ranks to LoRA layers immediately
        if self.verbose:
            print("Applying ranks to LoRA modules for this epoch...")

        lora_layers = get_lora_layers(model)
        config = model.peft_config.get("default")
        if not config:
            logger.error("Could not find PEFT config. Skipping update.")
            return

        init_lora_weights = getattr(config, "init_lora_weights", True)
        use_rslora = getattr(config, "use_rslora", False)
        use_dora = getattr(config, "use_dora", False)
        use_qalora = getattr(config, "use_qalora", False)
        lora_bias = getattr(config, "bias", "none")
        qalora_group_size = getattr(config, "qalora_group_size", 64)

        for name, layer in lora_layers.items():
            new_rank = new_ranks.get(name)
            if new_rank is None:
                continue

            current_rank = layer.r.get("default", 0)

            if current_rank != new_rank:
                if self.verbose:
                    print(
                        f"  - {name}: r={current_rank} -> {new_rank} "
                        f"(Score: {scores.get(name, 0):.4f})"
                    )

                if hasattr(layer, "update_layer"):
                    lora_dropout_p = 0.0
                    if hasattr(layer, "lora_dropout") and "default" in layer.lora_dropout:
                        lora_dropout_p = layer.lora_dropout["default"].p

                    layer.update_layer(
                        adapter_name="default",
                        r=new_rank,
                        lora_alpha=layer.lora_alpha.get("default", 1),
                        lora_dropout=lora_dropout_p,
                        init_lora_weights=init_lora_weights,
                        use_rslora=use_rslora,
                        use_dora=use_dora,
                        use_qalora=use_qalora,
                        lora_bias=lora_bias,
                        qalora_group_size=qalora_group_size,
                    )
            else:
                if self.verbose:
                    print(f"  - {name}: r={new_rank} (Unchanged, Score: {scores.get(name, 0):.4f})")

        # Save for logging after training
        self.latest_scores = scores
        self.latest_ranks = new_ranks
        if self.verbose:
            print(f"--- AdaptiveLoRA: Rank setup for Epoch {epoch} complete ---")

    # ============================================================
    # 📊 EPOCH-END: just log what we used
    # ============================================================
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs
    ):
        epoch = int(state.epoch)
        if hasattr(self, "latest_ranks") and hasattr(self, "latest_scores"):
            save_epoch_log(self.log_file, epoch, self.latest_ranks, self.latest_scores)
            if self.verbose:
                print(f"✅ Epoch {epoch}: Rank allocations logged to {self.log_file}")

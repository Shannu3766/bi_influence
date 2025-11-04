# Placeholder: BI score computation logic (reuse from previous version)

"""
Compute Block Influence (BI) scores for transformer layers.
"""

import torch
from tqdm import tqdm


def _flatten_tensor(t):
    return t.detach().cpu().reshape(t.shape[0], -1)


def compute_bi_scores(model, tokenizer=None, dataloader=None, device="cuda"):
    """
    Compute BI scores for transformer blocks in `model` using a dataloader (or iterable of inputs).

    Args:
        model: a Hugging Face transformers model (nn.Module)
        tokenizer: optional, used if dataloader yields raw text
        dataloader: an iterable yielding batches (dicts of tensors) or raw texts
        device: device string ("cuda" or "cpu")

    Returns:
        dict mapping layer_name -> BI score (float)
    """
    model.to(device)
    model.eval()

    # Identify blocks by heuristic (layer, block, encoder, decoder, etc.)
    block_names = []
    for name, _ in model.named_modules():
        if any(x in name.lower() for x in ["layer", "block", "encoder", "decoder"]):
            block_names.append(name)
    block_names = list(dict.fromkeys(block_names))  # dedupe

    activations_in = {n: [] for n in block_names}
    activations_out = {n: [] for n in block_names}
    hooks = []

    def hook_fn(name):
        def fn(mod, inp, outp):
            inp = inp[0] if isinstance(inp, (tuple, list)) else inp
            activations_in[name].append(_flatten_tensor(inp))
            activations_out[name].append(_flatten_tensor(outp))
        return fn

    for n in block_names:
        mod = dict(model.named_modules()).get(n, None)
        if mod is not None:
            hooks.append(mod.register_forward_hook(hook_fn(n)))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Adaptive LoRA] Computing BI"):
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            _ = model(**batch)

    for h in hooks:
        h.remove()

    bi_scores = {}
    for n in block_names:
        if not activations_in[n] or not activations_out[n]:
            continue
        Xin = torch.cat(activations_in[n], dim=0).float()
        Xout = torch.cat(activations_out[n], dim=0).float()
        m = min(Xin.shape[0], Xout.shape[0])
        Xin, Xout = Xin[:m], Xout[:m]
        cos = (Xin * Xout).sum(dim=1) / (
            (Xin.norm(p=2, dim=1) * Xout.norm(p=2, dim=1)).clamp(min=1e-8)
        )
        bi_scores[n] = float(1 - cos.mean().item())
    return bi_scores

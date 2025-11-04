import torch
import numpy as np
from tqdm import tqdm

def _flatten_tensor(t):
    return t.detach().cpu().reshape(t.shape[0], -1)

def compute_bi_scores(model, dataloader=None, device="cuda", total_rank=64, tau=0.5, r_min=1):
    """
    Computes Block Influence (BI) scores for attention layers and
    allocates LoRA ranks automatically. Prints BI + rank per layer.
    """
    model.to(device)
    model.eval()

    # Collect only attention-like layers
    block_names = []
    for name, _ in model.named_modules():
        if any(x in name.lower() for x in ["self_attn", "attention", "attn"]):
            block_names.append(name)
    block_names = list(dict.fromkeys(block_names))

    activations_in = {n: [] for n in block_names}
    activations_out = {n: [] for n in block_names}
    hooks = []

    def hook_fn(name):
        def fn(mod, inp, outp):
            if inp is None or (isinstance(inp, (tuple, list)) and len(inp) == 0):
                return
            if isinstance(inp, (tuple, list)):
                inp = inp[0]
            if not torch.is_tensor(inp) or not torch.is_tensor(outp):
                return
            try:
                activations_in[name].append(_flatten_tensor(inp))
                activations_out[name].append(_flatten_tensor(outp))
            except Exception:
                pass
        return fn

    # Register hooks
    for n in block_names:
        mod = dict(model.named_modules()).get(n, None)
        if mod is not None:
            hooks.append(mod.register_forward_hook(hook_fn(n)))

    # Forward pass
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Adaptive LoRA] Computing BI scores"):
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            _ = model(**batch)

    for h in hooks:
        h.remove()

    # Compute BI for each layer
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
        bi = float(1 - cos.mean().item())
        bi_scores[n] = bi

    # Allocate ranks based on BI scores
    names = list(bi_scores.keys())
    s = np.array([bi_scores[n] for n in names], dtype=float)
    s = s - s.max()
    weights = np.exp(s / tau)
    weights /= weights.sum()
    r_float = weights * total_rank
    r_int = np.floor(r_float).astype(int)
    r_int = np.maximum(r_int, r_min)

    current = r_int.sum()
    residuals = r_float - r_int
    if current < total_rank:
        for i in np.argsort(-residuals)[: total_rank - current]:
            r_int[i] += 1
    elif current > total_rank:
        for i in np.argsort(residuals)[: current - total_rank]:
            if r_int[i] > r_min:
                r_int[i] -= 1

    # Print BI + rank summary
    print("\n[Adaptive LoRA] ---- Layer-wise BI Score & Rank ----")
    for i, name in enumerate(names):
        print(f"  • {name:<60s}  BI = {bi_scores[name]:.6f}   →   Rank = {r_int[i]:>3d}")
    print(f"Total allocated rank = {r_int.sum()} / {total_rank}")
    print("[Adaptive LoRA] ---------------------------------------\n")

    return {names[i]: (bi_scores[names[i]], int(r_int[i])) for i in range(len(names))}

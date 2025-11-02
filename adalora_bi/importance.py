import torch
import numpy as np


def _make_pre_hook(storage, name):
    def pre_hook(module, inputs):
        x = inputs[0].detach().cpu()
        vec = x.mean(dim=tuple(range(0, x.ndim - 1))) if x.ndim > 1 else x
        storage[name]["in"].append(vec.view(-1))

    return pre_hook


def _make_forward_hook(storage, name):
    def forward_hook(module, inputs, output):
        x = output.detach().cpu()
        vec = x.mean(dim=tuple(range(0, x.ndim - 1))) if x.ndim > 1 else x
        storage[name]["out"].append(vec.view(-1))

    return forward_hook


def compute_bi_importance_from_dataloader(
    model,
    dataloader,
    device,
    target_module_name_substrings=None,
    max_batches=8,
    fast_mode=False,
):
    """
    Compute BI scores between input and output activations of Linear layers.

    fast_mode=True → only sample key modules (q_proj, v_proj, fc) for speed.
    """
    model.to(device)
    model.eval()

    # Select candidate Linear modules
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if fast_mode:
                # Only a few key modules
                if any(sub in name for sub in ["q_proj", "v_proj", "out_proj", "fc"]):
                    modules.append((name, module))
            elif target_module_name_substrings:
                if any(sub in name for sub in target_module_name_substrings):
                    modules.append((name, module))
            else:
                if any(
                    tok in name
                    for tok in ("q_proj", "k_proj", "v_proj", "out_proj", "fc", "dense")
                ):
                    modules.append((name, module))

    if fast_mode and not modules:
        # fallback if model uses different naming
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules.append((name, module))
                count += 1
                if count >= 10:
                    break
    elif not modules:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules.append((name, module))
        modules = modules[:20]

    module_names = [n for n, _ in modules]
    storage = {n: {"in": [], "out": []} for n in module_names}

    # Register hooks
    hooks = []
    for name, module in modules:
        hooks.append(module.register_forward_pre_hook(_make_pre_hook(storage, name)))
        hooks.append(module.register_forward_hook(_make_forward_hook(storage, name)))

    # Run forward passes
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch_on_device = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            try:
                _ = model(**batch_on_device)
            except TypeError:
                kwargs = {
                    k: batch_on_device[k]
                    for k in ("input_ids", "attention_mask")
                    if k in batch_on_device
                }
                _ = model(**kwargs)

    for h in hooks:
        h.remove()

    # Compute BI scores
    scores = []
    for name in module_names:
        xin = storage[name]["in"]
        xout = storage[name]["out"]
        M = min(len(xin), len(xout))
        if M == 0:
            scores.append(0.0)
            continue
        per_batch = []
        for j in range(M):
            a = xin[j].detach().to(torch.float32).cpu().numpy().ravel()
            b = xout[j].detach().to(torch.float32).cpu().numpy().ravel()
            n = min(a.size, b.size)
            a, b = a[:n], b[:n]
            rho = float((a * b).sum()) / (
                np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
            )
            per_batch.append(1 - rho)
        scores.append(float(np.mean(per_batch)))

    scores = np.array(scores)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    return module_names, scores.tolist()

import torch
import numpy as np

def _make_pre_hook(storage, name):
    def pre_hook(module, inputs):
        x = inputs[0]
        # handle tensor or tuple/list of tensors
        if isinstance(x, torch.Tensor):
            x_cpu = x.detach().to(torch.float32).cpu()
            vec = x_cpu.mean(dim=tuple(range(0, x_cpu.ndim - 1))) if x_cpu.ndim > 1 else x_cpu
            storage[name]["in"].append(vec.view(-1))
        elif isinstance(x, (list, tuple)):
            parts = []
            for item in x:
                if isinstance(item, torch.Tensor):
                    it = item.detach().to(torch.float32).cpu()
                    v = it.mean(dim=tuple(range(0, it.ndim - 1))) if it.ndim > 1 else it
                    parts.append(v.view(-1))
            if parts:
                storage[name]["in"].append(torch.cat(parts))
    return pre_hook

def _make_forward_hook(storage, name):
    def forward_hook(module, inputs, output):
        x = output
        if isinstance(x, torch.Tensor):
            x_cpu = x.detach().to(torch.float32).cpu()
            vec = x_cpu.mean(dim=tuple(range(0, x_cpu.ndim - 1))) if x_cpu.ndim > 1 else x_cpu
            storage[name]["out"].append(vec.view(-1))
        elif isinstance(x, (list, tuple)):
            parts = []
            for item in x:
                if isinstance(item, torch.Tensor):
                    it = item.detach().to(torch.float32).cpu()
                    v = it.mean(dim=tuple(range(0, it.ndim - 1))) if it.ndim > 1 else it
                    parts.append(v.view(-1))
            if parts:
                storage[name]["out"].append(torch.cat(parts))
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

    This implementation:
    - Temporarily casts the model parameters to float32 for the BI forward pass.
    - Safely converts any captured activations to float32 on CPU before using numpy.
    - Handles tuple/list outputs from hooks.
    """
    model.to(device)
    model.eval()

    # Select candidate Linear modules
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if fast_mode:
                if any(sub in name for sub in ["q_proj", "v_proj", "out_proj", "fc"]):
                    modules.append((name, module))
            elif target_module_name_substrings:
                if any(sub in name for sub in target_module_name_substrings):
                    modules.append((name, module))
            else:
                if any(tok in name for tok in ("q_proj", "k_proj", "v_proj", "out_proj", "fc", "dense")):
                    modules.append((name, module))

    # fallback if none matched
    if not modules:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules.append((name, module))
                if fast_mode and len(modules) >= 10:
                    break

    module_names = [n for n, _ in modules]
    storage = {n: {"in": [], "out": []} for n in module_names}

    # Register hooks
    hooks = []
    for name, module in modules:
        hooks.append(module.register_forward_pre_hook(_make_pre_hook(storage, name)))
        hooks.append(module.register_forward_hook(_make_forward_hook(storage, name)))

    # Save original dtype & device mapping to restore later
    # We assume model parameters share the same device; capture dtype from first param if present
    params = list(model.parameters())
    orig_dtype = params[0].dtype if params else torch.get_default_dtype()
    orig_devices = [p.device for p in params] if params else []

    # Temporarily cast model params to float32 for BI computation
    try:
        # Move to float32 in-place
        for p in model.parameters():
            p.data = p.data.to(torch.float32)

        # Forward pass (collect activations)
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

    finally:
        # Always remove hooks and restore original dtype (and device)
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        # Attempt to cast parameters back to original dtype
        if params:
            for p, d in zip(params, orig_devices):
                # moving back to original device and dtype
                p.data = p.data.to(device=d, dtype=orig_dtype)

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
            # xin/xout entries are CPU float32 tensors (from hooks)
            a = xin[j].numpy().ravel() if isinstance(xin[j], torch.Tensor) else np.array(xin[j]).ravel()
            b = xout[j].numpy().ravel() if isinstance(xout[j], torch.Tensor) else np.array(xout[j]).ravel()
            n = min(a.size, b.size)
            if n == 0:
                continue
            a, b = a[:n], b[:n]
            # cosine-based measure (1 - rho)
            rho = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            per_batch.append(1 - rho)
        scores.append(float(np.mean(per_batch)) if per_batch else 0.0)

    scores = np.array(scores)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    return module_names, scores.tolist()

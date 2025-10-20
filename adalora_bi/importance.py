import torch
import numpy as np
from collections import defaultdict

def _make_pre_hook(storage, name):
    def pre_hook(module, inputs):
        x = inputs[0].detach().cpu()
        if x.ndim > 1:
            vec = x.mean(dim=tuple(range(0, x.ndim - 1)))
        else:
            vec = x.view(-1)
        storage[name]["in"].append(vec.view(-1))
    return pre_hook

def _make_forward_hook(storage, name):
    def forward_hook(module, inputs, output):
        x = output.detach().cpu()
        if x.ndim > 1:
            vec = x.mean(dim=tuple(range(0, x.ndim - 1)))
        else:
            vec = x.view(-1)
        storage[name]["out"].append(vec.view(-1))
    return forward_hook

def compute_bi_importance_from_dataloader(
    model,
    dataloader,
    device,
    target_module_name_substrings=None,
    max_batches=32
):
    """
    Compute BI importance scores using forward_pre_hook and forward_hook.
    Returns (module_names, scores) where scores are normalized to [0,1] (or zeros if no variation).
    """
    model.to(device)
    model.eval()

    # select candidate Linear modules
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if target_module_name_substrings:
                if any(sub in name for sub in target_module_name_substrings):
                    modules.append((name, module))
            else:
                if any(tok in name for tok in ("q_proj", "k_proj", "v_proj", "out_proj", "fc", "dense", "lin", "proj")):
                    modules.append((name, module))
    if not modules:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules.append((name, module))
        modules = modules[:20]

    module_names = [n for n, _ in modules]
    storage = {name: {"in": [], "out": []} for name in module_names}
    hooks = []

    for name, module in modules:
        hooks.append(module.register_forward_pre_hook(_make_pre_hook(storage, name)))
        hooks.append(module.register_forward_hook(_make_forward_hook(storage, name)))

    num_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            # move tensors to device
            to_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    to_device[k] = v.to(device)
                else:
                    to_device[k] = v
            # robust forward
            try:
                _ = model(**to_device)
            except TypeError:
                # try common inputs
                kwargs = {}
                for k in ("input_ids", "attention_mask", "inputs_embeds"):
                    if k in to_device:
                        kwargs[k] = to_device[k]
                _ = model(**kwargs)
            num_batches += 1

    for h in hooks:
        h.remove()

    scores = []
    for name in module_names:
        xin_list = storage[name]["in"]
        xout_list = storage[name]["out"]
        M = min(len(xin_list), len(xout_list))
        if M == 0:
            scores.append(0.0)
            continue
        row_scores = []
        for j in range(M):
            xin = xin_list[j].numpy().ravel()
            xout = xout_list[j].numpy().ravel()
            minlen = min(xin.size, xout.size)
            xin = xin[:minlen]
            xout = xout[:minlen]
            num = float((xin * xout).sum())
            den = (np.linalg.norm(xin) * np.linalg.norm(xout) + 1e-12)
            rho = num / den
            row_scores.append(1.0 - rho)
        scores.append(float(np.mean(row_scores)))

    scores = np.array(scores, dtype=float)
    if scores.max() - scores.min() > 1e-12:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return module_names, scores.tolist()

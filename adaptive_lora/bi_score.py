import torch
from tqdm import tqdm

def _flatten_tensor(t):
    return t.detach().cpu().reshape(t.shape[0], -1)

def compute_bi_scores(model, dataloader=None, device='cuda'):
    model.to(device)
    model.eval()
    block_names = []
    for name, _ in model.named_modules():
        if any(x in name.lower() for x in ['self_attn','attention','attn']):
            block_names.append(name)
    block_names = list(dict.fromkeys(block_names))

    activations_in = {n: [] for n in block_names}
    activations_out = {n: [] for n in block_names}
    hooks = []

    def hook_fn(name):
        def fn(mod, inp, outp):
            try:
                if isinstance(inp, (tuple, list)) and len(inp) > 0:
                    inp = inp[0]
                if not torch.is_tensor(inp) or not torch.is_tensor(outp):
                    return
                activations_in[name].append(_flatten_tensor(inp))
                activations_out[name].append(_flatten_tensor(outp))
            except Exception:
                return
        return fn

    for n in block_names:
        mod = dict(model.named_modules()).get(n, None)
        if mod is not None:
            try:
                hooks.append(mod.register_forward_hook(hook_fn(n)))
            except Exception:
                pass

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='[Adaptive LoRA] Computing BI scores'):
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            try:
                _ = model(**batch)
            except Exception:
                _ = model(**{k:v for k,v in batch.items() if k != 'labels'})

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    bi_scores = {}
    for n in block_names:
        ins = activations_in.get(n, [])
        outs = activations_out.get(n, [])
        if not ins or not outs:
            continue
        Xin = torch.cat(ins, dim=0).float()
        Xout = torch.cat(outs, dim=0).float()
        m = min(Xin.shape[0], Xout.shape[0])
        Xin, Xout = Xin[:m], Xout[:m]
        cos = (Xin * Xout).sum(dim=1) / ((Xin.norm(p=2, dim=1) * Xout.norm(p=2, dim=1)).clamp(min=1e-8))
        bi = float(1 - cos.mean().item())
        bi_scores[n] = bi

    print('\n[Adaptive LoRA] Computed BI scores for', len(bi_scores), 'layers.')
    return bi_scores

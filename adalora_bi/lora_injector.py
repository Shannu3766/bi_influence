import torch
import torch.nn as nn
import math
from typing import List

class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear. Keeps original Linear as submodule.
    Forward: orig(x) + scaling * (x @ A.T @ B.T)
    A: r x in_features, B: out_features x r
    """
    def __init__(self, orig_linear: nn.Linear, r: int = 4, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(orig_linear, nn.Linear)
        self.orig = orig_linear
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if self.r > 0:
            self.A = nn.Parameter(torch.zeros(self.r, orig_linear.in_features))
            self.B = nn.Parameter(torch.zeros(orig_linear.out_features, self.r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        out = self.orig(x)
        if self.r > 0:
            lora_mid = torch.matmul(x, self.A.t())
            lora_out = torch.matmul(lora_mid, self.B.t())
            lora_out = self.dropout(lora_out) * self.scaling
            out = out + lora_out
        return out

def _find_module_by_name(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None, None, None
        parent = getattr(parent, p)
    attr = parts[-1]
    if not hasattr(parent, attr):
        return None, None, None
    return parent, attr, getattr(parent, attr)

def inject_adaptive_lora(model: nn.Module, module_names: List[str], ranks: List[int], alpha: int = 16, dropout: float = 0.0, replace_linear=True) -> List[str]:
    """
    Replace Linear modules with LoRALinear using given ranks.
    Returns list of patched module names.
    """
    assert len(module_names) == len(ranks)
    patched = []
    for name, r in zip(module_names, ranks):
        parent, attr, module = _find_module_by_name(model, name)
        if module is None:
            found = False
            for n, m in model.named_modules():
                if name in n and isinstance(m, nn.Linear):
                    parent, attr, module = _find_module_by_name(model, n)
                    found = True
                    break
            if not found:
                continue
        if isinstance(module, nn.Linear) and replace_linear:
            wrapped = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, wrapped)
            patched.append(name)
        else:
            child_patched = False
            for child_name, child_module in module.named_modules():
                if isinstance(child_module, nn.Linear):
                    child_parent, child_attr, _ = _find_module_by_name(module, child_name)
                    wrapped = LoRALinear(child_module, r=r, alpha=alpha, dropout=dropout)
                    setattr(child_parent, child_attr, wrapped)
                    patched.append(f"{name}.{child_name}")
                    child_patched = True
                    break
            if not child_patched:
                continue
    return patched

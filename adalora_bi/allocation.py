import numpy as np
from .utils import softmax, allocate_from_raw

def bi_allocate_ranks(scores, R, tau=1.0):
    """
    Algorithm 2 BI-based allocation.
    args:
      - scores: list or np.array of per-layer BI scores (si)
      - R: total LoRA rank budget
      - tau: temperature
    returns:
      - ranks: list of ints (length len(scores)), sum(ranks)==R, rmin implicitly 1
    """
    scores = np.array(scores, dtype=float)
    if scores.size == 0:
        return []
    weights = softmax(scores, tau)   # α_i
    raw = R * weights                # R * α_i
    ranks = allocate_from_raw(raw, R)
    return ranks

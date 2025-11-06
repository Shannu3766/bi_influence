import numpy as np

def allocate_ranks_softmax(bi_scores, total_rank=64, tau=0.5, r_min=1):
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

    print('\n[Adaptive LoRA] ---- Rank Allocation ----')
    for i, name in enumerate(names):
        print(f'  • {name:<60s} Rank = {r_int[i]:>3d}')
    print(f'Total allocated rank = {r_int.sum()} / {total_rank}')
    print('[Adaptive LoRA] --------------------------------\n')

    return {names[i]: int(r_int[i]) for i in range(len(names))}

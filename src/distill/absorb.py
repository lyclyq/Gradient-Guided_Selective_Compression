import torch
from src.signals.projection import randomized_low_rank_proj

def capture_ratio(X, r:int):
    if X is None:
        return 0.0
    proj = randomized_low_rank_proj(X, r)
    num = torch.linalg.norm(proj).item() ** 2
    den = torch.linalg.norm(X).item() ** 2 + 1e-12
    return float(num / den)

def absorb_if_mature(adapter, r:int, tau_cap:float=0.7, alpha:float=0.7):
    """If alt is mature (capture ratio high), absorb some into AB."""
    if (not adapter.use_alt) or (not adapter.use_ab):
        return 0.0
    with torch.no_grad():
        alt = adapter.deltaW_alt()
        cap = capture_ratio(alt, r)
        if cap >= tau_cap:
            adapter.absorb_alt_into_ab(r=r, alpha=alpha)
        return cap

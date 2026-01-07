import torch
from .projection import randomized_low_rank_proj

def dir_consistency(mats):
    """Directional consistency in [0,1]. mats: list of [out,in] tensors."""
    if len(mats) == 0:
        return 0.0
    vecs = []
    for M in mats:
        v = M.reshape(-1)
        n = torch.linalg.norm(v) + 1e-12
        vecs.append(v / n)
    V = torch.stack(vecs, dim=0)  # [V, d]
    mean = V.mean(dim=0)
    cons = torch.linalg.norm(mean).clamp(0, 1).item()
    return cons

def compute_consistency_votes(G_votes, r:int, R:int):
    """Return (C_ab, C_perp, C_R) computed from votes."""
    proj_r = [randomized_low_rank_proj(G, r) for G in G_votes]
    C_ab = dir_consistency(proj_r)
    # residual consistency: G - proj_r
    resid = [G - Pr for G, Pr in zip(G_votes, proj_r)]
    C_perp = dir_consistency(resid)
    proj_R = [randomized_low_rank_proj(G, R) for G in G_votes]
    C_R = dir_consistency(proj_R)
    return C_ab, C_perp, C_R

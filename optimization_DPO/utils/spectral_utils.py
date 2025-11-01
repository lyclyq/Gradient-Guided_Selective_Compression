# spectral_utils.py (additions; keep existing API)

import torch, math

class StableCountSketchLspec:
    """
    与 CountSketchLspec 等价的用途，但:
    - 对每个维度 d 缓存一次 (idx, sgn)，保证跨次调用一致
    - 当 m <= d 时无放回采样；m > d 时允许有放回补齐
    """
    def __init__(self, m=64, seed: int = 12345):
        self.m = int(m)
        self.seed = int(seed)
        self._cache = {}  # d -> (idx, sgn)

    @torch.no_grad()
    def _get_proj(self, device, d: int):
        if d in self._cache:
            idx, sgn = self._cache[d]
            return idx.to(device), sgn.to(device)
        g = torch.Generator(device=device)
        g.manual_seed(self.seed + d)  # 维度相关扰动，避免同 seed 同分布
        m = self.m
        if d >= m:
            # 无放回
            perm = torch.randperm(d, generator=g, device=device)
            idx = perm[:m]
        else:
            # 有放回补齐
            idx = torch.randint(0, d, (m,), generator=g, device=device)
        # ±1 Rademacher
        sgn = torch.where(torch.rand(m, generator=g, device=device) < 0.5,
                          torch.tensor(-1.0, device=device),
                          torch.tensor(+1.0, device=device))
        self._cache[d] = (idx.cpu(), sgn.cpu())
        return idx, sgn

    @torch.no_grad()
    def _proj(self, v: torch.Tensor):
        x = v.reshape(-1)
        d = x.numel()
        m = self.m
        if d == 0:
            return torch.zeros(m, device=v.device)
        idx, sgn = self._get_proj(v.device, d)
        return (x[idx] * sgn) / math.sqrt(m)

    @torch.no_grad()
    def lspec(self, g1: torch.Tensor, g2: torch.Tensor):
        z1 = self._proj(g1); z2 = self._proj(g2)
        cos = torch.nn.functional.cosine_similarity(z1, z2, dim=0).clamp(-1,1)
        return float(1.0 - cos.item())


@torch.no_grad()
def collect_flat_grads_with_offsets(model):
    flats, params, offsets = [], [], []
    off = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().reshape(-1)
        flats.append(g); params.append(p); offsets.append(off)
        off += g.numel()
    if not flats:
        device = next(model.parameters()).device
        return torch.tensor([], device=device), [], []
    return torch.cat(flats, dim=0), params, offsets

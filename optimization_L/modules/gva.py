
import torch
from collections import defaultdict

class GVAProjector:
    """
    Gradient Vector Alignment (GVA) with partial projection against a cached EMA validation gradient.
    - Grouping: list of (group_id, [params])
    - We store EMA(g_val) per group to reduce quickval noise.
    """
    def __init__(self, beta=0.9, tau=0.0, eta_proj=0.2, delta_max=0.3, eps=1e-8):
        self.beta = beta
        self.tau = tau
        self.eta_proj = eta_proj
        self.delta_max = delta_max
        self.eps = eps
        self._gval_ema = {}  # gid -> 1D tensor

    @staticmethod
    def _flatten_group_grad(params):
        vecs = []
        for p in params:
            if p.grad is None:
                continue
            vecs.append(p.grad.detach().reshape(-1))
        if not vecs:
            return None
        return torch.cat(vecs, dim=0)

    def cache_val_grad(self, groups):
        """
        Read current p.grad (assumed to be grads from a val micro-batch) and update EMA cache by group.
        """
        for gid, plist in groups:
            gv = self._flatten_group_grad(plist)
            if gv is None:
                continue
            if gid in self._gval_ema:
                self._gval_ema[gid] = self.beta * self._gval_ema[gid] + (1 - self.beta) * gv
            else:
                self._gval_ema[gid] = gv.clone()

    def project_train_grads(self, groups):
        """
        After loss.backward() on train batch (so p.grad holds g_tr), call this to apply partial projection
        against cached EMA(g_val). Returns stats dict for logging.
        """
        stats = {"num_groups": 0, "num_proj": 0, "cos_mean": None}
        cos_list = []

        for gid, plist in groups:
            # Flatten train grad
            gtr = self._flatten_group_grad(plist)
            gva = self._gval_ema.get(gid, None)

            if gtr is None or gva is None:
                continue

            ntr = gtr.norm()
            nva = gva.norm()
            if ntr < self.eps or nva < self.eps:
                continue

            c = torch.dot(gtr, gva)
            cos = (c / (ntr * nva + self.eps)).clamp(-1.0, 1.0).item()
            cos_list.append(cos)
            stats["num_groups"] += 1

            if c.item() < self.tau:
                # partial projection to remove opposite component along g_val
                coeff = c / (nva * nva + self.eps)
                delta = - self.eta_proj * coeff * gva
                # cap step
                max_shift = self.delta_max * ntr
                dnorm = delta.norm()
                if dnorm > max_shift:
                    delta = delta * (max_shift / (dnorm + self.eps))
                # scatter back
                offset = 0
                for p in plist:
                    if p.grad is None:
                        continue
                    sz = p.grad.numel()
                    p.grad.add_(delta[offset:offset+sz].view_as(p.grad))
                    offset += sz
                stats["num_proj"] += 1

        if cos_list:
            stats["cos_mean"] = float(sum(cos_list) / len(cos_list))
        return stats

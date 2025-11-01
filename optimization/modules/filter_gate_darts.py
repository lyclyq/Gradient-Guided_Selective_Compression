# modules/filter_gate_darts.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# def _init_depthwise_like_identity(w):
#     try:
#         nn.init.dirac_(w)
#     except Exception:
#         nn.init.kaiming_uniform_(w, a=1.0)
# 替换 _init_depthwise_like_identity
def _init_depthwise_like_identity(w):
    # w: [C, 1, K] (Conv1d) 或 [C,1,K,K] (Conv2d)
    with torch.no_grad():
        w.zero_()
        shape = w.shape
        if w.dim() == 3:  # 1D
            K = shape[-1]
            if K >= 3:
                base = torch.tensor([1,2,1], dtype=w.dtype, device=w.device)
                base = F.pad(base, ( (K-3)//2, K-3-(K-3)//2 ), value=0)
                base = base / base.sum()
                for c in range(shape[0]):
                    w[c, 0, :] = base
            else:
                w.uniform_(-1e-2, 1e-2)
        elif w.dim() == 4:  # 2D
            K = shape[-1]
            if K >= 3:
                g = torch.outer(torch.tensor([1,2,1], dtype=w.dtype, device=w.device),
                                torch.tensor([1,2,1], dtype=w.dtype, device=w.device))
                g = g / g.sum()
                g = F.pad(g, ( (K-3)//2, K-3-(K-3)//2, (K-3)//2, K-3-(K-3)//2 ))
                for c in range(shape[0]):
                    w[c, 0, :, :] = g
            else:
                w.uniform_(-1e-2, 1e-2)
        w.add_(0.01 * torch.randn_like(w))  # 轻噪声

class FilterGateDARTS1D(nn.Module):
    def __init__(self, channels: int, ks_list=(3,7,15), tau: float = 1.0):
        super().__init__()
        self.ks_list = tuple(int(k)|1 for k in ks_list)
        self.tau = float(tau)
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, k, padding=k//2, groups=channels, bias=False)
            for k in self.ks_list
        ])
        # self.alpha_logits = nn.Parameter(torch.zeros(len(self.ks_list)))
        self.alpha_logits = nn.Parameter(0.01 * torch.randn(len(self.ks_list)))

        for conv in self.convs:
            _init_depthwise_like_identity(conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ch = x.transpose(1, 2)  # [B,T,C] -> [B,C,T]
        w = F.softmax(self.alpha_logits / max(self.tau, 1e-4), dim=0)
        outs = [conv(x_ch) for conv in self.convs]
        y = sum(w[i] * outs[i] for i in range(len(outs)))  # [B,C,T]
        return y.transpose(1, 2)

    @torch.no_grad()
    def set_tau(self, tau: float): self.tau = float(tau)

class FilterGateDARTS2D(nn.Module):
    def __init__(self, channels: int, ks_list=(3,7,15), tau: float = 1.0):
        super().__init__()
        self.ks_list = tuple(int(k)|1 for k in ks_list)
        self.tau = float(tau)
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, k, padding=k//2, groups=channels, bias=False)
            for k in self.ks_list
        ])
        self.alpha_logits = nn.Parameter(torch.zeros(len(self.ks_list)))
        for conv in self.convs:
            _init_depthwise_like_identity(conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.alpha_logits / max(self.tau, 1e-4), dim=0)
        outs = [conv(x) for conv in self.convs]
        return sum(w[i] * outs[i] for i in range(len(outs)))

    @torch.no_grad()
    def set_tau(self, tau: float): self.tau = float(tau)

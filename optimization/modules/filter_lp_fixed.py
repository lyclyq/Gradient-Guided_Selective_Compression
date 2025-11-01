
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LowPassFixed1D", "LowPassFixed2D", "init_lowpass_1d", "init_lowpass_2d"]

def _tri_kernel_1d(K: int, device=None, dtype=None):
    k = torch.arange(K, device=device, dtype=dtype)
    c = (K - 1) / 2.0
    w = (1.0 - (k - c).abs() / (c + 1e-8)).clamp(min=0.0)
    w = w / w.sum().clamp_min(1e-8)
    return w

def _tri_kernel_2d(K: int, device=None, dtype=None):
    w1d = _tri_kernel_1d(K, device=device, dtype=dtype)
    w2d = torch.outer(w1d, w1d)
    w2d = w2d / w2d.sum().clamp_min(1e-8)
    return w2d

def init_lowpass_1d(conv: nn.Conv1d):
    with torch.no_grad():
        K = conv.kernel_size[0]
        base = _tri_kernel_1d(K, device=conv.weight.device, dtype=conv.weight.dtype)
        for c in range(conv.out_channels):
            conv.weight[c, 0, :] = base
        if conv.bias is not None:
            conv.bias.zero_()

def init_lowpass_2d(conv: nn.Conv2d):
    with torch.no_grad():
        K = conv.kernel_size[0]
        base = _tri_kernel_2d(K, device=conv.weight.device, dtype=conv.weight.dtype)
        for c in range(conv.out_channels):
            conv.weight[c, 0, :, :] = base
        if conv.bias is not None:
            conv.bias.zero_()

class _BaseLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_logits = None  # no arch params

    def arch_parameters(self):
        return []
    def weight_parameters(self):
        return list(self.parameters())

    @torch.no_grad()
    def set_tau(self, tau: float):
        return

class LowPassFixed1D(_BaseLP):
    """Depthwise 1D low-pass filter. Kernels are frozen (requires_grad=False)."""
    def __init__(self, channels: int, k: int = 7):
        super().__init__()
        k = int(k) | 1
        self.conv = nn.Conv1d(channels, channels, k, padding=k//2, groups=channels, bias=False)
        init_lowpass_1d(self.conv)
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.transpose(1, 2)
        y = self.conv(xc)
        return y.transpose(1, 2)

class LowPassFixed2D(_BaseLP):
    """Depthwise 2D low-pass filter. Kernels are frozen (requires_grad=False)."""
    def __init__(self, channels: int, k: int = 7):
        super().__init__()
        k = int(k) | 1
        self.conv = nn.Conv2d(channels, channels, k, padding=k//2, groups=channels, bias=False)
        init_lowpass_2d(self.conv)
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

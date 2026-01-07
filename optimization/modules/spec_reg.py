
import torch

__all__ = ["spectral_penalty_depthwise", "phi_linear"]

def _rfft_weight_1d(W: torch.Tensor):
    C, _, K = W.shape
    spec = torch.fft.rfft(W.squeeze(1), dim=-1)  # [C,Kf]
    mag2 = (spec.real**2 + spec.imag**2)         # power spectrum
    return mag2.mean(dim=0)                      # [Kf]

def _rfft_weight_2d(W: torch.Tensor):
    C, _, Kh, Kw = W.shape
    spec = torch.fft.rfftn(W.squeeze(1), dim=(-2,-1))  # [C,Kh,Kf]
    mag2 = (spec.real**2 + spec.imag**2)
    return mag2.mean(dim=(0,1))                        # [Kf]

def phi_linear(Kf: int, device=None, dtype=None, power: float = 1.0):
    w = torch.linspace(0.0, 1.0, Kf, device=device, dtype=dtype) ** power
    return w

def spectral_penalty_depthwise(controller_weights, lam: float = 1e-4, power: float = 1.0):
    total = None
    for W in controller_weights:
        if not isinstance(W, torch.Tensor) or not W.requires_grad:
            continue
        if W.ndim == 3:
            mag2 = _rfft_weight_1d(W)
        elif W.ndim == 4:
            mag2 = _rfft_weight_2d(W)
        else:
            continue
        phi = phi_linear(mag2.numel(), device=W.device, dtype=mag2.dtype, power=power)
        ls = (phi * mag2).sum()
        total = ls if total is None else total + ls
    return torch.tensor(0.0, requires_grad=True) if total is None else lam * total

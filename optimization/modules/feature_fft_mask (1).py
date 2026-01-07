
import torch
import torch.nn as nn

__all__ = ["FeatureFFTMaskWrapper1D", "FeatureFFTMaskWrapper2D", "SpectrumEMA"]

class SpectrumEMA(nn.Module):
    """EMA of magnitude spectra."""
    def __init__(self, beta: float = 0.95):
        super().__init__()
        self.beta = beta
        self.register_buffer("spec", None)

    @torch.no_grad()
    def update_1d(self, x_bt: torch.Tensor):
        x_ct = x_bt.transpose(1,2)
        spec = torch.fft.rfft(x_ct, dim=-1)
        mag = (spec.real**2 + spec.imag**2).sqrt().mean(dim=(0,1))
        self.spec = mag.clone() if self.spec is None else self.spec.mul(self.beta).add((1.0-self.beta)*mag)

    @torch.no_grad()
    def update_2d(self, x_bchw: torch.Tensor):
        spec = torch.fft.rfftn(x_bchw, dim=(-2,-1))
        mag = (spec.real**2 + spec.imag**2).sqrt().mean(dim=(0,1,2))
        self.spec = mag.clone() if self.spec is None else self.spec.mul(self.beta).add((1.0-self.beta)*mag)

    def get(self):
        return None if self.spec is None else self.spec.detach()

def _mask_from_overlap(train_mag: torch.Tensor, val_mag: torch.Tensor, gamma: float, a_min: float):
    eps = 1e-8
    p = train_mag / (train_mag.sum() + eps)
    q = val_mag / (val_mag.sum() + eps)
    s = (p.sqrt() - q.sqrt()).abs()
    A = torch.exp(-gamma * s)
    return torch.clamp(A, min=a_min, max=1.0) if a_min is not None else A

class _BaseMasker(nn.Module):
    def __init__(self, gamma: float = 1.0, a_min: float = 0.8, apply_on: str = "input", use_drv_gate: bool = False):
        super().__init__()
        assert apply_on in ("input","output")
        self.gamma = float(gamma)
        self.a_min = float(a_min)
        self.apply_on = apply_on
        self.use_drv_gate = bool(use_drv_gate)
        self.register_buffer("u_gate", torch.tensor(1.0))
        self.spec_ema = SpectrumEMA(beta=0.95)

    @torch.no_grad()
    def set_gate(self, u: float):
        self.u_gate.fill_(float(u))

    def _scale_gamma(self):
        return self.gamma * (0.5 + float(self.u_gate.item())) if self.use_drv_gate else self.gamma

class FeatureFFTMaskWrapper1D(_BaseMasker):
    def __init__(self, base_module: nn.Module, gamma: float = 1.0, a_min: float = 0.8, apply_on: str = "input", use_drv_gate: bool = False):
        super().__init__(gamma, a_min, apply_on, use_drv_gate)
        self.base = base_module

    def forward(self, x_bt: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.spec_ema.update_1d(x_bt)
            return self.base(x_bt)
        x_ct = x_bt.transpose(1,2)
        spec = torch.fft.rfft(x_ct, dim=-1)
        mag = (spec.real**2 + spec.imag**2).sqrt().mean(dim=(0,1))
        train_mag = self.spec_ema.get()
        if train_mag is None:
            return self.base(x_bt)
        A = _mask_from_overlap(train_mag, mag, gamma=self._scale_gamma(), a_min=self.a_min)
        if self.apply_on == "input":
            spec_att = spec * A.view(1,1,-1)
            x_filt = torch.fft.irfft(spec_att, n=x_ct.shape[-1], dim=-1)
            return self.base(x_filt.transpose(1,2))
        y_bt = self.base(x_bt)
        y_ct = y_bt.transpose(1,2)
        spec_y = torch.fft.rfft(y_ct, dim=-1) * A.view(1,1,-1)
        return torch.fft.irfft(spec_y, n=y_ct.shape[-1], dim=-1).transpose(1,2)

class FeatureFFTMaskWrapper2D(_BaseMasker):
    def __init__(self, base_module: nn.Module, gamma: float = 1.0, a_min: float = 0.8, apply_on: str = "input", use_drv_gate: bool = False):
        super().__init__(gamma, a_min, apply_on, use_drv_gate)
        self.base = base_module

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.spec_ema.update_2d(x_bchw)
            return self.base(x_bchw)
        spec = torch.fft.rfftn(x_bchw, dim=(-2,-1))
        mag = (spec.real**2 + spec.imag**2).sqrt().mean(dim=(0,1,2))
        train_mag = self.spec_ema.get()
        if train_mag is None:
            return self.base(x_bchw)
        A = _mask_from_overlap(train_mag, mag, gamma=self._scale_gamma(), a_min=self.a_min)
        if self.apply_on == "input":
            spec_att = spec * A.view(1,1,1,-1)
            x_filt = torch.fft.irfftn(spec_att, s=x_bchw.shape[-2:], dim=(-2,-1))
            return self.base(x_filt)
        y = self.base(x_bchw)
        spec_y = torch.fft.rfftn(y, dim=(-2,-1)) * A.view(1,1,1,-1)
        return torch.fft.irfftn(spec_y, s=y.shape[-2:], dim=(-2,-1))

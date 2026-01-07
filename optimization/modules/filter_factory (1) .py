
from typing import Sequence, Literal
import torch.nn as nn

try:
    from modules.filter_gate_darts import FilterGateDARTS1D, FilterGateDARTS2D
except Exception:
    FilterGateDARTS1D = FilterGateDARTS2D = None

from modules.filter_lp_fixed import LowPassFixed1D, LowPassFixed2D
from modules.feature_fft_mask import FeatureFFTMaskWrapper1D, FeatureFFTMaskWrapper2D

__all__ = ["make_filter_1d", "make_filter_2d"]

class Identity1D(nn.Module):
    def forward(self, x): return x

class Identity2D(nn.Module):
    def forward(self, x): return x

def make_filter_1d(channels: int,
                   backend: Literal["darts","lp_fixed","none"] = "darts",
                   ks_list: Sequence[int] = (3,7,15),
                   tau: float = 1.0,
                   lp_k: int = 7,
                   feature_mask: bool = False,
                   ff_gamma: float = 1.0,
                   ff_amin: float = 0.8,
                   ff_apply_on: Literal["input","output"] = "input",
                   ff_use_drv_gate: bool = False) -> nn.Module:
    if backend == "darts" and FilterGateDARTS1D is not None:
        base = FilterGateDARTS1D(channels, ks_list=ks_list, tau=tau)
    elif backend == "lp_fixed":
        base = LowPassFixed1D(channels, k=lp_k)
    else:
        base = Identity1D()

    if feature_mask:
        base = FeatureFFTMaskWrapper1D(base_module=base, gamma=ff_gamma, a_min=ff_amin,
                                       apply_on=ff_apply_on, use_drv_gate=ff_use_drv_gate)
    return base

def make_filter_2d(channels: int,
                   backend: Literal["darts","lp_fixed","none"] = "darts",
                   ks_list: Sequence[int] = (3,7,15),
                   tau: float = 1.0,
                   lp_k: int = 7,
                   feature_mask: bool = False,
                   ff_gamma: float = 1.0,
                   ff_amin: float = 0.8,
                   ff_apply_on: Literal["input","output"] = "input",
                   ff_use_drv_gate: bool = False) -> nn.Module:
    if backend == "darts" and FilterGateDARTS2D is not None:
        base = FilterGateDARTS2D(channels, ks_list=ks_list, tau=tau)
    elif backend == "lp_fixed":
        base = LowPassFixed2D(channels, k=lp_k)
    else:
        base = Identity2D()

    if feature_mask:
        base = FeatureFFTMaskWrapper2D(base_module=base, gamma=ff_gamma, a_min=ff_amin,
                                       apply_on=ff_apply_on, use_drv_gate=ff_use_drv_gate)
    return base

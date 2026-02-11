# src/lora_layers.py
from __future__ import annotations

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Baseline LoRA (rank = r):
      y = xW^T + (x A^T B^T) * (alpha/r)

    Where:
      A: [r, in]
      B: [out, r]
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.dropout = float(dropout)

        # baseline scaling: alpha/r
        self.scaling = self.alpha / max(1, self.r)

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Parameter(torch.empty(self.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.r))

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # B already zeros => ΔW starts 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r <= 0:
            return y

        x_d = F.dropout(x, p=self.dropout, training=self.training) if self.dropout > 0 else x
        lora = (x_d @ self.lora_A.t()) @ self.lora_B.t()
        return y + lora * self.scaling


class DualRankLoRALinear(nn.Module):
    """
    Ours: nested dual-rank LoRA

      ΔW_R = B_r A_r + B_hi A_hi
      scaling is UNIFIED at rank-R: (alpha/R)

    Params:
      A_r:  [r, in],   B_r:  [out, r]
      A_hi: [hi, in],  B_hi: [out, hi]
      hi = R - r
    """

    def __init__(self, base: nn.Linear, r: int, R: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)

        self.base = base
        self.r = int(r)
        self.R = int(R)
        self.alpha = float(alpha)
        self.dropout = float(dropout)

        self.hi = max(0, self.R - self.r)

        # ✅ unified scaling (rank-R view)
        self.scaling = self.alpha / max(1, self.R)

        # ✅ per-branch scalings (for trainer/shake_align scale-invariant votes)
        # In this design, both branches live under the same global scale alpha/R.
        self.scaling_r = float(self.scaling)
        self.scaling_hi = float(self.scaling)

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A_r = nn.Parameter(torch.empty(self.r, in_features))
        self.lora_B_r = nn.Parameter(torch.zeros(out_features, self.r))

        self.lora_A_hi = nn.Parameter(torch.empty(self.hi, in_features))
        self.lora_B_hi = nn.Parameter(torch.zeros(out_features, self.hi))

        nn.init.kaiming_uniform_(self.lora_A_r, a=5 ** 0.5)
        nn.init.normal_(self.lora_A_hi, mean=0.0, std=0.02)

        # B_r / B_hi zeros => ΔW starts 0, and hi branch starts “inactive”
        # trainer asserts hi ΔW=0 at init => satisfied.
        self.use_hi = True  # trainer will toggle for r-only eval



    def assert_scaling_ready(self) -> None:
        if not hasattr(self, "scaling"):
            raise RuntimeError("DualRankLoRALinear missing attribute: scaling")
        if not hasattr(self, "scaling_r"):
            raise RuntimeError("DualRankLoRALinear missing attribute: scaling_r")
        if not hasattr(self, "scaling_hi"):
            raise RuntimeError("DualRankLoRALinear missing attribute: scaling_hi")
        if float(self.scaling_r) != float(self.scaling) or float(self.scaling_hi) != float(self.scaling):
            raise RuntimeError(
                f"Scaling mismatch: scaling={self.scaling}, scaling_r={self.scaling_r}, scaling_hi={self.scaling_hi}"
            )

            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.R <= 0:
            return y

        x_d = F.dropout(x, p=self.dropout, training=self.training) if self.dropout > 0 else x

        # r contribution
        lora_r = (x_d @ self.lora_A_r.t()) @ self.lora_B_r.t()

        # hi contribution
        if self.hi > 0 and self.use_hi:
            lora_hi = (x_d @ self.lora_A_hi.t()) @ self.lora_B_hi.t()
        else:
            lora_hi = 0.0

        return y + (lora_r + lora_hi) * self.scaling


def _match_targets(name: str, targets: Optional[List[str]]) -> bool:
    if not targets:
        return True
    for t in targets:
        if t in name:
            return True
    return False


def inject_lora(
    model: nn.Module,
    *,
    mode: str,
    r: int,
    R: int,
    alpha: float,
    dropout: float = 0.0,
    target_substrings: Optional[List[str]] = None,
) -> None:
    """
    Replace Linear layers with LoRA-wrapped modules.

    mode:
      - "baseline": use LoRALinear with rank=r, scaling alpha/r
      - "ours":     use DualRankLoRALinear with (r,R), scaling alpha/R
    """
    mode = str(mode).strip().lower()

    # walk modules and replace in parent
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _match_targets(full_name, target_substrings):
            continue

        # locate parent
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = full_name

        base_linear = getattr(parent, child_name)
        if not isinstance(base_linear, nn.Linear):
            continue

        if mode == "baseline":
            wrapped = LoRALinear(base_linear, r=int(r), alpha=float(alpha), dropout=float(dropout))
        else:
            wrapped = DualRankLoRALinear(base_linear, r=int(r), R=int(R), alpha=float(alpha), dropout=float(dropout))

        setattr(parent, child_name, wrapped)


def debug_check_dualrank_init(
    model: nn.Module,
    assert_hi_zero: bool = True,
    max_blocks_to_print: int = 3,
) -> None:
    """
    Lightweight init sanity check for your trainer call.
    """
    cnt = 0
    for name, m in model.named_modules():
        if hasattr(m, "lora_A_r") and hasattr(m, "lora_A_hi"):
            cnt += 1

            if cnt <= max_blocks_to_print:
                Ar = m.lora_A_r
                Br = m.lora_B_r
                Ahi = m.lora_A_hi
                Bhi = m.lora_B_hi
                print(
                    f"[DBG][Init][{name}] "
                    f"Ar={tuple(Ar.shape)} Br={tuple(Br.shape)} "
                    f"Ahi={tuple(Ahi.shape)} Bhi={tuple(Bhi.shape)} "
                    f"||Ar||={float(torch.norm(Ar).item()):.4f} ||Br||={float(torch.norm(Br).item()):.4f} "
                    f"||Ahi||={float(torch.norm(Ahi).item()):.4f} ||Bhi||={float(torch.norm(Bhi).item()):.4f}"
                )

            if assert_hi_zero:
                # ΔW_hi = B_hi A_hi, but B_hi is zero => should hold.
                if hasattr(m, "lora_B_hi"):
                    if float(torch.norm(m.lora_B_hi).item()) > 1e-12:
                        raise AssertionError(f"[Init] B_hi not zero for {name}")

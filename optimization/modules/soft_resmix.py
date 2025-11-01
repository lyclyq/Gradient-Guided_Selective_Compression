# # # modules/soft_resmix.py
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # class SoftResMix(nn.Module):
# #     """
# #     Soft residual mixing gate:
# #       y = sigma(beta) * y_gate + (1 - sigma(beta)) * x_skip

# #     - x_skip, y_gate: 同形状 (B, C, H, W) 或 (B, T, C)
# #     - beta: 可学习的标量（每层一个），或外部设定的温度/衰减策略
# #     """
# #     def __init__(self, init: float = 0.0, learnable: bool = True):
# #         super().__init__()
# #         self.learnable = learnable
# #         if learnable:
# #             self.beta = nn.Parameter(torch.tensor(float(init)))
# #         else:
# #             self.register_buffer("beta", torch.tensor(float(init)))

# #     def forward(self, x_skip: torch.Tensor, y_gate: torch.Tensor) -> torch.Tensor:
# #         assert x_skip.shape == y_gate.shape, f"SoftResMix shape mismatch: {x_skip.shape} vs {y_gate.shape}"
# #         s = torch.sigmoid(self.beta)
# #         return s * y_gate + (1.0 - s) * x_skip

# #     @torch.no_grad()
# #     def set_beta(self, value: float):
# #         if self.learnable:
# #             self.beta.data.fill_(float(value))
# #         else:
# #             self.beta.copy_(torch.tensor(float(value)))

# #     def get_mix_ratio(self) -> float:
# #         with torch.no_grad():
# #             return float(torch.sigmoid(self.beta).item())

# # ===============================
# # File: modules/soft_resmix.py
# # ===============================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Optional

# class SoftResMix(nn.Module):
#     """
#     Soft residual mixing gate:
#       y = sigma(beta) * y_gate + (1 - sigma(beta)) * x_skip

#     - x_skip, y_gate: 同形状 (B, C, H, W) 或 (B, T, C)
#     - beta: 可学习的标量（每层一个），或外部设定的温度/衰减策略
#     """
#     def __init__(self, init: float = 0.0, learnable: bool = True):
#         super().__init__()
#         self.learnable = learnable
#         if learnable:
#             self.beta = nn.Parameter(torch.tensor(float(init)))
#         else:
#             self.register_buffer("beta", torch.tensor(float(init)))

#     def forward(self, x_skip: torch.Tensor, y_gate: torch.Tensor) -> torch.Tensor:
#         assert x_skip.shape == y_gate.shape, f"SoftResMix shape mismatch: {x_skip.shape} vs {y_gate.shape}"
#         s = torch.sigmoid(self.beta)
#         return s * y_gate + (1.0 - s) * x_skip

#     @torch.no_grad()
#     def set_beta(self, value: float):
#         if self.learnable:
#             self.beta.data.fill_(float(value))
#         else:
#             self.beta.copy_(torch.tensor(float(value)))

#     def get_mix_ratio(self) -> float:
#         with torch.no_grad():
#             return float(torch.sigmoid(self.beta).item())


# # class SoftResMixNudger:
# #     """
# #     External nudger for a stack of SoftResMix modules.
# #     Allows applying a small step to each layer's beta, optionally scaled per-layer.
# #     """
# #     def __init__(self, model_root: nn.Module):
# #         super().__init__()
# #         self.resmix_modules: List[SoftResMix] = []
# #         # collect all SoftResMix under model_root
# #         for m in model_root.modules():
# #             if isinstance(m, SoftResMix):
# #                 self.resmix_modules.append(m)

# #     @torch.no_grad()
# #     def nudge(self, step_size: float, layer_scales: Optional[List[float]] = None, clamp: float = 10.0):
# #         if not self.resmix_modules:
# #             return
# #         for i, m in enumerate(self.resmix_modules):
# #             ss = step_size
# #             if layer_scales is not None and i < len(layer_scales):
# #                 ss *= float(layer_scales[i])
# #             if hasattr(m, 'beta'):
# #                 m.beta.data.add_(ss).clamp_(-clamp, clamp)

# class SoftResMixNudger:
#     """
#     遍历模型里所有插桩层的 SoftResMix（比如 small_bert 里的 b.resmix），
#     提供两种接口：
#       - nudge(step_size): 所有层统一步长
#       - nudge_per_layer(step_dict): 每层不同步长，step_dict: {layer_id: step}
#     """
#     def __init__(self, model, clamp=(0.0, 1.0)):
#         self.model = model
#         self.clamp = clamp
#         # 建立：layer_id -> SoftResMix 映射（依赖模型在构造时把它们收集到了 model.resmix_modules 或者 model.layer_modules）
#         self.layer2resmix = {}
#         if hasattr(model, "layer_modules"):
#             for i, blk in enumerate(model.layer_modules):
#                 if hasattr(blk, "resmix") and isinstance(blk.resmix, SoftResMix):
#                     self.layer2resmix[i] = blk.resmix
#         elif hasattr(model, "resmix_modules"):  # 兼容旧字段
#             # 若只有 list，但没有 layer_id，可按顺序索引
#             for i, r in enumerate(model.resmix_modules):
#                 if isinstance(r, SoftResMix):
#                     self.layer2resmix[i] = r

#     def _apply_delta(self, resmix: SoftResMix, delta: float):
#         with torch.no_grad():
#             resmix.alpha.add_(float(delta))
#             if self.clamp is not None:
#                 lo, hi = self.clamp
#                 resmix.alpha.clamp_(lo, hi)

#     def nudge(self, step_size: float):
#         """所有插桩层统一步长 nudger。"""
#         for lid, r in self.layer2resmix.items():
#             self._apply_delta(r, step_size)

#     def nudge_per_layer(self, step_dict: dict):
#         """
#         不同层不同步长；例如 step_dict = {0: 0.0012, 3: 0.0005, ...}
#         未在字典中的层不做调整。
#         """
#         for lid, step in step_dict.items():
#             r = self.layer2resmix.get(lid, None)
#             if r is None:
#                 continue
#             self._apply_delta(r, float(step))


# modules/soft_resmix.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple

# ------------------------------
# Soft residual mixing gate
#   y = sigma(beta) * y_gate + (1 - sigma(beta)) * x_skip
# ------------------------------
class SoftResMix(nn.Module):
    """
    Soft residual mixing gate:
      y = sigma(beta) * y_gate + (1 - sigma(beta)) * x_skip

    - x_skip, y_gate: 同形状 (B, C, H, W) 或 (B, T, C)
    - beta: 可学习的标量（每层一个），或外部设定的温度/衰减策略
    """
    def __init__(self, init: float = 0.0, learnable: bool = True):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.beta = nn.Parameter(torch.tensor(float(init)))
        else:
            self.register_buffer("beta", torch.tensor(float(init)))

    def forward(self, x_skip: torch.Tensor, y_gate: torch.Tensor) -> torch.Tensor:
        assert x_skip.shape == y_gate.shape, f"SoftResMix shape mismatch: {x_skip.shape} vs {y_gate.shape}"
        s = torch.sigmoid(self.beta)
        return s * y_gate + (1.0 - s) * x_skip

    @torch.no_grad()
    def set_beta(self, value: float):
        if self.learnable:
            self.beta.data.fill_(float(value))
        else:
            self.beta.copy_(torch.tensor(float(value)))

    def get_mix_ratio(self) -> float:
        with torch.no_grad():
            return float(torch.sigmoid(self.beta).item())


# ------------------------------
# SoftResMixNudger (alpha/beta 兼容)
# ------------------------------
class SoftResMixNudger:
    """
    统一的 ResMix 轻 nudger，兼容两类门控实现：
      1) 概率参数：模块上有 .alpha ∈ [0,1]
      2) Logit 参数：模块上有 .beta ∈ ℝ，门 s = σ(beta)

    设计要点：
    - 默认把 step_size 当成 **概率域的期望增量 Δs**。
      * 若模块是 alpha：直接 alpha += Δs，并按 clamp 概率域截断
      * 若模块是 beta ：用 Δβ = Δs / (s(1-s)+eps) 转换，再对 beta += Δβ
        如果 clamp 是概率域[0,1]，会把更新后的 s clamp，再回写 beta = logit(s_clamped)

    - clamp 处理（自适应）：
      * clamp=(0,1) -> 视为概率域约束（推荐，默认）
      * clamp 其他范围（如 (-10, 10)）-> 视为参数域约束

    - 自动收集：遍历 model.modules() 抓取所有 SoftResMix
    """
    def __init__(self, model: nn.Module, clamp: Tuple[float, float] = (0.0, 1.0), clamp_domain: str = "auto"):
        """
        Args:
            model: 根模型
            clamp: (lo, hi)。若在 [0,1] 内 -> 概率域；否则参数域
            clamp_domain: 'auto' | 'prob' | 'param'
        """
        self.model = model
        self.clamp = clamp
        self.clamp_domain = clamp_domain  # auto / prob / param
        self.eps = 1e-8

        # 收集所有 SoftResMix
        self._layers: Dict[int, SoftResMix] = {}
        idx = 0
        for m in model.modules():
            if isinstance(m, SoftResMix):
                self._layers[idx] = m
                idx += 1

        # 向后兼容：也允许通过 model.layer_modules[*].resmix 这种结构
        if not self._layers and hasattr(model, "layer_modules"):
            for i, blk in enumerate(getattr(model, "layer_modules")):
                if hasattr(blk, "resmix") and isinstance(blk.resmix, SoftResMix):
                    self._layers[i] = blk.resmix

        # 判断 clamp 域
        if self.clamp_domain == "auto":
            lo, hi = self.clamp
            if 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0:
                self._clamp_is_prob = True
            else:
                self._clamp_is_prob = False
        else:
            self._clamp_is_prob = (self.clamp_domain == "prob")

    # ---------- utils ----------
    @staticmethod
    def _sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def _logit(self, s: torch.Tensor) -> torch.Tensor:
        s = torch.clamp(s, self.eps, 1.0 - self.eps)
        return torch.log(s / (1.0 - s))

    def _get_param_kind(self, m: nn.Module) -> str:
        """
        返回 'alpha' / 'beta'
        优先使用 alpha（若存在），否则 beta。
        """
        if hasattr(m, "alpha"):
            return "alpha"
        if hasattr(m, "beta"):
            return "beta"
        raise AttributeError("SoftResMixNudger: target module has neither 'alpha' nor 'beta'.")

    def _get_prob(self, m: nn.Module) -> torch.Tensor:
        """
        读取当前 gate 概率 s：
          - alpha: 直接读
          - beta : s = sigmoid(beta)
        """
        kind = self._get_param_kind(m)
        if kind == "alpha":
            return getattr(m, "alpha")
        else:
            return self._sigmoid(getattr(m, "beta"))

    def _add_delta_prob(self, m: nn.Module, delta_s: float):
        """
        以“概率增量 Δs”的语义对模块做 nudging。
        对 alpha：alpha += Δs
        对 beta ：beta += Δs / (s(1-s)+eps)
        """
        kind = self._get_param_kind(m)
        if kind == "alpha":
            p = getattr(m, "alpha")
            p.data.add_(float(delta_s))
            if self._clamp_is_prob and self.clamp is not None:
                lo, hi = self.clamp
                p.data.clamp_(lo, hi)
        else:
            # beta 参数
            b = getattr(m, "beta")
            with torch.no_grad():
                s = self._sigmoid(b)
                # Δβ = Δs / (s(1-s)+eps)
                denom = (s * (1.0 - s) + self.eps)
                d_beta = float(delta_s) / float(denom.item()) if denom.numel() == 1 else delta_s / denom
                b.data.add_(d_beta)

                # clamp
                if self.clamp is not None:
                    if self._clamp_is_prob:
                        # 概率域 clamp：先 clamp s 再回写 beta = logit(s)
                        s_new = self._sigmoid(b)
                        lo, hi = self.clamp
                        s_new = torch.clamp(s_new, lo + self.eps, hi - self.eps)
                        b.data.copy_(self._logit(s_new))
                    else:
                        # 参数域 clamp：直接 clamp beta
                        lo, hi = self.clamp
                        b.data.clamp_(lo, hi)

    # ---------- public API ----------
    @torch.no_grad()
    def nudge(self, step_size: float):
        """所有层统一步长（概率域 Δs）"""
        if not self._layers:
            return
        for _, m in self._layers.items():
            self._add_delta_prob(m, float(step_size))

    @torch.no_grad()
    def nudge_per_layer(self, step_dict: Dict[int, float]):
        """
        每层不同步长（概率域 Δs）: step_dict = {layer_id: Δs, ...}
        未在字典中的层不做调整。
        """
        for lid, delta in step_dict.items():
            m = self._layers.get(lid, None)
            if m is None:
                continue
            self._add_delta_prob(m, float(delta))

    # 便捷查询
    @torch.no_grad()
    def get_state(self) -> Dict[int, Tuple[str, float]]:
        """
        返回每层的 (kind, prob)：
          kind ∈ {'alpha','beta'}
          prob = 当前 s
        """
        state = {}
        for lid, m in self._layers.items():
            kind = self._get_param_kind(m)
            s = float(self._get_prob(m).item())
            state[lid] = (kind, s)
        return state

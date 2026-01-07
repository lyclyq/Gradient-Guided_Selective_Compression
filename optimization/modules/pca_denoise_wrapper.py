# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Module: pca_denoise_wrapper.py
# ------------------------------
# A light-weight, plug-and-play wrapper that performs **denoise-only** projection
# on layer activations based on PCA overlap analysis. It:
#   - reads train-side PCA basis U_tr from a PCAOverlapManager
#   - accepts per-component gains from quickval (set_gains)
#   - at forward(), *optionally* projects to U_tr, scales noisy components,
#     reconstructs, and blends with identity (residual keep ratio)

# Design goals:
#   - Safe: no train→valid compensation (no shifting of means), only variance scaling
#   - Cheap: only applied on selected layers (usually last / last2)
#   - Flexible: works with [B,T,C] or [N,C]

# Typical use in a block's forward:
#   if self.pca_wrapper is not None:
#       x = self.pca_wrapper(x)

# Where the trainer calls once per quickval:
#   gains = { 'comp': torch.tensor([g1,...,gk], device), 'res': torch.tensor(g_res) }
#   pca_wrapper.set_gains(layer_id, gains)

# Note: the PCA basis lives in PCAOverlapManager which should be updated during train:
#   pca_mgr.update_train(layer_id, X_btC)
# """

# from __future__ import annotations
# from typing import Optional, Dict
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class PCADenoiseWrapper(nn.Module):
#     def __init__(self,
#                  layer_id: int,
#                  pca_mgr,                 # PCAOverlapManager
#                  k: int = 32,
#                  a_min: float = 0.85,     # min gain per component
#                  blend: float = 1.0,      # 1.0 -> fully use denoised, <1 -> blend with identity
#                  device: Optional[torch.device] = None):
#         super().__init__()
#         self.layer_id = layer_id
#         self.pca_mgr = pca_mgr
#         self.k = k
#         self.register_buffer('comp_gain', None, persistent=False)   # [k]
#         self.register_buffer('res_gain', None, persistent=False)    # scalar
#         self.a_min = a_min
#         self.blend = blend
#         self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     @torch.no_grad()
#     def set_gains(self, layer_id: int, gains: Dict[str, torch.Tensor]):
#         if layer_id != self.layer_id:  # ignore wrong routing
#             return
#         comp = gains.get('comp', None)   # [k]
#         res  = gains.get('res', None)    # scalar
#         if comp is not None:
#             comp = comp.to(self.device)
#             comp = torch.clamp(comp, min=self.a_min, max=1.0)
#             if comp.dim() == 0:
#                 comp = comp.view(1)
#             self.comp_gain = comp
#         if res is not None:
#             self.res_gain = torch.as_tensor(float(res), device=self.device).clamp(min=self.a_min, max=1.0)

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         # fast path: no gains set -> passthrough
#         if (self.comp_gain is None) and (self.res_gain is None):
#             return X

#         # Read train basis from PCAOverlapManager; if missing, passthrough
#         st = self.pca_mgr.layer_state.get(self.layer_id, None)
#         if st is None or 'cov' not in st:
#             return X
#         cov = st['cov']
#         # top-k eigenvectors (C,k)
#         evals, evecs = torch.linalg.eigh(cov)
#         idx = torch.argsort(evals, descending=True)
#         U_tr = evecs[:, idx[:self.k]]

#         was_3d = (X.dim() == 3)
#         if was_3d:
#             B,T,C = X.shape
#             Xf = X.reshape(B*T, C)
#         else:
#             Xf = X

#         # center by train mean if available (mean-centering is not compensation)
#         mu_tr = st.get('mu', None)
#         if mu_tr is not None:
#             Xc = Xf - mu_tr
#         else:
#             Xc = Xf - Xf.mean(0, keepdim=True)

#         # project
#         Z = Xc @ U_tr   # [N,k]

#         # apply component gains (if provided)
#         if self.comp_gain is not None:
#             g = self.comp_gain
#             if g.numel() < self.k:
#                 # pad with ones to k
#                 pad = torch.ones(self.k - g.numel(), device=g.device, dtype=g.dtype)
#                 g = torch.cat([g, pad], dim=0)
#             Z = Z * g[:self.k]

#         # reconstruct head
#         X_head = Z @ U_tr.t()

#         # residual part: X_res = Xc - X_head
#         X_res = Xc - X_head
#         if self.res_gain is not None:
#             X_res = X_res * self.res_gain

#         X_dnz = X_head + X_res
#         # add back mean
#         if mu_tr is not None:
#             X_dnz = X_dnz + mu_tr

#         # blend with identity for extra safety
#         if self.blend < 1.0:
#             X_dnz = self.blend * X_dnz + (1.0 - self.blend) * Xf

#         if was_3d:
#             X_dnz = X_dnz.view(B, T, C)
#         return X_dnz

#     def get_component_gains(self):
#         # 返回 shape=[k] 的当前 g_i（Tensor），用于回退阶段读取
#         return self.comp_gain.detach().clone()

#     def set_component_gains(self, g):
#         # 写入新的 g_i，并做 clamp
#         if not torch.is_tensor(g):
#             g = torch.tensor(g, dtype=self.comp_gain.dtype, device=self.comp_gain.device)
#         g = torch.clamp(g, min=float(getattr(self, 'a_min', 0.0)), max=1.0)
#         with torch.no_grad():
#             self.comp_gain.copy_(g)


from __future__ import annotations
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCADenoiseWrapper(nn.Module):
    """
    PCA 软/硬去噪包裹模块（支持 quickval 在线调整）.

    兼容旧构造参数:
        - a_min: 等价于 a_min_comp（主成分最小缩放系数）
    新增:
        - a_min_comp: 主成分最小缩放下界
        - a_min_res : 残差部分最小缩放下界
    注意:
        - 不再使用 register_buffer(None)；改为普通属性 self.comp_gain / self.res_gain
    """
    def __init__(
        self,
        layer_id: int,
        pca_mgr,
        k: int = 32,
        a_min_comp: float = 0.90,        # 主成分最小缩放
        a_min_res: float  = 0.60,        # 残差最小缩放
        blend: float = 1.0,
        device: Optional[torch.device] = None,
        # 兼容老版本调用
        a_min: Optional[float] = None,
    ):
        super().__init__()
        if a_min is not None:            # 旧参兼容: a_min -> a_min_comp
            a_min_comp = float(a_min)

        self.layer_id = int(layer_id)
        self.pca_mgr = pca_mgr
        self.k = int(k)

        # 不注册为 buffer，避免 register_buffer(None) 的兼容性问题
        self.comp_gain: Optional[torch.Tensor] = None   # [k] 或 None
        self.res_gain:  Optional[torch.Tensor] = None   # 标量 Tensor 或 None

        self.a_min_comp = float(a_min_comp)
        self.a_min_res  = float(a_min_res)
        self.blend = float(blend)

        # 设备
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 记录最近一次输入（仅用于调试/统计，不参与梯度）
        self._last_input: Optional[torch.Tensor] = None

    # ---------- 外部控制 API（quickval / 回退会调用） ----------
    @torch.no_grad()
    def set_gains(self, layer_id: int, gains: Dict[str, torch.Tensor]):
        """同时设置 comp/res 两路增益（用于批量下发）"""
        if int(layer_id) != self.layer_id:
            return
        comp = gains.get('comp', None)
        res  = gains.get('res', None)
        if comp is not None:
            comp = comp.to(self.device)
            if comp.dim() == 0:
                comp = comp.view(1)
            comp = torch.clamp(comp, min=self.a_min_comp, max=1.0)
            self.comp_gain = comp.clone()
        if res is not None:
            rg = torch.as_tensor(float(res), device=self.device, dtype=torch.float32)
            rg = torch.clamp(rg, min=self.a_min_res, max=1.0)
            self.res_gain = rg

    @torch.no_grad()
    def set_component_gains(self, g: torch.Tensor):
        """只设置主成分增益向量"""
        # g = g.to(self.device).float()
        g = torch.as_tensor(g, device=self.device, dtype=torch.float32)
        if g.dim() == 0:
            g = g.view(1)
        g = torch.clamp(g, min=self.a_min_comp, max=1.0)
        if (self.comp_gain is None) or (self.comp_gain.shape != g.shape):
            self.comp_gain = g.clone()
        else:
            self.comp_gain.copy_(g)

    @torch.no_grad()
    def set_res_gain(self, v: float):
        """只设置残差增益标量"""
        rg = torch.as_tensor(float(v), device=self.device, dtype=torch.float32)
        self.res_gain = torch.clamp(rg, min=self.a_min_res, max=1.0)

    @torch.no_grad()
    def set_blend(self, v: float):
        """设置输入/输出线性混合比例"""
        self.blend = float(v)

    def get_component_gains(self) -> Optional[torch.Tensor]:
        """读主成分增益（clone 后返回，避免外部原地改动）"""
        return None if self.comp_gain is None else self.comp_gain.detach().clone()

    # ---------- 前向 ----------
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 记录输入（不影响梯度）
        try:
            self._last_input = X.detach()
        except Exception:
            pass

        # 若当前没有任何增益设置或没有状态，直接透传
        if (self.comp_gain is None) and (self.res_gain is None):
            return X

        st = getattr(self.pca_mgr, "layer_state", {}).get(self.layer_id, None)
        if (st is None) or ('cov' not in st):
            return X

        cov = st['cov']  # 期望 [C, C]
        # 计算特征向量（按特征值降序）
        evals, evecs = torch.linalg.eigh(cov)
        idx = torch.argsort(evals, descending=True)
        U_tr = evecs[:, idx[:self.k]]  # [C, k]

        # 展平到 [N, C]
        was_3d = (X.dim() == 3)
        if was_3d:
            B, T, C = X.shape
            Xf = X.reshape(B * T, C)
        else:
            Xf = X

        # 零均值化（如果状态里有均值就用之）
        mu_tr = st.get('mu', None)
        if mu_tr is None:
            Xc = Xf - Xf.mean(0, keepdim=True)
        else:
            Xc = Xf - mu_tr

        # 投影到主成分
        Z = Xc @ U_tr  # [N, k]

        # 主成分增益
        # if self.comp_gain is not None:
        #     g = self.comp_gain

        if self.comp_gain is not None:
            g = self.comp_gain
            if g.dim() == 0:
                g = g.view(1)        
            if g.numel() < self.k:
                pad = torch.ones(self.k - g.numel(), device=g.device, dtype=g.dtype)
                g = torch.cat([g, pad], dim=0)
            Z = Z * g[:self.k]

        # 重构 + 残差
        X_head = Z @ U_tr.t()      # 主成分重构
        X_res  = Xc - X_head       # 残差
        if self.res_gain is not None:
            X_res = X_res * self.res_gain

        X_dnz = X_head + X_res
        if mu_tr is not None:
            X_dnz = X_dnz + mu_tr

        # 与原输入线性混合
        if self.blend < 1.0:
            X_dnz = self.blend * X_dnz + (1.0 - self.blend) * Xf

        # 还原形状
        if was_3d:
            X_dnz = X_dnz.view(B, T, C)

        return X_dnz

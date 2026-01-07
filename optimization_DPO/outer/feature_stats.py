# ===== FILE: outer/feature_stats.py =====
import torch
import torch.nn as nn
from typing import Optional

# 关键点：复用 adaptor 里已经验证可用的工具，避免两份实现不一致
from .adaptors import (
    _discover_layers as _discover_layers_common,
    _hidden_size as _hidden_size_common,
)

def _pick_layer(layers, layer_index: int):
    L = len(layers)
    li = layer_index if layer_index >= 0 else (L + layer_index)
    if li < 0 or li >= L:
        raise IndexError(f"layer_index {layer_index} out of range 0..{L-1}")
    return layers[li]

def _find_hf_q_module(layer):
    """
    兼容 HF:
      - DistilBERT: layer.attention.q_lin / out_lin
      - BERT:       layer.attention.self.query / layer.attention.output.dense
    """
    attn = getattr(layer, "attention", None) or getattr(layer, "attentions", None)
    if attn is not None:
        q = getattr(attn, "q_lin", None)
        if q is None:
            self_attn = getattr(attn, "self", None)
            if self_attn is not None:
                q = getattr(self_attn, "query", None)
        return q, attn
    return None, None

class FeatureTracker:
    """
    维护两路特征向量（H/Q），给策略网络输入：
      - 运行中的激活强度统计（均值/方差近似）
      - 最近一次梯度强度统计（行/列范数）
    与 HF(BERT/DistilBERT) 和 PyTorch MHA 均兼容。
    """
    def __init__(self, model, layer_index: int = -1,
                 feat_dim_H: int = 8, feat_dim_Q: int = 8,
                 momentum: float = 0.9, debug: bool = False):
        self.model = model
        self.layer_index = int(layer_index)
        self.feat_dim_H = int(feat_dim_H)
        self.feat_dim_Q = int(feat_dim_Q)
        self.momentum = float(momentum)
        self.debug = bool(debug)

        # 统一使用 adaptor 的发现函数（避免“各写一份”导致一个能找一个找不到）
        layers = _discover_layers_common(self.model)
        self.layer = _pick_layer(layers, self.layer_index)
        self.H = _hidden_size_common(self.model)

        # 两条路径：HF(q_lin) 或 PyTorch MHA(self_attn: MultiheadAttention)
        self.q_lin, self.attn_mod = _find_hf_q_module(self.layer)
        self.mha: Optional[nn.MultiheadAttention] = None
        if self.q_lin is None:
            sa = getattr(self.layer, "self_attn", None)
            if isinstance(sa, nn.MultiheadAttention):
                self.mha = sa
        if self.q_lin is None and self.mha is None:
            raise RuntimeError("FeatureTracker: cannot find attention query projection in target layer")

        dev = next(self.model.parameters()).device
        self.act_H = torch.zeros(self.H, device=dev)
        self.act_Q = torch.zeros(self.H, device=dev)
        self.grad_rows = torch.zeros(self.H, device=dev)  # 行范数（对应 H）
        self.grad_cols = torch.zeros(self.H, device=dev)  # 列范数（对应 Q）

        self._h_pre = None
        self._h_post = None
        self._install_hooks()

        if self.debug:
            path = "HF" if self.q_lin is not None else "MHA"
            print(f"[FeatureTracker] layers={len(layers)} | pick={self.layer_index} | H={self.H} | path={path}")

    def _install_hooks(self):
        m = self.momentum

        def _update_running(vec_slot: torch.Tensor, new_vec: torch.Tensor):
            with torch.no_grad():
                if new_vec.device != vec_slot.device:
                    new_vec = new_vec.to(vec_slot.device)
                if new_vec.dtype != vec_slot.dtype:
                    new_vec = new_vec.to(vec_slot.dtype)
                vec_slot.mul_(m).add_(new_vec * (1.0 - m))

        if self.q_lin is not None:
            # HF：pre 捕获输入 hidden（H-维），post 捕获输出（H-维）
            def pre_hook(_mod, inputs):
                if not inputs: return inputs
                x = inputs[0]  # [B,T,H]
                if x is not None and x.dim() == 3:
                    v = x.detach().abs().mean(dim=(0, 1))  # [H]
                    _update_running(self.act_H, v)
                return inputs

            def post_hook(_mod, _inp, out):
                y = out
                if isinstance(y, tuple):
                    y = y[0]
                if y is not None and y.dim() == 3:
                    v = y.detach().abs().mean(dim=(0, 1))  # [H]
                    _update_running(self.act_Q, v)
                return out

            self._h_pre  = self.q_lin.register_forward_pre_hook(lambda m, a: pre_hook(m, a))
            self._h_post = self.q_lin.register_forward_hook(lambda m, a, o: post_hook(m, a, o))
            return

        # PyTorch MHA：pre 能拿到 (q, k, v)，这里统计 q 的激活；Q 路近似同量级
        mha = self.mha
        def pre_hook(_mod, inputs):
            if not inputs: return inputs
            q = inputs[0]  # [T,B,H] 或 [B,T,H] 取决于 batch_first
            if q is not None and q.dim() == 3:
                # 兼容 (T,B,H) 与 (B,T,H) 两种约定
                if q.shape[0] == q.shape[1]:  # 极少见，忽略
                    vH = q.detach().abs().mean(dim=(0, 1))
                else:
                    # 统一到 (B,T,H) 再做均值
                    if getattr(mha, "batch_first", False):
                        vH = q.detach().abs().mean(dim=(0, 1))
                    else:
                        vH = q.detach().transpose(0, 1).abs().mean(dim=(0, 1))
                _update_running(self.act_H, vH)
                _update_running(self.act_Q, vH)
            return inputs

        self._h_pre = mha.register_forward_pre_hook(lambda m, a: pre_hook(m, a))

    @torch.no_grad()
    def update_grad_stats(self):
        """
        从 Query 权重梯度估计行/列范数：
        - HF: 用 q_lin.weight 的 grad
        - MHA: 用 in_proj_weight[:H,:] 的 grad
        """
        if self.q_lin is not None:
            W = getattr(self.q_lin, "weight", None)
            if W is None or W.grad is None:
                return
            g = W.grad.detach()
            self.grad_rows.copy_(g.norm(dim=1))  # 按行
            self.grad_cols.copy_(g.norm(dim=0))  # 按列
            return

        if self.mha is not None:
            W = getattr(self.mha, "in_proj_weight", None)
            if W is None or W.grad is None:
                return
            gq = W.grad.detach()[:self.H, :]  # Q 段
            self.grad_rows.copy_(gq.norm(dim=1))
            self.grad_cols.copy_(gq.norm(dim=0))
            return

    def _downproj(self, v: torch.Tensor, out_dim: int) -> torch.Tensor:
        """把 [H] 向量用自适应 1D 池化缩到 [out_dim]，稳定且不引依赖。"""
        if v.numel() == out_dim:
            return v.clone()
        x = v.view(1, 1, -1)
        y = torch.nn.functional.adaptive_avg_pool1d(x, out_dim)
        return y.view(-1)

    @torch.no_grad()
    def build_features_H(self) -> torch.Tensor:
        # 拼接 激活统计 + 梯度行范数，再做缩放到 feat_dim_H
        v = torch.cat([self.act_H, self.grad_rows + 1e-8], dim=0)
        return self._downproj(v, self.feat_dim_H)

    @torch.no_grad()
    def build_features_Q(self) -> torch.Tensor:
        # 拼接 激活统计 + 梯度列范数，再做缩放到 feat_dim_Q
        v = torch.cat([self.act_Q, self.grad_cols + 1e-8], dim=0)
        return self._downproj(v, self.feat_dim_Q)

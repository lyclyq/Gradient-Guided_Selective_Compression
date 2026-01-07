# ===== FILE: outer/adaptors.py =====
from typing import List, Tuple, Optional

import os
import torch
import torch.nn as nn


# ----------------- helpers -----------------

def _discover_layers(model: nn.Module) -> List[nn.Module]:
    """
    返回“按 encoder 深度顺序”的层对象列表。
    兼容：
      - nn.TransformerEncoder / TransformerEncoderLayer
      - 任何包含 self_attn: nn.MultiheadAttention 的模块序列
    """
    # 1) 显式属性
    for name in ("encoder", "enc", "transformer"):
        root = getattr(model, name, None)
        if root is None:
            continue
        for seq_name in ("layers", "layer", "h"):
            seq = getattr(root, seq_name, None)
            if seq is not None and hasattr(seq, "__iter__"):
                layers = [m for m in seq if isinstance(m, nn.Module)]
                if layers:
                    return layers

    # 2) 广义：寻找包含 self_attn 的子模块
    layers = []
    for m in model.modules():
        if hasattr(m, "self_attn") and isinstance(m.self_attn, nn.MultiheadAttention):
            layers.append(m)
    if layers:
        return layers

    # 3) 兜底：搜 TransformerEncoderLayer
    layers = [m for m in model.modules() if isinstance(m, nn.TransformerEncoderLayer)]
    if layers:
        return layers

    raise RuntimeError("Cannot discover encoder layers for adaptor.")


def _hidden_size(model: nn.Module) -> int:
    # 找第一处 MultiheadAttention 的 embed_dim
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            return int(m.embed_dim)
    # 退化到线性层推断
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    raise RuntimeError("Cannot infer hidden size for adaptor.")


def _pick_layer(layers: List[nn.Module], layer_index: int) -> nn.Module:
    L = len(layers)
    li = layer_index if layer_index >= 0 else (L + layer_index)
    if li < 0 or li >= L:
        raise IndexError(f"layer_index {layer_index} out of range 0..{L-1}")
    return layers[li]


def _to_1d_alpha(x: torch.Tensor, H: int, device: torch.device, name: str) -> torch.Tensor:
    """
    将任意形状/标量 alpha 转为长度 H 的 1D 向量；若长度不为 1 且不为 H，则报错。
    """
    x = torch.as_tensor(x, dtype=torch.float32, device=device).flatten()
    if x.numel() == 1:
        x = x.repeat(H)
    if x.numel() != H:
        raise ValueError(f"alpha `{name}` length mismatch: got {x.numel()} vs H={H}")
    return x


# ----------------- Adaptors -----------------

class _BaseAdaptor:
    """
    双对角缩放的基础类：
      - aH: 对 in_proj(Q) 的行缩放（H 维）
      - aQ: 对 in_proj(Q) 的列缩放（H 维）
    注意：只做缩放，不做量化/截断。
    """
    def __init__(
        self,
        model: nn.Module,
        layer_index: int = -1,
        alpha_scale: float = 0.05,
        mode: str = "in_proj",
    ):
        self.model = model
        self.layers = _discover_layers(model)
        self.layer = _pick_layer(self.layers, layer_index)
        self.attn: nn.MultiheadAttention = self.layer.self_attn  # type: ignore
        self.H = _hidden_size(model)
        self.alpha_scale = float(alpha_scale)
        self.mode = str(mode)

        # debug
        self._apply_cnt: int = 0
        self._verbose = bool(int(os.environ.get("ADAPTOR_DEBUG", "0")))

        # clean snapshots
        self.q_clean = None     # [H, H]
        self.out_clean = None   # [H, H]
        self._cache_clean()

        # current gate snapshot（确保与设备/维度一致）
        dev = next(self.attn.parameters()).device
        self._snap: Tuple[torch.Tensor, torch.Tensor] = (
            torch.zeros(self.H, device=dev, dtype=torch.float32),
            torch.zeros(self.H, device=dev, dtype=torch.float32),
        )

    def _cache_clean(self):
        dev = next(self.attn.parameters()).device
        with torch.no_grad():
            if hasattr(self.attn, "in_proj_weight") and self.attn.in_proj_weight is not None:
                # q 部分在前 H 行
                self.q_clean = self.attn.in_proj_weight[: self.H, :].detach().clone()
            elif hasattr(self.attn, "q_proj"):
                self.q_clean = self.attn.q_proj.weight.detach().clone()  # type: ignore
            else:
                raise RuntimeError("Cannot locate Q projection weights.")
            if getattr(self.attn, "out_proj", None) is None or self.attn.out_proj.weight is None:
                raise RuntimeError("Cannot locate out_proj weights.")
            self.out_clean = self.attn.out_proj.weight.detach().clone()
            self.q_clean = self.q_clean.to(device=dev, dtype=torch.float32)
            self.out_clean = self.out_clean.to(device=dev, dtype=torch.float32)

    @torch.no_grad()
    def restore(self):
        """恢复到干预前干净权重。"""
        if self.q_clean is None or self.out_clean is None:
            self._cache_clean()
        if hasattr(self.attn, "in_proj_weight") and self.attn.in_proj_weight is not None:
            self.attn.in_proj_weight[: self.H, :].copy_(self.q_clean)
        else:
            self.attn.q_proj.weight.copy_(self.q_clean)  # type: ignore
        self.attn.out_proj.weight.copy_(self.out_clean)

    # ---- API ----
    @torch.no_grad()
    def snapshot(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回当前 (aH, aQ) 的**深拷贝**（长度 H 的 1D 向量）。
        """
        return (self._snap[0].detach().clone(), self._snap[1].detach().clone())

    @torch.no_grad()
    def export_alpha_now(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        与 snapshot 等价，命名更显式：用于 Val2 “回读核对”。
        """
        return self.snapshot()

    @torch.no_grad()
    def get_current_alpha(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        与 snapshot 等价。为了向后兼容其他调用名。
        """
        return self.snapshot()

    @torch.no_grad()
    def load_snapshot_(self, snap):
        """
        接收 (aH, aQ) 或仅 aQ（QMaskAdaptor 兼容），一律转换为长度 H 的 1D 向量；
        内部存为深拷贝；然后应用到权重（仅缩放）。
        """
        dev = next(self.attn.parameters()).device
        if isinstance(snap, tuple):
            aH, aQ = snap
        else:
            aH, aQ = 0.0, snap

        aH = _to_1d_alpha(aH, self.H, dev, "aH")
        aQ = _to_1d_alpha(aQ, self.H, dev, "aQ")

        # 深拷贝保存
        self._snap = (aH.detach().clone(), aQ.detach().clone())

        # 应用
        self._apply_alpha(aH, aQ)

    # ---- core scaling ----
    @torch.no_grad()
    def _apply_alpha(self, aH: torch.Tensor, aQ: torch.Tensor):
        """
        双对角门：行缩放（H），列缩放（Q）
        仅缩放，不量化/不截断。
        """
        self.restore()
        sH = 1.0 + self.alpha_scale * aH  # [H]
        sQ = 1.0 + self.alpha_scale * aQ  # [H]

        if hasattr(self.attn, "in_proj_weight") and self.attn.in_proj_weight is not None:
            Wq = self.attn.in_proj_weight[: self.H, :]  # [H, H]
        else:
            Wq = self.attn.q_proj.weight  # type: ignore

        # Wq 行乘 sH，列乘 sQ
        Wq.mul_(sH.view(self.H, 1))
        Wq.mul_(sQ.view(1, self.H))

        # out_proj 仅与 sH 相关
        self.attn.out_proj.weight.mul_(sH.view(self.H, 1))

        self._apply_cnt += 1
        if self._verbose and (self._apply_cnt % 500 == 0):
            try:
                # 轻量追踪，防止刷屏
                mh = float(aH.abs().mean().item())
                mq = float(aQ.abs().mean().item())
                print(f"[ADAPTOR] apply#{self._apply_cnt} | mean|aH|={mh:.4e} mean|aQ|={mq:.4e}")
            except Exception:
                pass


class QMaskAdaptor(_BaseAdaptor):
    """
    只对 Q 侧应用列缩放（aQ），H 侧固定为 0（等价于 sH=1）。
    """
    @torch.no_grad()
    def load_snapshot_(self, aQ):
        dev = next(self.attn.parameters()).device
        aQ = _to_1d_alpha(aQ, self.H, dev, "aQ")
        aH = torch.zeros_like(aQ)  # H 侧不动
        self._snap = (aH.detach().clone(), aQ.detach().clone())
        self._apply_alpha(aH, aQ)


class DoubleDiagAdaptor(_BaseAdaptor):
    """
    双对角门（H 行 × Q 列）
    """
    pass


class GradSuppressor:
    """
    训练时抑制注意力 Q/OUT 的梯度幅度，缓解门控导致的震荡。
    - step(): 全局统一缩放（与旧逻辑兼容）
    - apply(mask_H, mask_Q): 按行/列用 mH/mQ 做逐元素抑制（与你的训练脚本一致）
    """
    def __init__(self, model: nn.Module, layer_index: int = -1, rho: float = 0.3):
        layers = _discover_layers(model)
        self.layer = _pick_layer(layers, layer_index)
        self.attn: nn.MultiheadAttention = self.layer.self_attn  # type: ignore
        self.rho = float(rho)
        self.H = _hidden_size(model)

    @torch.no_grad()
    def step(self):
        """
        旧版：统一按 (1 - rho) 缩放梯度。
        """
        try:
            if hasattr(self.attn, "in_proj_weight") and self.attn.in_proj_weight is not None:
                qg = self.attn.in_proj_weight.grad
                if qg is not None:
                    qg[: self.H, :].mul_(1.0 - self.rho)
            if getattr(self.attn.out_proj, "weight", None) is not None and self.attn.out_proj.weight.grad is not None:
                self.attn.out_proj.weight.grad.mul_(1.0 - self.rho)
        except Exception:
            pass

    @torch.no_grad()
    def apply(self, mask_H: Optional[torch.Tensor] = None, mask_Q: Optional[torch.Tensor] = None):
        """
        新增：按候选的 mH/mQ 做“按行/列”的细粒度抑制。
        设 grad := ∂L/∂Wq（Wq 形状 [H,H]），则：
            grad *= (1 - rho*|mH|).view(H,1)
            grad *= (1 - rho*|mQ|).view(1,H)
        out_proj.weight 的梯度仅按行（mH）缩放。
        """
        try:
            if hasattr(self.attn, "in_proj_weight") and self.attn.in_proj_weight is not None:
                gq = self.attn.in_proj_weight.grad  # [3H, H]
                if gq is not None:
                    g = gq[: self.H, :]  # 只取 Q 部分 [H, H]
                    if mask_H is not None:
                        mH = torch.as_tensor(mask_H, dtype=torch.float32, device=g.device).flatten()
                        if mH.numel() == 1:
                            mH = mH.repeat(self.H)
                        mH = mH[: self.H].abs()
                        g.mul_(torch.clamp(1.0 - self.rho * mH, 0.0, 1.0).view(self.H, 1))
                    if mask_Q is not None:
                        mQ = torch.as_tensor(mask_Q, dtype=torch.float32, device=g.device).flatten()
                        if mQ.numel() == 1:
                            mQ = mQ.repeat(self.H)
                        mQ = mQ[: self.H].abs()
                        g.mul_(torch.clamp(1.0 - self.rho * mQ, 0.0, 1.0).view(1, self.H))

            gout = getattr(self.attn.out_proj, "weight", None)
            if gout is not None and gout.grad is not None and mask_H is not None:
                mH = torch.as_tensor(mask_H, dtype=torch.float32, device=gout.grad.device).flatten()
                if mH.numel() == 1:
                    mH = mH.repeat(self.H)
                mH = mH[: self.H].abs()
                gout.grad.mul_(torch.clamp(1.0 - self.rho * mH, 0.0, 1.0).view(self.H, 1))
        except Exception:
            pass

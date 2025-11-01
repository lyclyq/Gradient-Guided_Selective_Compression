# ===== FILE: models/transformer_text.py =====
import math
from collections import deque
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, T, H]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H] (batch_first)
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class TransformerTextClassifier(nn.Module):
    """
    轻量 Transformer 文本分类器，支持注意力探针（可选）：
      - 只 hook 指定层（probe_layer_index，默认最后一层）；
      - 采集时将 attn 权重 (per-head, softmax 后) 按 [0,1] 量化为 uint8，限制窗口大小 (probe_max_T)，超出即裁剪；
      - 采用环形缓冲，容量 attn_buf_cap，pop 即清除，避免显存/内存增长。
    """
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        pad_id: int = 0,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        # probe:
        probe_layer_index: int = -1,
        attn_probe_enable: bool = False,
        attn_probe_max_T: int = 256,
        attn_probe_pool: str = "avg",  # ["none", "avg"]
        attn_buf_cap: int = 8,
        attn_quant_bits: int = 8,
        probe_runtime_enable: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.num_labels = num_labels

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout=0.0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_labels)

        # probe settings
        self.probe_layer_index = probe_layer_index
        self.attn_probe_enable = bool(attn_probe_enable)
        self.probe_runtime_enable = bool(probe_runtime_enable)
        self.attn_probe_max_T = int(attn_probe_max_T)
        self.attn_probe_pool = str(attn_probe_pool).lower()
        self.attn_buf_cap = int(attn_buf_cap)
        self.attn_quant_bits = int(attn_quant_bits)
        self._attn_buffer: deque = deque(maxlen=self.attn_buf_cap)

        if self.attn_probe_enable:
            self._install_attn_probe()

    # ---------- probe buffer ops ----------
    def set_probe(self, enable: bool):
        self.probe_runtime_enable = bool(enable)

    def clear_attn_buffer(self):
        self._attn_buffer.clear()

    @staticmethod
    def _quantize_uint(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, int]:
        # 输入 x ∈ [0,1]，量化为 uint{bits}
        qmax = (1 << bits) - 1
        x = x.clamp_(0, 1).mul_(qmax).round_().to(torch.uint8)
        return x.cpu(), qmax

    @staticmethod
    def _dequantize_uint(q: torch.Tensor, qmax: int) -> torch.Tensor:
        return (q.to(torch.float32) / float(qmax)).clamp_(1e-6, 1.0)

    def _push_attn(self, attn: torch.Tensor):
        """
        attn: [B, nH, Tq, Tk] 已经是 softmax 后的概率
        做裁剪/池化/量化，并放入环形缓冲
        """
        if not (self.attn_probe_enable and self.probe_runtime_enable):
            return
        try:
            # 限制窗口
            B, H, Tq, Tk = attn.shape
            Tcap = min(self.attn_probe_max_T, Tq)
            Scap = min(self.attn_probe_max_T, Tk)
            attn = attn[:, :, :Tcap, :Scap]  # 裁剪到窗口

            # 可选池化（批维做平均以减小存储）
            if self.attn_probe_pool == "avg" and B > 1:
                attn = attn.mean(dim=0, keepdim=True)  # [1, nH, T, S]

            q, qmax = self._quantize_uint(attn.detach(), self.attn_quant_bits)
            self._attn_buffer.append((q, qmax))
        except Exception:
            # 采集失败不影响前向
            pass

    @torch.no_grad()
    def pop_attn_logits(self) -> Optional[torch.Tensor]:
        """
        取出最近一次采集的注意力“logits-like”（返回 log(prob)），并从缓冲中删除。
        若无数据返回 None。
        """
        if len(self._attn_buffer) == 0:
            return None
        q, qmax = self._attn_buffer.pop()
        probs = self._dequantize_uint(q, qmax)  # [B, nH, T, S]
        return probs.clamp_(1e-6, 1.0).log()

    # ---------- hook install ----------
    def _install_attn_probe(self):
        # 选择层
        layers = [m for m in self.enc.modules() if isinstance(m, nn.TransformerEncoderLayer)]
        if len(layers) == 0:
            # 极端：无显式层对象，也不报错，直接关闭探针
            self.attn_probe_enable = False
            return

        if self.probe_layer_index in (-1, None):
            idx = len(layers) - 1
        else:
            idx = max(0, min(self.probe_layer_index, len(layers) - 1))

        target = layers[idx]
        attn = target.self_attn  # nn.MultiheadAttention
        if not isinstance(attn, nn.MultiheadAttention):
            self.attn_probe_enable = False
            return

        if getattr(attn, "_orig_forward", None) is not None:
            return  # 已装过

        attn._orig_forward = attn.forward

        def _wrapped_forward(query, key, value, **kwargs):
            # 强制 need_weights=True 且 average_attn_weights=False 以获得 per-head 权重
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            out, attn_w = attn._orig_forward(query, key, value, **kwargs)
            # PyTorch 2.x 返回 attn_w 形状通常为 [B, nH, Tq, Tk]（batch_first=True）
            if isinstance(attn_w, torch.Tensor):
                # attn_w 已是 softmax 概率
                self._push_attn(attn_w)
            return out, attn_w

        attn.forward = _wrapped_forward

    # ---------- forward ----------
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: [B, T]
        B, T = input_ids.shape
        mask_pad = (input_ids == self.pad_id)  # True=pad

        x = self.emb(input_ids)  # [B, T, H]
        x = self.pos(x)
        # nn.TransformerEncoder 接收 src_key_padding_mask: True=忽略
        x = self.enc(x, src_key_padding_mask=mask_pad)
        # mask-aware mean pooling
        if attention_mask is None:
            attention_mask = (~mask_pad).to(x.dtype)
        else:
            attention_mask = attention_mask.to(x.dtype)
        attn_sum = attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / attn_sum  # [B, H]
        logits = self.head(pooled)  # [B, C]
        return logits

# ===== FILE: models/distillbert_hf.py =====
import math
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from typing import Optional


class DistillBertClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int = 2,
        model_name: str = "distilbert-base-uncased",
        attn_probe_layer_index: int = -1,
        attn_probe_enable: bool = True,
        attn_probe_max_T: int = 256,
        attn_probe_pool: str = "avg",  # ["none", "avg"]
        attn_probe_warn_only: bool = True,
    ):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        # === probe runtime state ===
        self.attn_logits: Optional[torch.Tensor] = None  # 外环读取（一次性缓存）
        self._probe_handle = None
        self._last_attention_mask: Optional[torch.Tensor] = None
        self._probe_max_T = int(attn_probe_max_T) if attn_probe_max_T else 0
        self._probe_pool = attn_probe_pool
        self._probe_enabled = bool(attn_probe_enable)
        self._probe_warn_only = bool(attn_probe_warn_only)

        try:
            self._install_attn_probe(attn_probe_layer_index)
        except Exception as e:
            if self._probe_warn_only:
                print(f"[WARN][DistilBertClassifier] install probe failed: {e}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 缓存 mask 给 hook 使用
        self._last_attention_mask = attention_mask
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        return out.logits

    @torch.no_grad()
    def pop_attn_logits(self) -> Optional[torch.Tensor]:
        x = self.attn_logits
        self.attn_logits = None
        return x

    # ===== 标准接口：供 FeatureGates 使用 =====
    def get_transformer_layers(self):
        """
        返回 HF DistilBERT 的 6 个 TransformerBlock（ModuleList）：
        self.backbone.distilbert.transformer.layer
        """
        bb = self.backbone
        try:
            layers = bb.distilbert.transformer.layer  # ModuleList
            return list(layers)
        except Exception:
            return []

    def get_hidden_size(self):
        """
        DistilBERT 的隐藏维度在 config.hidden_size 或 config.dim 中。
        """
        cfg = getattr(self.backbone, "config", None)
        if cfg is None:
            return 768
        return getattr(cfg, "hidden_size", None) or getattr(cfg, "dim", 768)

    def remove_attn_probe(self):
        if self._probe_handle is not None:
            try:
                self._probe_handle.remove()
            except Exception:
                pass
            self._probe_handle = None

    # ===== 内部：安装注意力 logits 探针 =====
    def _install_attn_probe(self, layer_index: int = -1):
        if not self._probe_enabled:
            return
        if self._probe_handle is not None:
            return  # 避免重复注册

        bb = self.backbone
        try:
            layers = bb.distilbert.transformer.layer
        except Exception:
            return
        L = len(layers)
        if L == 0:
            return
        li = layer_index if layer_index >= 0 else (L + layer_index)
        li = max(0, min(L - 1, li))
        block = layers[li]
        # block.attention（或 block.attention.q_lin 所在的模块）
        attn = getattr(block, "attention", None)
        if attn is None:
            return

        # DistilBERT: attn 是 MultiHeadSelfAttention，字段：q_lin, k_lin, n_heads, dim
        q_lin = getattr(attn, "q_lin", None)
        k_lin = getattr(attn, "k_lin", None)
        if q_lin is None or k_lin is None:
            if self._probe_warn_only:
                print("[WARN][DistilBERT] attention has no q_lin/k_lin; fall back to output proxy.")
            return
        n_heads = getattr(attn, "n_heads", None)
        dim = getattr(attn, "dim", None)
        if n_heads is None or dim is None:
            cfg = self.backbone.config
            n_heads = getattr(cfg, "n_heads", getattr(cfg, "num_attention_heads", 12))
            dim = getattr(cfg, "dim", getattr(cfg, "hidden_size", 768))
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        head_dim = dim // n_heads
        scale = 1.0 / math.sqrt(head_dim)

        def _maybe_pool_scores(scores: torch.Tensor, T: int) -> torch.Tensor:
            if (self._probe_max_T and T > self._probe_max_T) and self._probe_pool == "avg":
                import math as _m
                stride = _m.ceil(T / self._probe_max_T)
                s = scores.view(scores.size(0) * scores.size(1), 1, T, T)
                s = torch.nn.functional.avg_pool2d(s, kernel_size=stride, stride=stride, ceil_mode=True)
                return s.view(scores.size(0), scores.size(1), s.shape[-2], s.shape[-1])
            return scores

        def pre_hook(_mod, inputs):
            # inputs: (hidden_states, ...) in most versions
            if not inputs:
                return
            hidden_states = inputs[0]  # [B, T, H]
            # dtype/设备对齐
            weight_dtype = q_lin.weight.dtype
            hidden_states = hidden_states.to(dtype=weight_dtype)

            q = q_lin(hidden_states)  # [B, T, H]
            k = k_lin(hidden_states)  # [B, T, H]
            B, T, Hdim = q.shape

            def split_heads(x):
                return x.view(B, T, n_heads, head_dim).transpose(1, 2).contiguous()

            qh = split_heads(q)
            kh = split_heads(k)
            scores = torch.matmul(qh, kh.transpose(-1, -2)).to(dtype=torch.float32) * scale  # [B,nH,T,T]

            # padding 掩码（仅对 K 轴）
            attn_mask = self._last_attention_mask
            if attn_mask is not None:
                k_mask = (attn_mask > 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
                scores = scores.masked_fill(~k_mask, float("-inf"))

            scores = _maybe_pool_scores(scores, T)
            self.attn_logits = scores.detach()

        self._probe_handle = attn.register_forward_pre_hook(pre_hook)


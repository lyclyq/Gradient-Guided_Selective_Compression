
# ===== FILE: models/bert_hf.py =====
import math
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from typing import Optional


class BertClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int = 2,
        model_name: str = "bert-base-uncased",
        attn_probe_layer_index: int = -1,
        attn_probe_enable: bool = True,
        attn_probe_max_T: int = 256,
        attn_probe_pool: str = "avg",  # ["none", "avg"]
        attn_probe_warn_only: bool = True,
    ):
        """
        attn_probe_layer_index: 要探针的 encoder 层索引（默认 -1 表示最后一层）
        """
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        # === probe runtime state ===
        self.attn_logits: Optional[torch.Tensor] = None
        self._probe_handle = None
        self._last_attention_mask: Optional[torch.Tensor] = None
        self._probe_max_T = int(attn_probe_max_T) if attn_probe_max_T else 0
        self._probe_pool = attn_probe_pool
        self._probe_enabled = bool(attn_probe_enable)
        self._probe_warn_only = bool(attn_probe_warn_only)

        # 安装注意力探针（重算 Q/K 得到 logits）
        try:
            self._install_attn_probe(attn_probe_layer_index)
        except Exception as e:
            if self._probe_warn_only:
                print(f"[WARN][BertClassifier] install probe failed: {e}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 缓存 mask 供 hook 使用
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
        返回 HF BERT 的 encoder layers（ModuleList）： self.backbone.bert.encoder.layer
        """
        bb = self.backbone
        try:
            layers = bb.bert.encoder.layer  # ModuleList
            return list(layers)
        except Exception:
            return []

    def get_hidden_size(self):
        """
        BERT 的隐藏维度通常为 config.hidden_size。
        """
        cfg = getattr(self.backbone, "config", None)
        if cfg is None:
            return 768
        return getattr(cfg, "hidden_size", 768)

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
            return

        layers = self.get_transformer_layers()
        if not layers:
            if self._probe_warn_only:
                print("[WARN][BERT] No encoder layers; fall back to output proxy.")
            return
        L = len(layers)
        li = layer_index if layer_index >= 0 else (L + layer_index)
        li = max(0, min(L - 1, li))
        tgt_layer = layers[li]  # BertLayer
        # BertLayer.attention.self 是 BertSelfAttention
        attn_self = getattr(getattr(tgt_layer, "attention", None), "self", None)
        if attn_self is None:
            return

        # 读取模型配置
        cfg = self.backbone.config
        num_heads = getattr(cfg, "num_attention_heads", None)
        hidden_size = getattr(cfg, "hidden_size", None)
        assert num_heads and hidden_size, "Missing heads/hidden_size in config"
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        head_dim = hidden_size // num_heads
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
            # inputs[0] 是 hidden_states: [B, T, H]
            if not inputs:
                return
            h = inputs[0]
            # dtype/设备对齐
            weight_dtype = attn_self.query.weight.dtype
            h = h.to(dtype=weight_dtype)

            # 使用该自注意力模块的线性层重算 Q/K
            q = attn_self.query(h)  # [B, T, H]
            k = attn_self.key(h)    # [B, T, H]
            B, T, _ = q.shape

            def split_heads(x):
                return x.view(B, T, num_heads, head_dim).transpose(1, 2).contiguous()

            qh = split_heads(q)
            kh = split_heads(k)
            # scores: [B, nH, T, T]
            scores = torch.matmul(qh, kh.transpose(-1, -2)).to(dtype=torch.float32) * scale

            # padding 掩码（仅对 K 轴）
            attn_mask = self._last_attention_mask
            if attn_mask is not None:
                k_mask = (attn_mask > 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
                scores = scores.masked_fill(~k_mask, float("-inf"))

            scores = _maybe_pool_scores(scores, T)
            # 存到模型属性（detach，避免带图）
            self.attn_logits = scores.detach()

        # 用 pre_hook 确保每次 forward 都会在 softmax 之前得到 logits
        self._probe_handle = attn_self.register_forward_pre_hook(pre_hook)


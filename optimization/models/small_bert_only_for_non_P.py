# # models/small_bert.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional
# from utils.registry import MODEL_REGISTRY

# # ---- 可选 DARTS 门(1D) ----
# from modules.filter_factory import make_filter_1d

# # ---- 软残差（统一从 modules 导入，签名：forward(identity, gated)) ----
# from modules.soft_resmix import SoftResMix


# class ImagePatchEmbed1D(nn.Module):
#     """把图像切成 patch，再展平为序列 token: [B,C,H,W] -> [B,T,D]"""
#     def __init__(self, in_ch=3, dim=192, patch=4):
#         super().__init__()
#         self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
#         self.patch = patch

#     def forward(self, x):  # x:[B,C,H,W]
#         x = self.proj(x)   # [B,D,H',W']
#         B, D, Hp, Wp = x.shape
#         return x.flatten(2).transpose(1, 2), (Hp, Wp)  # [B,T,D], (Hp,Wp)


# class BertBlock1D(nn.Module):
#     def __init__(self, dim, heads=4, mlp_ratio=4.0, attn_drop=0.0, drop=0.1,
#                  use_gate=False, gate_tau=1.0, use_resmix=False, resmix_init=0.0,
#                  filter_backend: str = "darts", ks_list=(3,7,15), lp_k: int = 7,
#                  feature_mask: bool = False, ff_gamma: float = 1.0, ff_amin: float = 0.8,
#                  ff_apply_on: str = "input", ff_use_drv_gate: bool = False):
#         super().__init__()
#         self.use_gate = bool(use_gate)
#         self.use_resmix = bool(use_resmix)

#         if self.use_gate:
#             self.pre_attn_gate = make_filter_1d(
#                 channels=dim, backend=filter_backend, ks_list=ks_list, tau=gate_tau, lp_k=lp_k,
#                 feature_mask=feature_mask, ff_gamma=ff_gamma, ff_amin=ff_amin,
#                 ff_apply_on=ff_apply_on, ff_use_drv_gate=ff_use_drv_gate
#             )
#         if self.use_resmix:
#             self.resmix = SoftResMix(init=resmix_init)  # forward(identity, gated)

#         self.ln1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
#         self.drop = nn.Dropout(drop)
#         self.ln2 = nn.LayerNorm(dim)
#         hidden = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
#             nn.Linear(hidden, dim), nn.Dropout(drop)
#         )

#     def set_tau(self, tau: float):
#         if self.use_gate and hasattr(self.pre_attn_gate, "set_tau"):
#             self.pre_attn_gate.set_tau(tau)

#     def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
#         # 低通门控 + 软残差（不开 gate 时，gated=x）
#         h_in = x
#         gated = self.pre_attn_gate(x) if self.use_gate else x
#         if self.use_resmix:
#             x = self.resmix(h_in, gated)  # 统一：identity, gated
#         else:
#             x = gated

#         h = x
#         x = self.ln1(x)
#         # 注意：key_padding_mask=True 表示 padding 位置
#         x, _ = self.attn(x, x, x, key_padding_mask=attn_mask, need_weights=False)
#         x = h + self.drop(x)

#         h = x
#         x = self.ln2(x)
#         x = h + self.mlp(x)
#         return x


# @MODEL_REGISTRY.register("small_bert")
# class SmallBERT(nn.Module):
#     """
#     轻量 BERT 样式 Encoder，支持图像/文本双路：
#     - 文本：
#         * text_task="cls"：输入 ids [B,T] -> logits [B,num_classes]（masked mean-pool）
#         * text_task="tokcls"：输入 ids [B,T] -> logits [B,T,num_tags]
#         * text_task=None：LM（不常用）
#     - 图像：输入 [B,C,H,W] -> patch-embed -> [B,T,D] -> mean-pool -> logits [B,num_classes]
#     """
#     def __init__(self,
#                  # 通用
#                  dim=192, n_layers=4, heads=4, mlp_ratio=4.0, drop=0.1, attn_drop=0.0,
#                  gate_tau=1.0,
#                  # ---- 文本配置 ----
#                  vocab_size=30522, max_len=512, pad_id=0, cls_token=False,
#                  text_task: Optional[str] = None,            # None / "cls" / "tokcls"
#                  text_num_classes: Optional[int] = None,     # for "cls"
#                  tok_num_classes: Optional[int] = None,      # for "tokcls"
#                  # ---- 图像配置 ----
#                  in_ch=3, num_classes=10, img_patch=4,
#                  # ---- 可选：DARTS 门 & 软残差 ----
#                  use_gate=False, use_resmix=False, resmix_init=0.0):
#         super().__init__()
#         self.dim = dim
#         self.n_layers = n_layers
#         self.heads = heads
#         self.cls_token = cls_token
#         self.pad_id = pad_id

#         # 文本任务类型
#         assert text_task in (None, "cls", "tokcls"), f"text_task must be None/'cls'/'tokcls', got {text_task}"
#         self.text_task = text_task
#         self.text_num_classes = text_num_classes
#         self.tok_num_classes = tok_num_classes

#         # ---- 文本 embedding ----
#         self.text_vocab_size = vocab_size
#         self.text_pos = nn.Embedding(max_len + (1 if cls_token else 0), dim)
#         self.tok = nn.Embedding(vocab_size, dim, padding_idx=pad_id)

#         # ---- 图像 embedding ----
#         self.img_embed = ImagePatchEmbed1D(in_ch=in_ch, dim=dim, patch=img_patch)

#         # ---- Transformer blocks（共享）----
#         self.blocks = nn.ModuleList([
#             BertBlock1D(dim, heads, mlp_ratio, attn_drop, drop,
#                         use_gate=use_gate, gate_tau=gate_tau,
#                         use_resmix=use_resmix, resmix_init=resmix_init)
#             for _ in range(n_layers)
#         ])
#         self.ln = nn.LayerNorm(dim)

#         # ---- 输出头 ----
#         self.cls_head = nn.Linear(dim, num_classes)  # 图像分类
#         self.lm_head = nn.Linear(dim, vocab_size, bias=False)  # 语言建模（默认不用）
#         self.textcls_head = nn.Linear(dim, text_num_classes) if text_task == "cls" and text_num_classes is not None else None
#         self.tokcls_head  = nn.Linear(dim, tok_num_classes)  if text_task == "tokcls" and tok_num_classes  is not None else None

#         self._has_gate = bool(use_gate and (FilterGateDARTS1D is not None))

#     # ------- DARTS 接口 -------
#     def arch_parameters(self):
#         params = []
#         for b in self.blocks:
#             if hasattr(b, "pre_attn_gate") and hasattr(b.pre_attn_gate, "alpha_logits"):
#                 params.append(b.pre_attn_gate.alpha_logits)
#         return params

#     def weight_parameters(self):
#         arch_ids = {id(p) for p in self.arch_parameters()}
#         return [p for p in self.parameters() if id(p) not in arch_ids]

#     def set_tau(self, tau: float):
#         for b in self.blocks:
#             if hasattr(b, "set_tau"):
#                 b.set_tau(tau)

#     def controller_kernel_weights(self):
#         """供 DCA 读取 DWConv kernel 权重，增强结构感知"""
#         ws = []
#         for b in self.blocks:
#             if getattr(b, "use_gate", False) and hasattr(b, "pre_attn_gate"):
#                 gate = b.pre_attn_gate
#                 if hasattr(gate, "convs"):
#                     for conv in gate.convs:
#                         ws.append(conv.weight)
#                 elif hasattr(gate, "conv"):
#                     ws.append(gate.conv.weight)
#         return ws

#     # ------- 前向 -------
#     def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
#         """
#         - 图像：x:[B,C,H,W] -> logits [B,num_classes]
#         - 文本：
#             * textcls：x:Long[B,T] -> logits [B,num_classes]（masked mean-pool）
#             * tokcls ：x:Long[B,T] -> logits [B,T,num_tags]
#             * 其他：LM [B,T,V]
#         """
#         # -------- 文本路径 --------
#         if x.dtype in (torch.int32, torch.int64) and x.dim() == 2:
#             ids = x  # [B,T]
#             B, T = ids.shape

#             pos_ids = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
#             h = self.tok(ids) + self.text_pos(pos_ids)           # [B,T,D]
#             padmask = (ids == self.pad_id)                       # True 表示 padding

#             for b in self.blocks:
#                 h = b(h, attn_mask=padmask)
#             h = self.ln(h)                                       # [B,T,D]

#             if self.text_task == "cls" and self.textcls_head is not None:
#                 if padmask.any():
#                     lens = (~padmask).sum(dim=1).clamp_min(1).unsqueeze(-1)  # [B,1]
#                     masked = h.masked_fill(padmask.unsqueeze(-1), 0.0)
#                     feat = masked.sum(dim=1) / lens                          # [B,D]
#                 else:
#                     feat = h.mean(dim=1)
#                 return self.textcls_head(feat)                                # [B,num_classes]

#             if self.text_task == "tokcls" and self.tokcls_head is not None:
#                 return self.tokcls_head(h)                                    # [B,T,num_tags]

#             # 默认：LM 头
#             return self.lm_head(h)                                            # [B,T,V]

#         # -------- 图像路径 --------
#         elif x.dim() == 4:
#             tok, _ = self.img_embed(x)                                        # [B,T,D]
#             h = tok
#             for b in self.blocks:
#                 h = b(h, attn_mask=None)
#             h = self.ln(h)
#             h = h.mean(dim=1)                                                 # mean-pool
#             return self.cls_head(h)                                           # [B,num_classes]

#         else:
#             raise ValueError(f"Unsupported input shape {tuple(x.shape)} / dtype {x.dtype}")


# ===============================
# File: models/small_bert.py (patched to expose insert_layers + PCA wrapper hooks)
# ===============================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from utils.registry import MODEL_REGISTRY

from modules.filter_factory import make_filter_1d
from modules.soft_resmix import SoftResMix
from modules.pca_denoise_wrapper import PCADenoiseWrapper


class BertBlock1D(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0, attn_drop=0.0, drop=0.1,
                 use_gate=False, gate_tau=1.0, use_resmix=False, resmix_init=0.0,
                 filter_backend: str = "darts", ks_list=(3,7,15), lp_k: int = 7,
                 feature_mask: bool = False, ff_gamma: float = 1.0, ff_amin: float = 0.8,
                 ff_apply_on: str = "input", ff_use_drv_gate: bool = False,
                 pca_k: int = 32, pca_amin: float = 0.85, enable_pca: bool = False,
                 layer_id: int = -1):
        super().__init__()
        self.layer_id = layer_id
        self.use_gate = bool(use_gate)
        self.use_resmix = bool(use_resmix)
        self.enable_pca = bool(enable_pca)
        self.pca_wrap = PCADenoiseWrapper(layer_id=layer_id, pca_mgr=None, k=pca_k, a_min=pca_amin, blend=1.0) if enable_pca else None

        # ✅ 先给属性占位，避免属性不存在
        self.pre_attn_gate = None
        self.resmix = None

        if self.use_gate:
            self.pre_attn_gate = make_filter_1d(
                channels=dim, backend=filter_backend, ks_list=ks_list, tau=gate_tau, lp_k=lp_k,
                feature_mask=feature_mask, ff_gamma=ff_gamma, ff_amin=ff_amin,
                ff_apply_on=ff_apply_on, ff_use_drv_gate=ff_use_drv_gate
            )
        if self.use_resmix:
            self.resmix = SoftResMix(init=resmix_init)

        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop)
        )

    def attach_pca_manager(self, pca_mgr):
        if self.pca_wrap is not None:
            self.pca_wrap.pca_mgr = pca_mgr

    def set_tau(self, tau: float):
        if self.use_gate and hasattr(self.pre_attn_gate, "set_tau"):
            self.pre_attn_gate.set_tau(tau)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # --- PCA: train统计 / eval缓存 / 降噪 ---
        if self.pca_wrap is not None:
            if self.training and self.pca_wrap.pca_mgr is not None:
                self.pca_wrap.pca_mgr.update_train(self.layer_id, x.detach())
            else:
                self.pca_wrap._last_input = x.detach()
            x = self.pca_wrap(x)

        # 低通门控 + 软残差
        h_in = x
        gated = self.pre_attn_gate(x) if self.use_gate else x
        x = self.resmix(h_in, gated) if self.use_resmix else gated

        h = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=attn_mask, need_weights=False)
        x = h + self.drop(x)

        h = x
        x = self.ln2(x)
        x = h + self.mlp(x)
        return x


@MODEL_REGISTRY.register("small_bert")
class SmallBERT(nn.Module):
    def __init__(self,
                 dim=192, n_layers=4, heads=4, mlp_ratio=4.0, drop=0.1, attn_drop=0.0,
                 gate_tau=1.0,
                 vocab_size=30522, max_len=512, pad_id=0, cls_token=False,
                 text_task: Optional[str] = None,
                 text_num_classes: Optional[int] = None,
                 tok_num_classes: Optional[int] = None,
                 in_ch=3, num_classes=10, img_patch=4,
                 use_gate=False, use_resmix=False, resmix_init=0.0,
                 insert_layers: str = 'last',
                 filter_backend: str = 'lp_fixed', ks_list=(3,7,15), lp_k: int = 7,
                 feature_mask: bool = False, ff_gamma: float = 1.0, ff_amin: float = 0.8,
                 ff_apply_on: str = 'input', ff_use_drv_gate: bool = False,
                 pca_enable: bool = True, pca_k: int = 32, pca_amin: float = 0.85):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.heads = heads
        self.cls_token = cls_token
        self.pad_id = pad_id
        self.text_task = text_task
        self.text_num_classes = text_num_classes
        self.tok_num_classes = tok_num_classes

        # Embeddings
        self.text_vocab_size = vocab_size
        self.text_pos = nn.Embedding(max_len + (1 if cls_token else 0), dim)
        self.tok = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.img_embed = nn.Conv2d(in_ch, dim, kernel_size=img_patch, stride=img_patch)

        # parse insert layers
        def parse_layers(arg: str, n: int) -> List[int]:
            a = (arg or 'last').lower()
            if a == 'last':
                return [n-1]
            if a in ('last2','last-2','last_2'):
                return [max(0,n-2), n-1]
            if a == 'all':
                return list(range(n))
            try:
                idxs = [int(x.strip()) for x in arg.split(',') if x.strip()!='']
                return [i for i in sorted(set(idxs)) if 0 <= i < n]
            except Exception:
                return [n-1]
        insert_ids = parse_layers(insert_layers, n_layers)

        # Blocks
        self.blocks = nn.ModuleList()
        self.layer_modules: List[BertBlock1D] = []
        self.resmix_modules: List[SoftResMix] = []
        self.pca_wrappers: Dict[int, PCADenoiseWrapper] = {}
        for i in range(n_layers):
            use_here = (i in insert_ids)
            b = BertBlock1D(dim, heads, mlp_ratio, attn_drop, drop,
                            use_gate=use_here and use_resmix or use_here,  # 只要插桩就允许gate
                            gate_tau=gate_tau,
                            use_resmix=use_here and use_resmix,
                            resmix_init=resmix_init,
                            filter_backend=filter_backend, ks_list=ks_list, lp_k=lp_k,
                            feature_mask=feature_mask, ff_gamma=ff_gamma, ff_amin=ff_amin,
                            ff_apply_on=ff_apply_on, ff_use_drv_gate=ff_use_drv_gate,
                            pca_k=pca_k, pca_amin=pca_amin, enable_pca=pca_enable, layer_id=i)
            self.blocks.append(b)
            self.layer_modules.append(b)
            if b.resmix is not None:
                self.resmix_modules.append(b.resmix)
            if b.pca_wrap is not None:
                self.pca_wrappers[i] = b.pca_wrap

        self.ln = nn.LayerNorm(dim)
        self.img_head = nn.Linear(dim, num_classes)
        self.textcls_head = nn.Linear(dim, text_num_classes) if text_task == 'cls' and text_num_classes is not None else None
        self.tokcls_head  = nn.Linear(dim, tok_num_classes)  if text_task == 'tokcls' and tok_num_classes  is not None else None

    def attach_pca_manager(self, pca_mgr):
        for b in self.blocks:
            b.attach_pca_manager(pca_mgr)

    def controller_kernel_weights(self):
        ws = []
        for b in self.blocks:
            if hasattr(b, 'pre_attn_gate') and b.pre_attn_gate is not None:
                g = b.pre_attn_gate
                if hasattr(g, 'convs'):
                    for c in g.convs:
                        ws.append(c.weight)
                elif hasattr(g, 'conv'):
                    ws.append(g.conv.weight)
        return ws

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        if x.dtype in (torch.int32, torch.int64) and x.dim() == 2:
            ids = x
            B, T = ids.shape
            pos_ids = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
            h = self.tok(ids) + self.text_pos(pos_ids)           # [B,T,D]
            padmask = (ids == self.pad_id)
            for b in self.blocks:
                h = b(h, attn_mask=padmask)
            h = self.ln(h)
            if self.text_task == 'cls' and self.textcls_head is not None:
                if padmask.any():
                    lens = (~padmask).sum(dim=1).clamp_min(1).unsqueeze(-1)
                    masked = h.masked_fill(padmask.unsqueeze(-1), 0.0)
                    feat = masked.sum(dim=1) / lens
                else:
                    feat = h.mean(dim=1)
                return {'logits': self.textcls_head(feat)}
            if self.text_task == 'tokcls' and self.tokcls_head is not None:
                return {'logits': self.tokcls_head(h)}
            return {'logits': h}  # LM
        elif x.dim() == 4:
            tok = self.img_embed(x)                     # [B,D,H',W']
            B,D,H,W = tok.shape
            h = tok.flatten(2).transpose(1,2)           # [B,T,D]
            for b in self.blocks:
                h = b(h)
            h = self.ln(h)
            h = h.mean(dim=1)
            return {'logits': self.img_head(h)}
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)} / dtype {x.dtype}")

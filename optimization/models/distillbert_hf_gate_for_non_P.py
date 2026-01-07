# # models/distillbert_hf_gate.py
# import torch
# import torch.nn as nn
# from transformers import DistilBertModel, DistilBertConfig
# from utils.registry import MODEL_REGISTRY

# # 可选 DARTS 1D gate
# try:
#     from modules.filter_gate_darts import FilterGateDARTS1D
# except Exception:
#     FilterGateDARTS1D = None

# # 软残差（统一签名：forward(identity, gated)）
# from modules.soft_resmix import SoftResMix


# class DistilBertLayerWithGate(nn.Module):
#     """
#     在 DistilBERT 每层前插入 1D Gate(DARTS) + 可选 SoftResMix
#     保持 HuggingFace 层的接口（*args, **kwargs 转发）
#     """
#     def __init__(self, layer, dim, use_gate=True, use_resmix=False, resmix_init=0.0, gate_tau=1.0):
#         super().__init__()
#         self.layer = layer
#         self.use_gate = bool(use_gate and (FilterGateDARTS1D is not None))
#         self.use_resmix = bool(use_resmix)
#         if self.use_gate:
#             self.pre_attn_gate = FilterGateDARTS1D(dim, ks_list=(3, 7, 15), tau=gate_tau)
#         if self.use_resmix:
#             self.resmix = SoftResMix(init=resmix_init)

#     def set_tau(self, tau: float):
#         if self.use_gate and hasattr(self.pre_attn_gate, "set_tau"):
#             self.pre_attn_gate.set_tau(tau)

#     def forward(self, hidden_states, *args, **kwargs):
#         h_in = hidden_states
#         gated = self.pre_attn_gate(h_in) if self.use_gate else h_in
#         x = self.resmix(h_in, gated) if self.use_resmix else gated
#         outputs = self.layer(x, *args, **kwargs)
#         return outputs  # 与 HF 保持一致：(hidden_states, ...)


# @MODEL_REGISTRY.register("distillbert_hf_gate")
# class DistillBertHFGate(nn.Module):
#     """
#     DistilBERT + (可选)Gate/ResMix：
#       - 支持 text_task="cls"/"tokcls"
#       - 暴露 arch_parameters/weight_parameters，便于 DARTS & DCA
#       - `controller_kernel_weights()` 供 DCA 读取 DWConv 权重
#     """
#     def __init__(self,
#                  text_task="cls",
#                  text_num_classes=2,
#                  tok_num_classes=None,
#                  pad_id=0,
#                  use_gate=True,
#                  use_resmix=False,
#                  resmix_init=0.0,
#                  gate_tau=1.0,
#                  pretrained="distilbert-base-uncased"):
#         super().__init__()
#         self.text_task = text_task
#         self.text_num_classes = text_num_classes
#         self.tok_num_classes = tok_num_classes
#         self.pad_id = pad_id

#         # backbone
#         self.config = DistilBertConfig.from_pretrained(pretrained)
#         self.bert = DistilBertModel.from_pretrained(pretrained, config=self.config)

#         dim = self.config.hidden_size
#         # wrap transformer layers
#         new_layers = []
#         for layer in self.bert.transformer.layer:
#             new_layers.append(
#                 DistilBertLayerWithGate(
#                     layer, dim, use_gate=use_gate, use_resmix=use_resmix,
#                     resmix_init=resmix_init, gate_tau=gate_tau
#                 )
#             )
#         self.bert.transformer.layer = nn.ModuleList(new_layers)

#         # heads
#         if text_task == "cls":
#             self.cls_head = nn.Linear(dim, text_num_classes)
#         elif text_task == "tokcls":
#             assert tok_num_classes is not None, "tok_num_classes must be provided for tokcls"
#             self.tok_head = nn.Linear(dim, tok_num_classes)
#         else:
#             raise ValueError("text_task must be 'cls' or 'tokcls'")

#     # --- DARTS 接口 ---
#     def arch_parameters(self):
#         params = []
#         for layer in self.bert.transformer.layer:
#             if hasattr(layer, "pre_attn_gate"):
#                 params.append(layer.pre_attn_gate.alpha_logits)
#         return params

#     def weight_parameters(self):
#         arch_ids = {id(p) for p in self.arch_parameters()}
#         return [p for p in self.parameters() if id(p) not in arch_ids]

#     def set_tau(self, tau: float):
#         for layer in self.bert.transformer.layer:
#             if hasattr(layer, "set_tau"):
#                 layer.set_tau(tau)

#     def controller_kernel_weights(self):
#         """供 DCA 读取 DWConv kernel 权重"""
#         ws = []
#         for layer in self.bert.transformer.layer:
#             if hasattr(layer, "pre_attn_gate") and hasattr(layer.pre_attn_gate, "convs"):
#                 for conv in layer.pre_attn_gate.convs:
#                     ws.append(conv.weight)
#         return ws

#     def forward(self, x, attention_mask=None, **kwargs):
#         # HF 期望 input_ids + attention_mask
#         out = self.bert(input_ids=x, attention_mask=attention_mask, **kwargs)

#         if self.text_task == "cls":
#             # 改为 masked mean-pool（比取首 token 更稳）
#             last = out.last_hidden_state              # [B,T,H]
#             if attention_mask is None:
#                 attention_mask = (x != self.pad_id).long()
#             mask = attention_mask.unsqueeze(-1).float()
#             pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
#             return self.cls_head(pooled)              # [B,C]
#         else:
#             return self.tok_head(out.last_hidden_state)  # [B,T,C]

# ===============================
# File: models/distillbert_hf_gate.py
# ===============================
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from transformers import DistilBertModel, DistilBertConfig
from utils.registry import MODEL_REGISTRY

from modules.filter_factory import make_filter_1d
from modules.soft_resmix import SoftResMix
from modules.pca_denoise_wrapper import PCADenoiseWrapper


def _parse_insert_layers(arg: str, n_layers: int) -> List[int]:
    a = (arg or 'last').lower()
    if a == 'last':
        return [n_layers - 1]
    if a in ('last2', 'last-2', 'last_2'):
        return [max(0, n_layers - 2), n_layers - 1]
    if a == 'all':
        return list(range(n_layers))
    try:
        idxs = [int(x.strip()) for x in arg.split(',') if x.strip() != '']
        return [i for i in sorted(set(idxs)) if 0 <= i < n_layers]
    except Exception:
        return [n_layers - 1]


class DistilBertLayerWithAdapters(nn.Module):
    """一层 DistilBERT + 可选的 1D 低通/FFT 门控 + SoftResMix + PCA 降噪 wrapper。
    - pre_attn_gate: 由 filter_factory 创建（lp_fixed/feature_mask/none）
    - resmix: SoftResMix
    - pca_wrap: 仅做降噪（不补偿）
    """
    def __init__(self, layer, dim: int, use_gate: bool, use_resmix: bool,
                 filter_backend: str, gate_tau: float, lp_k: int,
                 feature_mask: bool, ff_gamma: float, ff_amin: float,
                 ff_apply_on: str, ff_use_drv_gate: bool,
                 pca_mgr=None, pca_k: int = 32, pca_amin: float = 0.85):
        super().__init__()
        self.layer = layer
        self.use_gate = bool(use_gate)
        self.use_resmix = bool(use_resmix)
        self.pre_attn_gate = None
        if self.use_gate:
            self.pre_attn_gate = make_filter_1d(
                channels=dim, backend=filter_backend, ks_list=(3,7,15), tau=gate_tau,
                lp_k=lp_k, feature_mask=feature_mask, ff_gamma=ff_gamma, ff_amin=ff_amin,
                ff_apply_on=ff_apply_on, ff_use_drv_gate=ff_use_drv_gate
            )
        self.resmix = SoftResMix(init=0.0) if self.use_resmix else None
        # PCA wrapper（默认不开启，需 attach_pca_manager 后生效）
        self.pca_wrap = PCADenoiseWrapper(layer_id=-1, pca_mgr=pca_mgr, k=pca_k, a_min=pca_amin, blend=1.0)
        self._layer_id: Optional[int] = None  # 由外层设置真实层号

    def set_layer_id(self, lid: int):
        self._layer_id = int(lid)
        if self.pca_wrap is not None:
            self.pca_wrap.layer_id = self._layer_id

    def attach_pca_manager(self, pca_mgr):
        if self.pca_wrap is not None:
            self.pca_wrap.pca_mgr = pca_mgr

    def forward(self, hidden_states, *args, **kwargs):
        h_in = hidden_states  # [B,T,H]

        # --- PCA 统计/缓存/降噪 ---
        if self.pca_wrap is not None and self._layer_id is not None:
            # 训练：仅统计 train 端（不降噪）；评估：缓存 last_input，并应用降噪（若有 gains）
            if self.training and self.pca_wrap.pca_mgr is not None:
                self.pca_wrap.pca_mgr.update_train(self._layer_id, h_in.detach())
            else:
                # cache for quickval overlap scoring
                self.pca_wrap._last_input = h_in.detach()
            h_in = self.pca_wrap(h_in)

        # --- 低通/FFT 门控 + SoftResMix ---
        gated = self.pre_attn_gate(h_in) if self.pre_attn_gate is not None else h_in
        x = self.resmix(h_in, gated) if self.resmix is not None else gated

        # --- 原始 HF 层 ---
        outputs = self.layer(x, *args, **kwargs)
        return outputs


@MODEL_REGISTRY.register("distillbert_hf_gate")
class DistillBertHFGate(nn.Module):
    """DistilBERT + （可选）门控 / ResMix / PCA 降噪。
    对外暴露：n_layers / layer_modules / resmix_modules / pca_wrappers / attach_pca_manager
    """
    def __init__(self,
                 text_task="cls",
                 text_num_classes=2,
                 tok_num_classes=None,
                 pad_id=0,
                 use_gate=True,
                 use_resmix=False,
                 resmix_init=0.0,
                 gate_tau=1.0,
                 pretrained="distilbert-base-uncased",
                 # 插桩相关
                 insert_layers: str = 'last',
                 filter_backend: str = 'lp_fixed',
                 lp_k: int = 7,
                 feature_mask: bool = False,
                 ff_gamma: float = 1.0,
                 ff_amin: float = 0.8,
                 ff_apply_on: str = 'input',
                 ff_use_drv_gate: bool = False,
                 # PCA 降噪
                 pca_enable: bool = True,
                 pca_k: int = 32,
                 pca_amin: float = 0.85):
        super().__init__()
        self.text_task = text_task
        self.text_num_classes = text_num_classes
        self.tok_num_classes = tok_num_classes
        self.pad_id = pad_id

        self.config = DistilBertConfig.from_pretrained(pretrained)
        self.bert = DistilBertModel.from_pretrained(pretrained, config=self.config)
        dim = self.config.hidden_size

        n_layers = len(self.bert.transformer.layer)
        insert_ids = _parse_insert_layers(insert_layers, n_layers)

        # wrap transformer layers
        new_layers = []
        self.layer_modules: List[DistilBertLayerWithAdapters] = []
        self.resmix_modules: List[SoftResMix] = []
        self.pca_wrappers: Dict[int, PCADenoiseWrapper] = {}

        for i, layer in enumerate(self.bert.transformer.layer):
            use_here = (i in insert_ids)
            mod = DistilBertLayerWithAdapters(
                layer=layer, dim=dim,
                use_gate=use_here and use_gate,
                use_resmix=use_here and use_resmix,
                filter_backend=filter_backend, gate_tau=gate_tau, lp_k=lp_k,
                feature_mask=feature_mask, ff_gamma=ff_gamma, ff_amin=ff_amin,
                ff_apply_on=ff_apply_on, ff_use_drv_gate=ff_use_drv_gate,
                pca_mgr=None if not pca_enable else None, pca_k=pca_k, pca_amin=pca_amin
            )
            mod.set_layer_id(i)
            new_layers.append(mod)
            self.layer_modules.append(mod)
            if mod.resmix is not None:
                self.resmix_modules.append(mod.resmix)
            if mod.pca_wrap is not None:
                self.pca_wrappers[i] = mod.pca_wrap

        self.bert.transformer.layer = nn.ModuleList(new_layers)

        # heads
        if text_task == "cls":
            self.cls_head = nn.Linear(dim, text_num_classes)
        elif text_task == "tokcls":
            assert tok_num_classes is not None, "tok_num_classes must be provided for tokcls"
            self.tok_head = nn.Linear(dim, tok_num_classes)
        else:
            raise ValueError("text_task must be 'cls' or 'tokcls'")

    # --- Introspection for trainer ---
    @property
    def n_layers(self) -> int:
        return len(self.layer_modules)

    def attach_pca_manager(self, pca_mgr):
        for m in self.layer_modules:
            m.attach_pca_manager(pca_mgr)

    def controller_kernel_weights(self):
        ws = []
        for m in self.layer_modules:
            if m.pre_attn_gate is not None and hasattr(m.pre_attn_gate, 'convs'):
                for conv in m.pre_attn_gate.convs:
                    ws.append(conv.weight)
            elif m.pre_attn_gate is not None and hasattr(m.pre_attn_gate, 'conv'):
                ws.append(m.pre_attn_gate.conv.weight)
        return ws

    def forward(self, x, attention_mask=None, **kwargs):
        out = self.bert(input_ids=x, attention_mask=attention_mask, **kwargs)
        last = out.last_hidden_state  # [B,T,H]
        if self.text_task == "cls":
            if attention_mask is None:
                # guess pad by zeros? safer: assume all valid
                feat = last.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).float()
                feat = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return {'logits': self.cls_head(feat), 'labels': kwargs.get('labels', None)}
        else:
            return {'logits': self.tok_head(last), 'labels': kwargs.get('labels', None)}

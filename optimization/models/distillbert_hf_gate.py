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


# def _parse_insert_layers(arg: str, n_layers: int) -> List[int]:
#     a = (arg or 'last').lower()
#     if a == 'last':
#         return [n_layers - 1]
#     if a in ('last2', 'last-2', 'last_2'):
#         return [max(0, n_layers - 2), n_layers - 1]
#     if a == 'all':
#         return list(range(n_layers))
#     try:
#         idxs = [int(x.strip()) for x in arg.split(',') if x.strip() != '']
#         return [i for i in sorted(set(idxs)) if 0 <= i < n_layers]
#     except Exception:
#         return [n_layers - 1]


def _parse_insert_layers(arg, n_layers):
    """
    Parse insert layer spec.
    Examples: "last" -> [n_layers-1], "all" -> [0..n_layers-1], "0,3,5"
    Empty/none/off -> [] (no insertion)
    """
    if arg is None:
        return []
    a = str(arg).strip().lower()
    if a in ("", "none", "no", "off", "false", "0"):
        return []
    if a == "last":
        return [n_layers - 1]
    if a == "all":
        return list(range(n_layers))
    if "," in a:
        try:
            return [int(x) for x in a.split(",") if x.strip().isdigit()]
        except Exception:
            return []
    return []



class DistilBertLayerWithAdapters(nn.Module):
    """
    单层 DistilBERT 外挂：
      - PCA 降噪包装（训练期更新子空间；eval/quickval 投影-重建-混合）
      - 1D 低通门（可选 FFT 特征频域掩码）
      - SoftResMix 软残差
    顺序：PCA → Gate(+FFT) → ResMix → 原始 HF 层
    """
    def __init__(self,
                 layer: nn.Module,
                 dim: int,
                 use_gate: bool,
                 use_resmix: bool,
                 resmix_init: float,
                 # filter / fft
                 filter_backend: str,
                 gate_tau: float,
                 lp_k: int,
                 feature_mask: bool,
                 ff_gamma: float,
                 ff_amin: float,
                 ff_apply_on: str,
                 ff_use_drv_gate: bool,
                 # PCA
                 pca_enable: bool,
                 pca_k: int,
                 pca_amin: float):
        super().__init__()
        self.layer = layer
        self.use_gate = bool(use_gate)
        self.use_resmix = bool(use_resmix)

        # PCA wrapper（默认启用可配置；训练端更新子空间、eval时可降噪）
        self.pca_wrap = PCADenoiseWrapper(
            layer_id=-1, pca_mgr=None, k=pca_k, a_min=pca_amin, blend=1.0
        ) if pca_enable else None

        # 低通门（可带 FFT 掩码）
        self.pre_attn_gate = None
        if self.use_gate:
            self.pre_attn_gate = make_filter_1d(
                channels=dim, backend=filter_backend, ks_list=(3, 7, 15),
                tau=gate_tau, lp_k=lp_k,
                feature_mask=feature_mask,
                ff_gamma=ff_gamma, ff_amin=ff_amin,
                ff_apply_on=ff_apply_on, ff_use_drv_gate=ff_use_drv_gate
            )

        # SoftResMix 残差门
        self.resmix = SoftResMix(init=resmix_init) if self.use_resmix else None

        self._layer_id: Optional[int] = None

    # ==== 外部控制钩子 ====
    def set_layer_id(self, lid: int):
        self._layer_id = int(lid)
        if self.pca_wrap is not None:
            self.pca_wrap.layer_id = self._layer_id

    def attach_pca_manager(self, pca_mgr):
        if self.pca_wrap is not None:
            self.pca_wrap.pca_mgr = pca_mgr

    def set_tau(self, tau: float):
        if self.pre_attn_gate is not None and hasattr(self.pre_attn_gate, "set_tau"):
            self.pre_attn_gate.set_tau(tau)

    # ==== 前向 ====
    def forward(self, hidden_states, *args, **kwargs):
        x = hidden_states  # [B,T,H]

        # 1) PCA：训练端更新子空间；eval/quickval 缓存 + 降噪
        if self.pca_wrap is not None and self._layer_id is not None:
            if self.training and self.pca_wrap.pca_mgr is not None:
                self.pca_wrap.pca_mgr.update_train(self._layer_id, x.detach())
            else:
                self.pca_wrap._last_input = x.detach()
            x = self.pca_wrap(x)

        # 2) Gate(+FFT) + ResMix
        if self.pre_attn_gate is not None:
            g = self.pre_attn_gate(x)
            x = self.resmix(x, g) if self.resmix is not None else g

        # 3) 原始 HF 层
        outputs = self.layer(x, *args, **kwargs)
        return outputs


@MODEL_REGISTRY.register("distillbert_hf_gate")
class DistillBertHFGate(nn.Module):
    """
    DistilBERT + （可选）1D 低通/FFT 门控 + SoftResMix + PCA 降噪
    对外暴露：
      - n_layers / layer_modules / resmix_modules / pca_wrappers
      - attach_pca_manager(pca_mgr)
      - scan_feature_fft_wrappers() / scan_gate_modules()
      - controller_kernel_weights() 供 DCA
    """
    def __init__(self,
                 text_task: str = "cls",          # "cls" | "tokcls"
                 text_num_classes: int = 2,
                 tok_num_classes: Optional[int] = None,
                 pad_id: int = 0,
                 # 插桩与门控
                 use_gate: bool = True,
                 use_resmix: bool = False,
                 resmix_init: float = 0.0,
                 gate_tau: float = 1.0,
                 insert_layers: str = "last",
                 filter_backend: str = "lp_fixed",
                 lp_k: int = 7,
                 feature_mask: bool = False,
                 ff_gamma: float = 1.0,
                 ff_amin: float = 0.8,
                 ff_apply_on: str = "input",
                 ff_use_drv_gate: bool = False,
                 # PCA
                 pca_enable: bool = True,
                 pca_k: int = 32,
                 pca_amin: float = 0.85,
                 # 预训练
                 pretrained: str = "distilbert-base-uncased"):
        super().__init__()
        self.text_task = text_task
        self.text_num_classes = text_num_classes
        self.tok_num_classes = tok_num_classes
        self.pad_id = pad_id

        # HF backbone
        self.config = DistilBertConfig.from_pretrained(pretrained)
        self.bert = DistilBertModel.from_pretrained(pretrained, config=self.config)
        dim = self.config.hidden_size

        # 包装每一层
        n_layers = len(self.bert.transformer.layer)
        insert_ids = _parse_insert_layers(insert_layers, n_layers)

        self.layer_modules: List[DistilBertLayerWithAdapters] = []
        self.resmix_modules: List[SoftResMix] = []
        self.pca_wrappers: Dict[int, PCADenoiseWrapper] = {}
        self._fft_modules: List[nn.Module] = []
        self._gate_modules: List[nn.Module] = []

        new_layers = []
        for i, layer in enumerate(self.bert.transformer.layer):
            use_here = (i in insert_ids)
            mod = DistilBertLayerWithAdapters(
                layer=layer, dim=dim,
                use_gate=(use_here and use_gate),
                use_resmix=(use_here and use_resmix),
                resmix_init=resmix_init,
                filter_backend=filter_backend, gate_tau=gate_tau, lp_k=lp_k,
                feature_mask=feature_mask, ff_gamma=ff_gamma, ff_amin=ff_amin,
                ff_apply_on=ff_apply_on, ff_use_drv_gate=ff_use_drv_gate,
                pca_enable=pca_enable, pca_k=pca_k, pca_amin=pca_amin
            )
            mod.set_layer_id(i)
            new_layers.append(mod)
            self.layer_modules.append(mod)
            if mod.resmix is not None:
                self.resmix_modules.append(mod.resmix)
            if mod.pca_wrap is not None:
                self.pca_wrappers[i] = mod.pca_wrap
            if mod.pre_attn_gate is not None and use_here:
                self._gate_modules.append(mod.pre_attn_gate)
                fm = getattr(mod.pre_attn_gate, "feature_mask", None)
                if fm is not None:
                    self._fft_modules.append(fm)

        self.bert.transformer.layer = nn.ModuleList(new_layers)

        # 任务头
        if text_task == "cls":
            self.cls_head = nn.Linear(dim, text_num_classes)
        elif text_task == "tokcls":
            assert tok_num_classes is not None, "tok_num_classes must be provided for tokcls"
            self.tok_head = nn.Linear(dim, tok_num_classes)
        else:
            raise ValueError("text_task must be 'cls' or 'tokcls'")


    def disable_all_inserts(self):
        """Hard remove all optional gate/filter/PCA/ResMix modules."""
        if hasattr(self, "gates"):
            self.gates = nn.ModuleList([])
        if hasattr(self, "filters"):
            self.filters = nn.ModuleList([])
        if hasattr(self, "pca_wrapper"):
            self.pca_wrapper = None
        if hasattr(self, "resmix"):
            self.resmix = None
        if hasattr(self, "insert_layers"):
            self.insert_layers = []


    # ===== 训练脚本用的接口 =====
    @property
    def n_layers(self) -> int:
        return len(self.layer_modules)

    def attach_pca_manager(self, pca_mgr):
        for m in self.layer_modules:
            m.attach_pca_manager(pca_mgr)

    def scan_feature_fft_wrappers(self) -> List[nn.Module]:
        return list(self._fft_modules)

    def scan_gate_modules(self) -> List[nn.Module]:
        return list(self._gate_modules)

    def controller_kernel_weights(self):
        ws = []
        for m in self.layer_modules:
            g = getattr(m, "pre_attn_gate", None)
            if g is None:
                continue
            if hasattr(g, "convs"):
                for conv in g.convs:
                    ws.append(conv.weight)
            elif hasattr(g, "conv"):
                ws.append(g.conv.weight)
            elif hasattr(g, "ops"):
                for op in g.ops:
                    if hasattr(op, "weight"):
                        ws.append(op.weight)
        return ws

    # ===== 前向 =====
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        # x: input_ids [B,T]
        if attention_mask is None:
            attention_mask = (x != self.pad_id).long()

        out = self.bert(input_ids=x, attention_mask=attention_mask, **kwargs)
        last = out.last_hidden_state  # [B,T,H]

        if self.text_task == "cls":
            mask = attention_mask.unsqueeze(-1).float()          # [B,T,1]
            feat = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            logits = self.cls_head(feat)                         # [B,C]
            return {'logits': logits, 'labels': kwargs.get('labels', None)}
        else:  # tokcls
            logits = self.tok_head(last)                         # [B,T,C]
            return {'logits': logits, 'labels': kwargs.get('labels', None)}

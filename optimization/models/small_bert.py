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

        # === PCA wrapper：quickval 时设置门，训练时更新子空间 ===
        self.pca_wrap = PCADenoiseWrapper(
            layer_id=layer_id, pca_mgr=None, k=pca_k, a_min=pca_amin, blend=1.0
        ) if enable_pca else None

        # 低通门（可带特征频域掩码）
        self.pre_attn_gate = None
        if self.use_gate:
            self.pre_attn_gate = make_filter_1d(
                channels=dim, backend=filter_backend, ks_list=ks_list, tau=gate_tau, lp_k=lp_k,
                feature_mask=feature_mask, ff_gamma=ff_gamma, ff_amin=ff_amin,
                ff_apply_on=ff_apply_on, ff_use_drv_gate=ff_use_drv_gate
            )

        # SoftResMix 残差门
        self.resmix = SoftResMix(init=resmix_init) if self.use_resmix else None

        # 标准 BERT block
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop)
        )

    # 由训练脚本注入 PCA 管理器
    def attach_pca_manager(self, pca_mgr):
        if self.pca_wrap is not None:
            self.pca_wrap.pca_mgr = pca_mgr

    # quickval 可调用（LP-DARTS 冷却/尖化）
    def set_tau(self, tau: float):
        if self.use_gate and hasattr(self.pre_attn_gate, "set_tau"):
            self.pre_attn_gate.set_tau(tau)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # === 1) PCA（与 gate 解耦；始终先做） ===
        if self.pca_wrap is not None:
            if self.training and self.pca_wrap.pca_mgr is not None:
                # 训练端：更新子空间
                self.pca_wrap.pca_mgr.update_train(self.layer_id, x.detach())
            else:
                # quickval/推理：缓存验证窗
                self.pca_wrap._last_input = x.detach()
            x = self.pca_wrap(x)  # 投影-重建-混合

        # === 2) 低通门 + SoftResMix（可选） ===
        h_in = x
        gated = self.pre_attn_gate(x) if self.use_gate else x
        x = self.resmix(h_in, gated) if self.use_resmix else gated

        # === 3) 标准 BERT block ===
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

        # === Text embeddings ===
        self.text_vocab_size = vocab_size
        self.text_pos = nn.Embedding(max_len + (1 if cls_token else 0), dim)
        self.tok = nn.Embedding(vocab_size, dim, padding_idx=pad_id)

        # === Image path (可选) ===
        self.img_embed = nn.Conv2d(in_ch, dim, kernel_size=img_patch, stride=img_patch)

        # 解析插装层
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

        # 容器（给 quickval 扫描）
        self._fft_modules: List[nn.Module] = []
        self._gate_modules: List[nn.Module] = []
        self.pca_wrappers: Dict[int, PCADenoiseWrapper] = {}

        # Blocks
        self.blocks = nn.ModuleList()
        self.layer_modules: List[BertBlock1D] = []
        self.resmix_modules: List[SoftResMix] = []

        for i in range(n_layers):
            use_here = (i in insert_ids)
            b = BertBlock1D(dim, heads, mlp_ratio, attn_drop, drop,
                            use_gate=(use_here and use_gate),
                            gate_tau=gate_tau,
                            use_resmix=(use_here and use_resmix),
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
            if use_here and getattr(b, 'pre_attn_gate', None) is not None:
                self._gate_modules.append(b.pre_attn_gate)
                # 如果 1D gate 外挂了特征频域掩码，工厂里通常存放在 gate.feature_mask
                fm = getattr(b.pre_attn_gate, "feature_mask", None)
                if fm is not None:
                    self._fft_modules.append(fm)

        self.ln = nn.LayerNorm(dim)
        self.img_head = nn.Linear(dim, num_classes)
        self.textcls_head = nn.Linear(dim, text_num_classes) if text_task == 'cls' and text_num_classes is not None else None
        self.tokcls_head  = nn.Linear(dim, tok_num_classes)  if text_task == 'tokcls' and tok_num_classes  is not None else None

    # quickval：注入 PCA 管理器
    def attach_pca_manager(self, pca_mgr):
        for b in self.blocks:
            b.attach_pca_manager(pca_mgr)

    # quickval：收集可学习/可调的 kernel（供 DCA 或可视化）
    def controller_kernel_weights(self):
        ws = []
        for b in self.blocks:
            g = getattr(b, 'pre_attn_gate', None)
            if g is None:
                continue
            if hasattr(g, 'convs'):
                for c in g.convs:
                    ws.append(c.weight)
            elif hasattr(g, 'conv'):
                ws.append(g.conv.weight)
            elif hasattr(g, 'ops'):  # 一些实现用 ops 列表
                for op in g.ops:
                    if hasattr(op, 'weight'):
                        ws.append(op.weight)
        return ws

    # === quickval 扫描接口 ===
    def scan_feature_fft_wrappers(self) -> List[nn.Module]:
        return list(self._fft_modules)

    def scan_gate_modules(self) -> List[nn.Module]:
        return list(self._gate_modules)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # 文本路径
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
            return {'logits': h}  # LM / 其他上游任务可继续接头

        # 图像路径（保持与 small_transformer 一致的接口）
        elif x.dim() == 4:
            tok = self.img_embed(x)                     # [B,D,H',W']
            B, D, H, W = tok.shape
            h = tok.flatten(2).transpose(1,2)           # [B,T,D]
            for b in self.blocks:
                h = b(h)                                # 仍走 1D gate（沿时间/patch 序列）
            h = self.ln(h)
            h = h.mean(dim=1)
            return {'logits': self.img_head(h)}
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)} / dtype {x.dtype}")
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

        # === PCA wrapper：quickval 时设置门，训练时更新子空间 ===
        self.pca_wrap = PCADenoiseWrapper(
            layer_id=layer_id, pca_mgr=None, k=pca_k, a_min=pca_amin, blend=1.0
        ) if enable_pca else None

        # 低通门（可带特征频域掩码）
        self.pre_attn_gate = None
        if self.use_gate:
            self.pre_attn_gate = make_filter_1d(
                channels=dim, backend=filter_backend, ks_list=ks_list, tau=gate_tau, lp_k=lp_k,
                feature_mask=feature_mask, ff_gamma=ff_gamma, ff_amin=ff_amin,
                ff_apply_on=ff_apply_on, ff_use_drv_gate=ff_use_drv_gate
            )

        # SoftResMix 残差门
        self.resmix = SoftResMix(init=resmix_init) if self.use_resmix else None

        # 标准 BERT block
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop)
        )

    # 由训练脚本注入 PCA 管理器
    def attach_pca_manager(self, pca_mgr):
        if self.pca_wrap is not None:
            self.pca_wrap.pca_mgr = pca_mgr

    # quickval 可调用（LP-DARTS 冷却/尖化）
    def set_tau(self, tau: float):
        if self.use_gate and hasattr(self.pre_attn_gate, "set_tau"):
            self.pre_attn_gate.set_tau(tau)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # === 1) PCA（与 gate 解耦；始终先做） ===
        if self.pca_wrap is not None:
            if self.training and self.pca_wrap.pca_mgr is not None:
                # 训练端：更新子空间
                self.pca_wrap.pca_mgr.update_train(self.layer_id, x.detach())
            else:
                # quickval/推理：缓存验证窗
                self.pca_wrap._last_input = x.detach()
            x = self.pca_wrap(x)  # 投影-重建-混合

        # === 2) 低通门 + SoftResMix（可选） ===
        h_in = x
        gated = self.pre_attn_gate(x) if self.use_gate else x
        x = self.resmix(h_in, gated) if self.use_resmix else gated

        # === 3) 标准 BERT block ===
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

        # === Text embeddings ===
        self.text_vocab_size = vocab_size
        self.text_pos = nn.Embedding(max_len + (1 if cls_token else 0), dim)
        self.tok = nn.Embedding(vocab_size, dim, padding_idx=pad_id)

        # === Image path (可选) ===
        self.img_embed = nn.Conv2d(in_ch, dim, kernel_size=img_patch, stride=img_patch)

        # 解析插装层
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

        # 容器（给 quickval 扫描）
        self._fft_modules: List[nn.Module] = []
        self._gate_modules: List[nn.Module] = []
        self.pca_wrappers: Dict[int, PCADenoiseWrapper] = {}

        # Blocks
        self.blocks = nn.ModuleList()
        self.layer_modules: List[BertBlock1D] = []
        self.resmix_modules: List[SoftResMix] = []

        for i in range(n_layers):
            use_here = (i in insert_ids)
            b = BertBlock1D(dim, heads, mlp_ratio, attn_drop, drop,
                            use_gate=(use_here and use_gate),
                            gate_tau=gate_tau,
                            use_resmix=(use_here and use_resmix),
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
            if use_here and getattr(b, 'pre_attn_gate', None) is not None:
                self._gate_modules.append(b.pre_attn_gate)
                # 如果 1D gate 外挂了特征频域掩码，工厂里通常存放在 gate.feature_mask
                fm = getattr(b.pre_attn_gate, "feature_mask", None)
                if fm is not None:
                    self._fft_modules.append(fm)

        self.ln = nn.LayerNorm(dim)
        self.img_head = nn.Linear(dim, num_classes)
        self.textcls_head = nn.Linear(dim, text_num_classes) if text_task == 'cls' and text_num_classes is not None else None
        self.tokcls_head  = nn.Linear(dim, tok_num_classes)  if text_task == 'tokcls' and tok_num_classes  is not None else None

    # quickval：注入 PCA 管理器
    def attach_pca_manager(self, pca_mgr):
        for b in self.blocks:
            b.attach_pca_manager(pca_mgr)

    # quickval：收集可学习/可调的 kernel（供 DCA 或可视化）
    def controller_kernel_weights(self):
        ws = []
        for b in self.blocks:
            g = getattr(b, 'pre_attn_gate', None)
            if g is None:
                continue
            if hasattr(g, 'convs'):
                for c in g.convs:
                    ws.append(c.weight)
            elif hasattr(g, 'conv'):
                ws.append(g.conv.weight)
            elif hasattr(g, 'ops'):  # 一些实现用 ops 列表
                for op in g.ops:
                    if hasattr(op, 'weight'):
                        ws.append(op.weight)
        return ws

    # === quickval 扫描接口 ===
    def scan_feature_fft_wrappers(self) -> List[nn.Module]:
        return list(self._fft_modules)

    def scan_gate_modules(self) -> List[nn.Module]:
        return list(self._gate_modules)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # 文本路径
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
            return {'logits': h}  # LM / 其他上游任务可继续接头

        # 图像路径（保持与 small_transformer 一致的接口）
        elif x.dim() == 4:
            tok = self.img_embed(x)                     # [B,D,H',W']
            B, D, H, W = tok.shape
            h = tok.flatten(2).transpose(1,2)           # [B,T,D]
            for b in self.blocks:
                h = b(h)                                # 仍走 1D gate（沿时间/patch 序列）
            h = self.ln(h)
            h = h.mean(dim=1)
            return {'logits': self.img_head(h)}
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)} / dtype {x.dtype}")

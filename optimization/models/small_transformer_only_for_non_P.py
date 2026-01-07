
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from utils.registry import MODEL_REGISTRY
# from modules.filter_factory import make_filter_2d
# from modules.soft_resmix import SoftResMix


# class PatchEmbed(nn.Module):
#     def __init__(self, in_ch: int = 3, embed_dim: int = 192, patch: int = 4):
#         super().__init__()
#         self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
#         self.last_hw = None

#     def forward(self, x):
#         x = self.proj(x)  # [B,D,H',W']
#         _, D, H, W = x.shape
#         self.last_hw = (H, W)
#         return x.flatten(2).transpose(1, 2)  # [B,H'*W',D]


# class MLP(nn.Module):
#     def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
#         super().__init__()
#         hidden = int(dim * mlp_ratio)
#         self.fc1 = nn.Linear(dim, hidden)
#         self.fc2 = nn.Linear(hidden, dim)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.drop(F.gelu(self.fc1(x)))
#         return self.drop(self.fc2(x))


# class EncoderBlock(nn.Module):
#     def __init__(
#         self,
#         dim,
#         heads=3,
#         mlp_ratio=4.0,
#         attn_drop=0.0,
#         drop=0.0,
#         use_gate: bool = True,
#         use_resmix: bool = False,
#         resmix_init: float = 0.0,
#         gate_tau: float = 1.0,
#         has_cls: bool = False,
#         # filter backend options
#         filter_backend: str = "darts",
#         ks_list=(3, 7, 15),
#         lp_k: int = 7,
#         feature_mask: bool = False,
#         ff_gamma: float = 1.0,
#         ff_amin: float = 0.8,
#         ff_apply_on: str = "input",
#         ff_use_drv_gate: bool = False,
#     ):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(
#             embed_dim=dim, num_heads=heads, dropout=attn_drop, batch_first=True
#         )
#         self.use_gate = use_gate
#         self.use_resmix = use_resmix
#         self.has_cls = has_cls

#         if use_gate:
#             self.pre_attn_gate = make_filter_2d(
#                 channels=dim,
#                 backend=filter_backend,
#                 ks_list=ks_list,
#                 tau=gate_tau,
#                 lp_k=lp_k,
#                 feature_mask=feature_mask,
#                 ff_gamma=ff_gamma,
#                 ff_amin=ff_amin,
#                 ff_apply_on=ff_apply_on,
#                 ff_use_drv_gate=ff_use_drv_gate,
#             )
#         if use_resmix:
#             self.resmix = SoftResMix(init=resmix_init, learnable=True)

#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = MLP(dim, mlp_ratio, drop)
#         self.drop = nn.Dropout(drop)
#         self._hw = None  # (H,W)
#         self.dca_tap = True

#     def set_grid(self, hw):
#         self._hw = hw

#     def _infer_hw(self, N: int):
#         side = int(math.isqrt(N))
#         if side * side == N:
#             return (side, side)
#         for h in range(side, 0, -1):
#             if N % h == 0:
#                 return (h, N // h)
#         return (N, 1)

#     def _gridify(self, tok):
#         B, N, C = tok.shape
#         if self._hw is not None and self._hw[0] * self._hw[1] == N:
#             H, W = self._hw
#         else:
#             H, W = self._infer_hw(N)
#             if H * W != N:
#                 return None, None
#         return tok.transpose(1, 2).reshape(B, C, H, W), (H, W)

#     def _flatten(self, x2d):
#         B, D, H, W = x2d.shape
#         return x2d.reshape(B, D, H * W).transpose(1, 2)

#     def forward(self, x):
#         if self.use_gate:
#             if self.has_cls and x.size(1) >= 2:
#                 cls, toks = x[:, :1, :], x[:, 1:, :]
#                 x2d, _ = self._gridify(toks)
#                 if x2d is not None:
#                     x2d_raw = x2d
#                     y2d = self.pre_attn_gate(x2d)
#                     x2d = self.resmix(x2d_raw, y2d) if self.use_resmix else y2d
#                     toks = self._flatten(x2d)
#                     x = torch.cat([cls, toks], dim=1)
#             else:
#                 x2d, _ = self._gridify(x)
#                 if x2d is not None:
#                     x2d_raw = x2d
#                     y2d = self.pre_attn_gate(x2d)
#                     x2d = self.resmix(x2d_raw, y2d) if self.use_resmix else y2d
#                     x = self._flatten(x2d)

#         h = x
#         x = self.norm1(x)
#         x, _ = self.attn(x, x, x, need_weights=False)
#         x = h + self.drop(x)

#         h = x
#         x = self.norm2(x)
#         x = h + self.drop(self.mlp(x))
#         return x


# @MODEL_REGISTRY.register("small_transformer")
# class SmallTransformer(nn.Module):
#     def __init__(
#         self,
#         in_ch=3,
#         num_classes=10,
#         embed_dim=192,
#         patch=4,
#         n_layers=3,
#         heads=3,
#         mlp_ratio=4.0,
#         drop=0.1,
#         gate_tau=1.0,
#         cls_token=False,
#         use_gate: bool = True,
#         use_resmix: bool = False,
#         resmix_init: float = 0.0,
#         # pass-through filter options to blocks
#         filter_backend: str = "darts",
#         ks_list=(3, 7, 15),
#         lp_k: int = 7,
#         feature_mask: bool = False,
#         ff_gamma: float = 1.0,
#         ff_amin: float = 0.8,
#         ff_apply_on: str = "input",
#         ff_use_drv_gate: bool = False,
#     ):
#         super().__init__()
#         self.embed = PatchEmbed(in_ch, embed_dim, patch)
#         self.cls_token = cls_token
#         self.patch = patch

#         n_tok_init = (32 // patch) * (32 // patch) + (1 if cls_token else 0)
#         self.pos = nn.Parameter(torch.zeros(1, n_tok_init, embed_dim))
#         nn.init.trunc_normal_(self.pos, std=0.02)

#         if cls_token:
#             self.cls_vec = nn.Parameter(torch.zeros(1, 1, embed_dim))
#             nn.init.trunc_normal_(self.cls_vec, std=0.02)
#         else:
#             self.cls_vec = None

#         self.blocks = nn.ModuleList(
#             [
#                 EncoderBlock(
#                     embed_dim,
#                     heads,
#                     mlp_ratio,
#                     drop,
#                     drop,
#                     use_gate=use_gate,
#                     use_resmix=use_resmix,
#                     resmix_init=resmix_init,
#                     gate_tau=gate_tau,
#                     has_cls=cls_token,
#                     filter_backend=filter_backend,
#                     ks_list=ks_list,
#                     lp_k=lp_k,
#                     feature_mask=feature_mask,
#                     ff_gamma=ff_gamma,
#                     ff_amin=ff_amin,
#                     ff_apply_on=ff_apply_on,
#                     ff_use_drv_gate=ff_use_drv_gate,
#                 )
#                 for _ in range(n_layers)
#             ]
#         )
#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = nn.Linear(embed_dim, num_classes)

#     def _ensure_pos(self, needed_len: int):
#         cur_len = self.pos.size(1)
#         if needed_len <= cur_len:
#             return
#         B, _, D = self.pos.shape
#         new_pos = torch.zeros(B, needed_len, D, device=self.pos.device, dtype=self.pos.dtype)
#         new_pos[:, :cur_len, :] = self.pos.data
#         nn.init.trunc_normal_(new_pos[:, cur_len:, :], std=0.02)
#         self.pos = nn.Parameter(new_pos)

#     def forward(self, x):
#         toks = self.embed(x)
#         if self.cls_token:
#             cls = self.cls_vec.expand(toks.size(0), -1, -1)
#             toks = torch.cat([cls, toks], dim=1)
#         self._ensure_pos(toks.size(1))
#         x = toks + self.pos[:, :toks.size(1), :]

#         if self.embed.last_hw is not None:
#             grid_hw = self.embed.last_hw
#             for b in self.blocks:
#                 if hasattr(b, "set_grid"):
#                     b.set_grid(grid_hw)

#         for b in self.blocks:
#             x = b(x)
#         x = self.norm(x)
#         x = x[:, 0] if self.cls_token else x.mean(dim=1)
#         return self.head(x)

#     def arch_parameters(self):
#         params = []
#         for b in self.blocks:
#             if getattr(b, "use_gate", False) and hasattr(b, "pre_attn_gate"):
#                 if hasattr(b.pre_attn_gate, "alpha_logits"):
#                     params.append(b.pre_attn_gate.alpha_logits)
#         return params

#     def weight_parameters(self):
#         arch_ids = {id(p) for p in self.arch_parameters()}
#         return [p for p in self.parameters() if id(p) not in arch_ids]

#     def controller_kernel_weights(self):
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

#     def resmix_ratios(self):
#         ratios = []
#         for b in self.blocks:
#             if getattr(b, "use_resmix", False):
#                 ratios.append(b.resmix.get_mix_ratio())
#         return ratios


# ===============================
# File: models/small_transformer.py (patched similarly)
# ===============================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from utils.registry import MODEL_REGISTRY
from modules.filter_factory import make_filter_2d
from modules.soft_resmix import SoftResMix
from modules.pca_denoise_wrapper import PCADenoiseWrapper
# + 新增
from modules.filter_factory import make_filter_1d
from modules.soft_resmix import SoftResMix, SoftResMixNudger
from modules.pca_denoise_wrapper import PCADenoiseWrapper


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 3, embed_dim: int = 192, patch: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.last_hw = None

    def forward(self, x):
        x = self.proj(x)  # [B,D,H',W']
        _, D, H, W = x.shape
        self.last_hw = (H, W)
        return x.flatten(2).transpose(1, 2)  # [B,H'*W',D]


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(F.gelu(self.fc1(x)))
        return self.drop(self.fc2(x))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=3,
        mlp_ratio=4.0,
        attn_drop=0.0,
        drop=0.0,
        use_gate: bool = True,
        use_resmix: bool = False,
        resmix_init: float = 0.0,
        gate_tau: float = 1.0,
        has_cls: bool = False,
        # filter backend options
        filter_backend: str = "darts",
        ks_list=(3, 7, 15),
        lp_k: int = 7,
        feature_mask: bool = False,
        ff_gamma: float = 1.0,
        ff_amin: float = 0.8,
        ff_apply_on: str = "input",
        ff_use_drv_gate: bool = False,
        # PCA
        layer_id: int = -1,
        pca_enable: bool = True,
        pca_k: int = 32,
        pca_amin: float = 0.85,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=attn_drop, batch_first=True
        )
        self.use_gate = use_gate
        self.use_resmix = use_resmix
        self.has_cls = has_cls
        self.layer_id = layer_id
        self.pca_wrap = PCADenoiseWrapper(layer_id=layer_id, pca_mgr=None, k=pca_k, a_min=pca_amin, blend=1.0) if pca_enable else None

        if use_gate:
            self.pre_attn_gate = make_filter_2d(
                channels=dim,
                backend=filter_backend,
                ks_list=ks_list,
                tau=gate_tau,
                lp_k=lp_k,
                feature_mask=feature_mask,
                ff_gamma=ff_gamma,
                ff_amin=ff_amin,
                ff_apply_on=ff_apply_on,
                ff_use_drv_gate=ff_use_drv_gate,
            )
        if use_resmix:
            self.resmix = SoftResMix(init=resmix_init, learnable=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.drop = nn.Dropout(drop)
        self._hw = None  # (H,W)

    def attach_pca_manager(self, pca_mgr):
        if self.pca_wrap is not None:
            self.pca_wrap.pca_mgr = pca_mgr

    def set_grid(self, hw):
        self._hw = hw

    def _infer_hw(self, N: int):
        side = int(math.isqrt(N))
        if side * side == N:
            return (side, side)
        for h in range(side, 0, -1):
            if N % h == 0:
                return (h, N // h)
        return (N, 1)

    def _gridify(self, tok):
        B, N, C = tok.shape
        if self._hw is not None and self._hw[0] * self._hw[1] == N:
            H, W = self._hw
        else:
            H, W = self._infer_hw(N)
            if H * W != N:
                return None, None
        return tok.transpose(1, 2).reshape(B, C, H, W), (H, W)

    def _flatten(self, x2d):
        B, D, H, W = x2d.shape
        return x2d.reshape(B, D, H * W).transpose(1, 2)

    def forward(self, x):
        # PCA on token grid before gate
        if self.use_gate:
            if self.has_cls and x.size(1) >= 2:
                cls, toks = x[:, :1, :], x[:, 1:, :]
                x2d, _ = self._gridify(toks)
                if x2d is not None:
                    # PCA stats/cache/denoise on flattened tokens
                    h_tok = self._flatten(x2d)
                    if self.pca_wrap is not None:
                        if self.training and self.pca_wrap.pca_mgr is not None:
                            self.pca_wrap.pca_mgr.update_train(self.layer_id, h_tok.detach())
                        else:
                            self.pca_wrap._last_input = h_tok.detach()
                        h_tok = self.pca_wrap(h_tok)
                        x2d = h_tok.transpose(1,2).reshape(x2d.shape)
                    x2d_raw = x2d
                    y2d = self.pre_attn_gate(x2d)
                    x2d = self.resmix(x2d_raw, y2d) if self.use_resmix else y2d
                    toks = self._flatten(x2d)
                    x = torch.cat([cls, toks], dim=1)
            else:
                x2d, _ = self._gridify(x)
                if x2d is not None:
                    h_tok = self._flatten(x2d)
                    if self.pca_wrap is not None:
                        if self.training and self.pca_wrap.pca_mgr is not None:
                            self.pca_wrap.pca_mgr.update_train(self.layer_id, h_tok.detach())
                        else:
                            self.pca_wrap._last_input = h_tok.detach()
                        h_tok = self.pca_wrap(h_tok)
                        x2d = h_tok.transpose(1,2).reshape(x2d.shape)
                    x2d_raw = x2d
                    y2d = self.pre_attn_gate(x2d)
                    x2d = self.resmix(x2d_raw, y2d) if self.use_resmix else y2d
                    x = self._flatten(x2d)

        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = h + self.drop(x)

        h = x
        x = self.norm2(x)
        x = h + self.drop(self.mlp(x))
        return x


@MODEL_REGISTRY.register("small_transformer")
class SmallTransformer(nn.Module):
    def __init__(
        self,
        in_ch=3,
        num_classes=10,
        embed_dim=192,
        patch=4,
        n_layers=3,
        heads=3,
        mlp_ratio=4.0,
        drop=0.1,
        gate_tau=1.0,
        cls_token=False,
        use_gate: bool = True,
        use_resmix: bool = False,
        resmix_init: float = 0.0,
        filter_backend: str = "lp_fixed",
        ks_list=(3, 7, 15),
        lp_k: int = 7,
        feature_mask: bool = False,
        ff_gamma: float = 1.0,
        ff_amin: float = 0.8,
        ff_apply_on: str = "input",
        ff_use_drv_gate: bool = False,
        insert_layers: str = 'last',
        pca_enable: bool = True,
        pca_k: int = 32,
        pca_amin: float = 0.85,
    ):
        super().__init__()
        self.embed = PatchEmbed(in_ch, embed_dim, patch)
        self.cls_token = cls_token
        self.patch = patch

        n_tok_init = (32 // patch) * (32 // patch) + (1 if cls_token else 0)
        self.pos = nn.Parameter(torch.zeros(1, n_tok_init, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls_vec = nn.Parameter(torch.zeros(1, 1, embed_dim)) if cls_token else None
        if cls_token:
            nn.init.trunc_normal_(self.cls_vec, std=0.02)

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

        self.blocks = nn.ModuleList()
        self.layer_modules: List[EncoderBlock] = []
        self.resmix_modules: List[SoftResMix] = []
        self.pca_wrappers: Dict[int, PCADenoiseWrapper] = {}
        for i in range(n_layers):
            use_here = (i in insert_ids)
            b = EncoderBlock(
                embed_dim, heads, mlp_ratio, drop, drop,
                use_gate=use_here and use_gate,
                use_resmix=use_here and use_resmix,
                resmix_init=resmix_init, gate_tau=gate_tau, has_cls=cls_token,
                filter_backend=filter_backend, ks_list=ks_list, lp_k=lp_k,
                feature_mask=feature_mask, ff_gamma=ff_gamma, ff_amin=ff_amin,
                ff_apply_on=ff_apply_on, ff_use_drv_gate=ff_use_drv_gate,
                layer_id=i, pca_enable=pca_enable, pca_k=pca_k, pca_amin=pca_amin,
            )
            self.blocks.append(b)
            self.layer_modules.append(b)
            if b.resmix is not None:
                self.resmix_modules.append(b.resmix)
            if b.pca_wrap is not None:
                self.pca_wrappers[i] = b.pca_wrap

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def _ensure_pos(self, needed_len: int):
        cur_len = self.pos.size(1)
        if needed_len <= cur_len:
            return
        B, _, D = self.pos.shape
        new_pos = torch.zeros(B, needed_len, D, device=self.pos.device, dtype=self.pos.dtype)
        new_pos[:, :cur_len, :] = self.pos.data
        nn.init.trunc_normal_(new_pos[:, cur_len:, :], std=0.02)
        self.pos = nn.Parameter(new_pos)

    def forward(self, x):
        toks = self.embed(x)
        if self.cls_token:
            cls = self.cls_vec.expand(toks.size(0), -1, -1)
            toks = torch.cat([cls, toks], dim=1)
        self._ensure_pos(toks.size(1))
        x = toks + self.pos[:, :toks.size(1), :]

        if self.embed.last_hw is not None:
            grid_hw = self.embed.last_hw
            for b in self.blocks:
                if hasattr(b, "set_grid"):
                    b.set_grid(grid_hw)

        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        x = x[:, 0] if self.cls_token else x.mean(dim=1)
        return {'logits': self.head(x)}

    def controller_kernel_weights(self):
        ws = []
        for b in self.blocks:
            if getattr(b, "use_gate", False) and hasattr(b, "pre_attn_gate"):
                gate = b.pre_attn_gate
                if hasattr(gate, "convs"):
                    for conv in gate.convs:
                        ws.append(conv.weight)
                elif hasattr(gate, "conv"):
                    ws.append(gate.conv.weight)
        return ws

    @property
    def n_layers(self) -> int:
        return len(self.layer_modules)

    def attach_pca_manager(self, pca_mgr):
        for b in self.blocks:
            b.attach_pca_manager(pca_mgr)

# modules/laplace_spec.py
import torch
import torch.nn as nn
from typing import List, Tuple
import math
import numpy as np

EPS_GRAM = 1e-6    # 防奇异：给 Gram 矩阵加抖动
EPS_LOG  = 1e-8    # 防 log(0)
MIN_GROUPS_FOR_SPECTRUM = 2

def _flatten_tensor_list(ts: List[torch.Tensor]) -> torch.Tensor:
    """把一组参数的 grad 扁平化为一条向量。允许 grad 为 None（用 0 填充）。"""
    flats = []
    for t in ts:
        if t is None or t.grad is None:
            flats.append(torch.zeros_like(t).flatten())
        else:
            flats.append(t.grad.detach().flatten())
    if len(flats) == 0:
        return torch.tensor([], device=ts[0].device)
    return torch.cat(flats, dim=0)

def collect_group_flat_grads(groups: List[Tuple[str, List[torch.nn.Parameter]]]) -> List[torch.Tensor]:
    """
    收集每个 group 的梯度并扁平化，返回长度 = 组数 的 list，每个元素是一条 1D 向量。
    注意：梯度需在调用前通过 backward() 写入到 .grad。
    """
    out = []
    for _, params in groups:
        vec = []
        for p in params:
            if p.grad is None:
                vec.append(torch.zeros_like(p).flatten())
            else:
                vec.append(p.grad.detach().flatten())
        if len(vec) == 0:
            out.append(torch.tensor([], device=params[0].device))
        else:
            out.append(torch.cat(vec, dim=0))
    return out

# class SpectralAlign(nn.Module):
#     """
#     计算 train/val 梯度“组向量”的谱失配：
#       1) 对每个参数组，取其扁平化梯度向量，并做 L2 归一
#       2) 组成矩阵 G \in R^{G x D}（G=组数），Gram K = G G^T
#       3) 取 K 的特征值（对称正定），做 log，再对两者的 log-谱做 MSE

#     返回:
#       L_spec (float), scale \in [0,1]（可用于调节 GVA 强度的单调缩放）
#     """
#     def __init__(self, m: int = 32, center: bool = False):
#         super().__init__()
#         self.m = m
#         self.center = center

#     @torch.no_grad()
#     def _build_G(self, vecs: List[torch.Tensor]) -> torch.Tensor:
#         """把各组 1D 向量拼成 G (num_groups x D)，并逐行归一化。"""
#         if len(vecs) == 0:
#             return torch.empty(0)
#         # 对齐长度（D 可能不同：极少见，但保个底）
#         D = max([v.numel() for v in vecs])
#         rows = []
#         device = vecs[0].device
#         dtype  = vecs[0].dtype
#         for v in vecs:
#             if v.numel() < D:
#                 pad = torch.zeros(D - v.numel(), device=device, dtype=dtype)
#                 vv = torch.cat([v, pad], dim=0)
#             else:
#                 vv = v
#             # 行归一化
#             nrm = torch.linalg.norm(vv) + 1e-12
#             rows.append((vv / nrm).unsqueeze(0))
#         G = torch.cat(rows, dim=0)  # [G, D]
#         if self.center:
#             G = G - G.mean(dim=1, keepdim=True)
#         return G

#     @torch.no_grad()
#     def compute(self,
#                 groups: List[Tuple[str, List[torch.nn.Parameter]]],
#                 gsrc_tr: List[torch.Tensor],
#                 gsrc_val: List[torch.Tensor]) -> Tuple[float, float]:
#         """
#         输入:
#           groups: 分组描述（未用到名字，仅保证 gsrc_* 与 groups 对齐）
#           gsrc_tr/gsrc_val: 与 groups 等长，每个元素是“该组参数的扁平梯度向量”

#         输出:
#           (L_spec, scale)
#         """
#         # 1) 构建 G_tr / G_val
#         G_tr = self._build_G(gsrc_tr)  # [G, D]
#         G_val = self._build_G(gsrc_val)
#         G = G_tr.size(0)

#         # 如果组太少，谱信息退化，直接回退到 cos-based 的缩放并给一个很小的 L_spec
#         if G < MIN_GROUPS_FOR_SPECTRUM or G_tr.numel() == 0 or G_val.numel() == 0:
#             # 用均值向量的余弦相似做 scale
#             v_tr = G_tr.flatten()
#             v_val = G_val.flatten()
#             if v_tr.numel() == 0 or v_val.numel() == 0:
#                 return 0.0, 1.0
#             cos = torch.nn.functional.cosine_similarity(v_tr, v_val, dim=0).clamp(-1.0, 1.0)
#             scale = 0.5 * (cos + 1.0)  # [-1,1] -> [0,1]
#             return 0.0, float(scale.item())

#         # 2) Gram 矩阵 + 抖动
#         K_tr = (G_tr @ G_tr.t()).clone()
#         K_val = (G_val @ G_val.t()).clone()
#         eye = torch.eye(G, device=K_tr.device, dtype=K_tr.dtype)
#         K_tr = K_tr + EPS_GRAM * eye
#         K_val = K_val + EPS_GRAM * eye

#         # 3) 对称矩阵的特征值（升序）
#         eig_tr = torch.linalg.eigvalsh(K_tr)  # [G]
#         eig_val = torch.linalg.eigvalsh(K_val)

#         # 4) 取较小维度的后 m 个（近似主谱）
#         k = int(min(self.m, eig_tr.numel(), eig_val.numel()))
#         if k <= 0:
#             return 0.0, 1.0
#         # 末尾最大的 k 个
#         eig_tr_k = eig_tr[-k:]
#         eig_val_k = eig_val[-k:]

#         # 5) log 安全化
#         log_tr = torch.log(eig_tr_k.clamp_min(EPS_LOG))
#         log_val = torch.log(eig_val_k.clamp_min(EPS_LOG))

#         # 6) L_spec = MSE(log谱)
#         lspec = torch.mean((log_tr - log_val) ** 2)

#         # 7) scale：用平均方向的 cos（单调、稳定）
#         mean_tr = G_tr.mean(dim=0)
#         mean_val = G_val.mean(dim=0)
#         cos = torch.nn.functional.cosine_similarity(mean_tr, mean_val, dim=0).clamp(-1.0, 1.0)
#         scale = 0.5 * (cos + 1.0)  # [0,1]

#         return float(lspec.item()), float(scale.item())
class SpectralAlign:
    def __init__(self, m=32):
        self.m = int(m)

    @torch.no_grad()
    def _proj_vec(self, v: torch.Tensor) -> torch.Tensor:
        """
        把任意长度的一条梯度向量 v 投到 R^m（GPU上做）。
        使用稀疏 CountSketch 风格：随机挑 m 个索引 + Rademacher 符号；显存占用 ~O(m)。
        """
        dev   = v.device
        dtype = v.dtype
        x = v.reshape(-1).to(dev)

        d = x.numel()
        m = self.m
        # 随机索引与符号（用相同随机源可复现；如需强复现可把种子和 step/batch 做哈希）
        idx = torch.randint(0, d, (m,), device=dev)
        sgn = torch.where(torch.rand(m, device=dev) < 0.5,
                          torch.tensor(-1.0, device=dev),
                          torch.tensor(+1.0, device=dev))
        # m 维投影：每一维只取 x[idx[i]] * sgn[i]，再做 1/sqrt(m) 归一化
        y = x[idx] * sgn / math.sqrt(m)
        return y.to(dtype)

    @torch.no_grad()
    def _build_Z(self, vecs):
        Z = [self._proj_vec(v) for v in vecs]
        return torch.stack(Z, dim=0)  # [G, m]

    @torch.no_grad()
    def compute(self, groups, gsrc_tr, gsrc_val):
        Z_tr  = self._build_Z(gsrc_tr)   # [G, m] (GPU)
        Z_val = self._build_Z(gsrc_val)  # [G, m]

        # 用你之前的定义算 L_spec / scale（下面是稳妥的默认：1-cos 作为谱失配；|cos| 作为scale）
        cos = torch.nn.functional.cosine_similarity(Z_tr.flatten(), Z_val.flatten(), dim=0).clamp(-1, 1)
        L_spec = float((1.0 - cos).item())
        scale  = float(abs(cos).item())
        return L_spec, scale


@torch.no_grad()
def collect_group_flat_grads(groups):
    outs = []
    for _, plist in groups:
        flat = []
        for p in plist:
            if p.grad is not None and p.grad.numel() > 0:
                flat.append(p.grad.detach().reshape(-1))
        if len(flat) == 0:
            outs.append(torch.tensor([], device=plist[0].device if len(plist)>0 else "cpu"))
        else:
            outs.append(torch.cat(flat, dim=0))
    return outs
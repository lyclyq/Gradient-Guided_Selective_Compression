# ===== FILE: outer/val2_metrics.py =====
import math
from typing import Iterable, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def _device_of(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


@torch.no_grad()
def _to_device_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    # standardize field names
    if "labels" not in out and "label" in out:
        out["labels"] = out["label"]
    return out


@torch.no_grad()
def _eval_once(model: torch.nn.Module, batch: dict) -> Tuple[float, float]:
    device = _device_of(model)
    b = _to_device_batch(batch, device)
    logits = model(input_ids=b["input_ids"], attention_mask=b.get("attention_mask", None))
    loss = F.cross_entropy(logits, b["labels"])
    acc = (logits.argmax(dim=-1) == b["labels"]).float().mean().item()
    return float(loss.item()), float(acc)


@torch.no_grad()
def robust_eval_val2(
    model: torch.nn.Module,
    adaptor,
    dataloader: Iterable,
    base_snap,
    cand_snap,
    seeds: int = 5,
) -> Tuple[float, float, float, float]:
    """
    对 base 与 candidate 快照进行稳健评估：取 dataloader 的前 `seeds` 个小批，
    分别计算 acc 的均值与差异统计，返回：
        acc_base_mean, acc_cand_mean, delta_mean, standard_error
    """
    device = _device_of(model)

    acc0, acc1 = [], []
    it = iter(dataloader)
    for _ in range(max(1, int(seeds))):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)

        b = _to_device_batch(batch, device)

        # base
        adaptor.load_snapshot_(base_snap)
        logits0 = model(input_ids=b["input_ids"], attention_mask=b.get("attention_mask", None))
        a0 = (logits0.argmax(dim=-1) == b["labels"]).float().mean().item()
        acc0.append(a0)

        # cand
        adaptor.load_snapshot_(cand_snap)
        logits1 = model(input_ids=b["input_ids"], attention_mask=b.get("attention_mask", None))
        a1 = (logits1.argmax(dim=-1) == b["labels"]).float().mean().item()
        acc1.append(a1)

    acc0 = torch.tensor(acc0, dtype=torch.float32, device=device)
    acc1 = torch.tensor(acc1, dtype=torch.float32, device=device)
    delta = acc1 - acc0

    n = max(1, len(delta))
    acc0_mean = float(acc0.mean().item())
    acc1_mean = float(acc1.mean().item())
    delta_mean = float(delta.mean().item())
    se = float(delta.std(unbiased=(n > 1)).item() / math.sqrt(n)) if n > 1 else 0.0
    return acc0_mean, acc1_mean, delta_mean, se


@torch.no_grad()
def _mask_to_BTT(mask: torch.Tensor, B: int, Tq: int, Tk: int) -> torch.Tensor:
    """
    将可能为 [B,T] / [B,Tq,Tk] / [T] / [Tq,Tk] 的 mask 统一成 [B,Tq,Tk]，
    True/1 表示可见，False/0 表示屏蔽。
    """
    if mask is None:
        return torch.ones(B, Tq, Tk, dtype=torch.bool)

    if mask.dim() == 1:
        # [T]，既当作 Tk，也可当 Tq；这里按列（key）扩展
        if mask.numel() == Tk:
            m = mask.bool().view(1, 1, Tk).expand(B, Tq, Tk)
        elif mask.numel() == Tq:
            m = mask.bool().view(1, Tq, 1).expand(B, Tq, Tk)
        else:
            raise ValueError(f"1D mask length {mask.numel()} incompatible with Tq={Tq}, Tk={Tk}")
        return m

    if mask.dim() == 2:
        # [B,T] or [Tq,Tk]
        if mask.shape == (B, Tk):
            return mask.bool().view(B, 1, Tk).expand(B, Tq, Tk)
        if mask.shape == (B, Tq):
            return mask.bool().view(B, Tq, 1).expand(B, Tq, Tk)
        if mask.shape == (Tq, Tk):
            return mask.bool().view(1, Tq, Tk).expand(B, Tq, Tk)
        raise ValueError(f"2D mask shape {tuple(mask.shape)} incompatible with B={B},Tq={Tq},Tk={Tk}")

    if mask.dim() == 3:
        m = mask.bool()
        if m.shape == (B, Tq, Tk):
            return m
        # 允许 [1,Tq,Tk] 这种广播
        if m.shape == (1, Tq, Tk):
            return m.expand(B, Tq, Tk)
        raise ValueError(f"3D mask shape {tuple(mask.shape)} incompatible with B={B},Tq={Tq},Tk={Tk}")

    raise ValueError(f"Unsupported mask dim={mask.dim()}")


@torch.no_grad()
def attn_distribution_distance(
    scores: torch.Tensor,
    ref_scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    metric: str = "js",
) -> float:
    """
    计算注意力分布之间的行级分布距离（默认 JS-divergence，按 query 维平均）。
    关键修复点：当 scores/ref 为 [B,H,T,T] 时，mask 的 batch 维需要 repeat_interleave(H)。

    返回：float（均值）
    """
    metric = str(metric).lower()
    assert metric in ("js", "kl"), f"Unsupported metric: {metric}"

    # 规范到 [B*H, Tq, Tk]
    had_heads = (scores.dim() == 4)
    if had_heads:
        # [B,H,Tq,Tk] -> [B*H, Tq, Tk]
        B, H, Tq, Tk = scores.shape
        S = scores.contiguous().view(B * H, Tq, Tk)
        R = ref_scores.contiguous().view(B * H, Tq, Tk)
        if mask is None:
            M = torch.ones(B, Tq, Tk, dtype=torch.bool, device=S.device)
        else:
            M0 = _mask_to_BTT(mask.to(S.device), B, Tq, Tk)  # [B,Tq,Tk]
            M = M0.repeat_interleave(H, dim=0)               # [B*H,Tq,Tk]
    elif scores.dim() == 3:
        B, Tq, Tk = scores.shape
        S = scores
        R = ref_scores
        M = _mask_to_BTT(mask.to(S.device) if mask is not None else None, B, Tq, Tk).to(S.device)
    elif scores.dim() == 2:
        # [Tq,Tk]（单样本）
        Tq, Tk = scores.shape
        S = scores.unsqueeze(0)
        R = ref_scores.unsqueeze(0)
        M = _mask_to_BTT(mask.to(S.device) if mask is not None else None, 1, Tq, Tk).to(S.device)
    else:
        raise ValueError(f"Unsupported scores shape: {tuple(scores.shape)}")

    # masked softmax -> 概率分布
    NEG_INF = torch.finfo(S.dtype).min
    P = F.softmax(S.masked_fill(~M, NEG_INF), dim=-1)
    Q = F.softmax(R.masked_fill(~M, NEG_INF), dim=-1)

    if metric == "kl":
        # 逐行 KL(P||Q)
        eps = 1e-8
        kl = (P * (torch.log(P + eps) - torch.log(Q + eps))).sum(dim=-1)  # [N, Tq]
        return float(kl.mean().item())

    # JS(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M),  M=0.5*(P+Q)
    eps = 1e-8
    Mmid = 0.5 * (P + Q)
    kl_pm = (P * (torch.log(P + eps) - torch.log(Mmid + eps))).sum(dim=-1)
    kl_qm = (Q * (torch.log(Q + eps) - torch.log(Mmid + eps))).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)  # [N, Tq]
    return float(js.mean().item())


@torch.no_grad()
def grad_consistency(g_base: torch.Tensor, g_cand: torch.Tensor) -> float:
    """
    计算梯度一致性 cos，相比旧版：
    - 支持空向量/零范数时返回 1.0（保守）
    - 数值稳定处理
    """
    if g_base is None or g_cand is None:
        return 1.0
    if g_base.numel() == 0 or g_cand.numel() == 0:
        return 1.0
    gb = g_base.flatten().float()
    gc = g_cand.flatten().float()
    nb = gb.norm(p=2)
    nc = gc.norm(p=2)
    if nb <= 0 or nc <= 0:
        return 1.0
    cos = F.cosine_similarity(gb.unsqueeze(0), gc.unsqueeze(0), dim=1).clamp(-1, 1).item()
    if math.isnan(cos) or math.isinf(cos):
        cos = 1.0
    return float(cos)

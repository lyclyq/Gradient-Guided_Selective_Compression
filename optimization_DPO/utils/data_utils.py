# utils/debug_utils.py
import warnings
import torch
import math

# --- 关掉 transformers 的噪音提示（你不想它刷屏） ---
try:
    import transformers
    transformers.utils.logging.set_verbosity_error()
except Exception:
    pass

# 也可以屏蔽 sklearn/pandas 等的一般 UserWarning（可选）
warnings.filterwarnings("ignore", category=UserWarning)

def _safe_stats(x: torch.Tensor):
    x = x.detach()
    if x.numel() == 0:
        return dict(mean=float('nan'), std=float('nan'), min=float('nan'), max=float('nan'), nnz=0)
    mean = float(x.float().mean().item())
    # 避免 numel=1 时 std nan 的刷屏
    std = float(x.float().std(unbiased=False).item()) if x.numel() > 1 else 0.0
    minv = float(x.min().item())
    maxv = float(x.max().item())
    nnz = int((x != 0).sum().item())
    return dict(mean=mean, std=std, min=minv, max=maxv, nnz=nnz)

def fprint_tensor(name: str, x, head_k: int = 8):
    """比原来简洁很多，只在需要的信息上说话。"""
    if x is None:
        print(f"[DBG] {name}: None")
        return
    if not torch.is_tensor(x):
        print(f"[DBG] {name}: type={type(x)} value={x}")
        return
    x = x.detach()
    st = _safe_stats(x)
    dev = str(x.device)
    dt  = str(x.dtype)
    shp = tuple(x.shape)
    nzr = 100.0 * (st["nnz"] / max(1, x.numel()))
    head = x.flatten()[:head_k].tolist()
    print(f"[DBG] {name}: shape={shp} dtype={dt} dev={dev} "
          f"| mean={st['mean']:.6g} std={st['std']:.6g} "
          f"| min={st['min']:.6g} max={st['max']:.6g} "
          f"| nnz={st['nnz']}/{x.numel()} ({nzr:.1f}%) | head={head}")

def cosine(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
    u = u.detach().float().view(-1)
    v = v.detach().float().view(-1)
    un = u.norm() + eps
    vn = v.norm() + eps
    return float((u @ v) / (un * vn))

def pairwise_cosines(vecs, name: str = "cos"):
    """vecs: List[Tensor]（同形状向量），打印一个小的余弦相似度矩阵。"""
    n = len(vecs)
    if n == 0: 
        return
    M = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            try:
                M[i][j] = cosine(vecs[i], vecs[j])
            except Exception:
                M[i][j] = float('nan')
    # 只打印前 8x8 防刷屏
    lim = min(n, 8)
    print(f"[DBG] {name} (top-left {lim}x{lim}):")
    for i in range(lim):
        row = " ".join(f"{M[i][j]:+0.3f}" for j in range(lim))
        print("   ", row)

def align_check(alpha: torch.Tensor, attn_logits: torch.Tensor, tag: str):
    """
    alpha: (H,)
    attn_logits: (B, h, T, T) 或 (B, T, T)
    确认 len(alpha) 与 T 对齐。
    """
    if attn_logits is None:
        print(f"[ALIGN] {tag}: attn_logits=None（可能未开启 probe）")
        return
    if attn_logits.dim() == 4:
        T = attn_logits.shape[-1]
    elif attn_logits.dim() == 3:
        T = attn_logits.shape[-1]
    else:
        print(f"[ALIGN] {tag}: unexpected attn shape={tuple(attn_logits.shape)}")
        return
    H = alpha.numel()
    ok = (H == T)
    print(f"[ALIGN] {tag}: len(alpha)={H} vs T={T} -> match={ok}")
    if not ok:
        print(f"[ALIGN] {tag}: !!! 长度不一致，注意 adaptor/probe_max_T 或 实际序列截断。")

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a is None or b is None:
        return float('nan')
    return float((a.detach().float() - b.detach().float()).abs().max().item())

def vector_noise_like(a0: torch.Tensor, kind: str = "gauss", bw: int = 0):
    """
    产生与 a0 形状相同的噪声，用于候选扰动：
      - kind="gauss": 标准高斯
      - kind="smooth": 先高斯再一维平滑（带宽 bw>0）
    """
    z = torch.randn_like(a0)
    if kind == "smooth" and bw > 0 and a0.dim() == 1 and a0.numel() > 4*bw:
        # 简单 1D 滑窗平滑（避免太尖锐的抖动）
        k = 2*bw+1
        pad = (bw, bw)
        z2 = torch.nn.functional.pad(z.view(1,1,-1), pad, mode="reflect")
        ker = torch.ones(1,1,k, device=z.device, dtype=z.dtype) / k
        z = torch.nn.functional.conv1d(z2, ker).view(-1)
    return z

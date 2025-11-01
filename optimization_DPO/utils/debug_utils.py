# utils/debug_utils.py
from __future__ import annotations

import warnings
from typing import Any, Iterable, Optional

import numpy as np
import torch


def _to_tensor(x: Any) -> Optional[torch.Tensor]:
    """Best-effort convert to tensor; return None if impossible."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x
    try:
        return torch.as_tensor(x)
    except Exception:
        return None


def _device_str(t: torch.Tensor) -> str:
    if not torch.is_tensor(t):
        return "n/a"
    return f"cuda:{t.device.index}" if t.is_cuda else "cpu"


def _addr_ptr(t: torch.Tensor) -> int:
    """Get a stable-ish pointer for hashing/inspection."""
    try:
        return int(t.storage().data_ptr())
    except Exception:
        try:
            return int(t.data_ptr())
        except Exception:
            return 0


def _hash8_from_ptr(ptr: int) -> str:
    """Make a 16-hex pseudo-hash from a pointer (no heavy ops)."""
    # 64-bit multiplicative hash (Knuth's constant)
    return f"{(ptr * 11400714819323198485) & ((1 << 64) - 1):016x}"


def _numel_safe(t: torch.Tensor) -> int:
    try:
        return int(t.numel())
    except Exception:
        return 0


def _head_list(t_cpu: torch.Tensor, k: int = 4):
    try:
        flat = t_cpu.reshape(-1)
        k = min(k, flat.numel())
        return flat[:k].tolist()
    except Exception:
        return []


def fp(x: Any, title: str = "") -> None:
    """
    Pretty-print a tensor/array/number's quick stats without throwing warnings.

    - For numel == 0: prints NaNs for stats where appropriate.
    - For numel == 1: std := 0.0 (avoid DoF warnings).
    - For numel >= 2: std uses unbiased=False to avoid warnings.
    - Works with CPU/GPU tensors and non-tensor inputs (best-effort).
    """
    try:
        t = _to_tensor(x)
        if t is None:
            print(f"[FP] {title}: <None or non-tensor-like>")
            return

        # Keep original dtype/device for header; do stats on CPU float32
        dev_str = _device_str(t)
        dtype = str(t.dtype)

        # Safe CPU copy for stats
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_cpu = t.detach().to("cpu", dtype=torch.float32, copy=True)

        n = _numel_safe(t_cpu)
        shape = tuple(t.shape) if hasattr(t, "shape") else ()

        # Stats (careful branches to avoid DoF warnings)
        if n == 0:
            mean = float("nan")
            absmean = float("nan")
            std = 0.0
            vmin = float("nan")
            vmax = float("nan")
            nnz = 0
            nzr = 0.0
            head = []
        else:
            mean = float(t_cpu.mean())
            absmean = float(t_cpu.abs().mean())
            std = float(t_cpu.std(unbiased=False)) if n > 1 else 0.0  # ← fix: no DoF warning
            vmin = float(t_cpu.min())
            vmax = float(t_cpu.max())
            # Count non-zeros (NaNs count as non-zero by != 0.0; that is fine for a quick sparsity feel)
            try:
                nnz = int((t_cpu != 0).sum())
            except Exception:
                nnz = n
            nzr = 100.0 * (nnz / n) if n > 0 else 0.0
            head = _head_list(t_cpu, 4)

        addr = _addr_ptr(t)
        h8 = _hash8_from_ptr(addr)

        print(
            f"[FP] {title}: "
            f"shape={shape} dtype={dtype} dev={dev_str} addr={addr} hash8={h8} "
            f"mean={mean:.6g} absmean={absmean:.6g} std={std:.6g} "
            f"min={vmin:.6g} max={vmax:.6g} nnz={nnz} nzr={nzr:.3f}% head={head}"
        )
    except Exception as e:
        print(f"[FP] ERROR {title}: {e}")


def fp_many(items: Iterable[Any], prefix: str = "X") -> None:
    """
    Convenience: print a small collection with indexed titles.
    Example:
        fp_many([t1, t2, t3], prefix="VAL1.cand")
    """
    try:
        for i, obj in enumerate(items):
            fp(obj, title=f"{prefix}[{i}]")
    except Exception as e:
        print(f"[FP] ERROR fp_many({prefix}): {e}")


__all__ = ["fp", "fp_many"]

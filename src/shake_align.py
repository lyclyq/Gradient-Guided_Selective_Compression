# src/shake_align.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class BlockStats:
    C_r: float
    C_R: float
    A_b: float


class ShakeAlignController:
    """
    Implements (paper-aligned):
      - vote stats: C_r, C_R, A_b
      - Gate0 trigger
      - chi* routing
      - pull on head (either R->r or r->R)
      - compensation:
          g_hi <- tail(g_R_exec) + P(head(g_R_exec) - g_r_exec)
    """

    def __init__(self, cfg: Dict[str, Any], lora_modules: Optional[Dict[str, torch.nn.Module]] = None):
        self.cfg = cfg
        self.eps = 1e-8

        ours = cfg.get("method", {}).get("ours", {}) or {}
        self.V = int(ours.get("votes", 8))

        # legacy EMA window
        self.H_legacy = int(ours.get("ema_H", 4))

        # history smoothing config
        hist = ours.get("history", {}) or {}
        self.hist_enabled = bool(hist.get("enabled", False))
        self.hist_window = int(hist.get("window_steps", self.H_legacy))
        self.hist_weighting = str(hist.get("weighting", "exp"))
        self.hist_beta = float(hist.get("exp_beta", 0.7))

        self._hist: Dict[str, Deque[BlockStats]] = {}
        self._ema: Dict[str, BlockStats] = {}

        self.lora_modules: Dict[str, torch.nn.Module] = lora_modules or {}

    def set_lora_modules(self, lora_modules: Dict[str, torch.nn.Module]) -> None:
        self.lora_modules = lora_modules

    # -------------------------
    # Stats
    # -------------------------
    def compute_stats_from_votes(self, votes_r: torch.Tensor, votes_hi: torch.Tensor) -> BlockStats:
        V = votes_r.shape[0]
        if V < 2:
            return BlockStats(C_r=0.0, C_R=0.0, A_b=1.0)

        # Gram matrices
        D_r = votes_r @ votes_r.t()
        if votes_hi is not None and votes_hi.numel():
            D_hi = votes_hi @ votes_hi.t()
        else:
            D_hi = torch.zeros_like(D_r)

        D_R = D_r + D_hi

        # diagonal energies
        S_r = torch.diag(D_r).clamp_min(0)
        S_R = torch.diag(D_R).clamp_min(0)

        # upper triangle mask (exclude diagonal)
        mask = torch.triu(torch.ones((V, V), device=votes_r.device, dtype=torch.bool), diagonal=1)

        denom_r = (S_r.sqrt().unsqueeze(1) * S_r.sqrt().unsqueeze(0) + self.eps)
        denom_R = (S_R.sqrt().unsqueeze(1) * S_R.sqrt().unsqueeze(0) + self.eps)

        C_r = (D_r[mask] / denom_r[mask]).mean().item()
        C_R = (D_R[mask] / denom_R[mask]).mean().item()

        sigma_r_sq = D_r.sum()
        sigma_hi_sq = (
            D_hi.sum()
            if (votes_hi is not None and votes_hi.numel())
            else torch.tensor(0.0, device=votes_r.device)
        )
        sigma_R_sq = sigma_r_sq + sigma_hi_sq

        A_b = (sigma_r_sq.sqrt() / (sigma_R_sq.sqrt() + self.eps)).item()

        return BlockStats(C_r=float(C_r), C_R=float(C_R), A_b=float(A_b))

    # -------------------------
    # Smoothing
    # -------------------------
    def ema_update(self, name: str, fresh: BlockStats) -> BlockStats:
        if self.hist_enabled:
            return self._update_history(name, fresh)

        ours = (self.cfg.get("method", {}) or {}).get("ours", {}) or {}
        H = ours.get("ema_H", 1)
        try:
            H = int(H)
        except Exception:
            H = 1

        if H <= 1:
            return fresh

        beta = float(H - 1) / float(H)
        prev = self._ema.get(name, None)

        if prev is None:
            out = fresh
        else:
            out = BlockStats(
                C_r=float(beta * prev.C_r + (1.0 - beta) * fresh.C_r),
                C_R=float(beta * prev.C_R + (1.0 - beta) * fresh.C_R),
                A_b=float(beta * prev.A_b + (1.0 - beta) * fresh.A_b),
            )

        self._ema[name] = out
        return out

    def _weighted_average(self, seq: List[BlockStats]) -> BlockStats:
        n = len(seq)
        if n == 0:
            return BlockStats(0.0, 0.0, 1.0)

        if self.hist_weighting == "uniform":
            w = np.ones((n,), dtype=np.float64)
        elif self.hist_weighting == "linear":
            w = np.arange(1, n + 1, dtype=np.float64)
        else:
            beta = float(self.hist_beta)
            w = np.array([beta ** (n - 1 - i) for i in range(n)], dtype=np.float64)

        w = w / max(1e-12, float(np.sum(w)))

        Cr = float(np.sum([w[i] * float(seq[i].C_r) for i in range(n)]))
        CR = float(np.sum([w[i] * float(seq[i].C_R) for i in range(n)]))
        Ab = float(np.sum([w[i] * float(seq[i].A_b) for i in range(n)]))

        return BlockStats(C_r=Cr, C_R=CR, A_b=Ab)

    def _update_history(self, name: str, s: BlockStats) -> BlockStats:
        if not self.hist_enabled:
            return s

        dq = self._hist.get(name, None)
        if dq is None:
            dq = deque(maxlen=int(self.hist_window))
            self._hist[name] = dq

        dq.append(s)
        return self._weighted_average(list(dq))

    # -------------------------
    # Utils
    # -------------------------
    def _sigmoid(self, x: float) -> float:
        x = float(x)
        if x >= 0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        else:
            z = np.exp(x)
            return float(z / (1.0 + z))

    def _cos(self, a: torch.Tensor, b: torch.Tensor) -> float:
        na = float(torch.norm(a).item())
        nb = float(torch.norm(b).item())
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(torch.dot(a, b).item() / (na * nb + self.eps))

    def _pad_or_trim(self, vec: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Safe shape align WITHOUT semantic repeat:
          - if shorter: pad zeros
          - if longer: trim
        """
        dim = int(dim)
        if vec.numel() == dim:
            return vec
        if vec.numel() > dim:
            return vec[:dim]
        pad = torch.zeros((dim - vec.numel(),), device=vec.device, dtype=vec.dtype)
        return torch.cat([vec, pad], dim=0)

    def _repeat_to_dim(self, head_vec: torch.Tensor, hi_dim: int) -> torch.Tensor:
        """
        Legacy debug fallback only (NOT mathematically meaningful).
        Keep it so old runs don't crash, but the main path should avoid it.
        """
        if head_vec.numel() == 0:
            return torch.zeros((hi_dim,), device=head_vec.device, dtype=head_vec.dtype)
        rep = int(np.ceil(hi_dim / head_vec.numel()))
        return head_vec.repeat(rep)[:hi_dim]

    # -------------------------
    # Gates
    # -------------------------
    def noise_gates(self, s: BlockStats) -> Tuple[float, float]:
        ours = self.cfg.get("method", {}).get("ours", {}) or {}
        ng = ours.get("gate0_noise", None)
        if not isinstance(ng, dict):
            raise RuntimeError("[ShakeAlign] missing method.ours.gate0_noise")

        tau_n = float(ng["tau"])
        kappa_n = float(ng["kappa"])

        b_r = self._sigmoid(kappa_n * (tau_n - float(s.C_r)))
        b_R = self._sigmoid(kappa_n * (tau_n - float(s.C_R)))

        return float(b_r), float(b_R)

    def trusted_reference(self, s: BlockStats) -> str:
        ours = self.cfg.get("method", {}).get("ours", {}) or {}
        delta = float(ours.get("routing_delta", 0.02))
        dC = float(s.C_r - s.C_R)
        return "r" if (dC >= delta) else "R"

    # -------------------------
    # P mapping (the missing compensation)
    # -------------------------
    def _compensation_P(
        self,
        mod: torch.nn.Module,
        res_head: torch.Tensor,
        *,
        lam: float = 1e-4,
    ) -> torch.Tensor:
        """
        Compute P(res_head) in ΔW semantics.

        res_head corresponds to head(g_R_exec) - g_r_exec which is a residual over
        the (A_r, B_r) gradient vector.

        Steps:
          1) split res_head -> dA_r, dB_r
          2) build dW_res ≈ B_r dA_r + dB_r A_r
          3) find (dA_hi, dB_hi) such that:
                B_hi dA_hi + dB_hi A_hi ≈ dW_res
             using ridge-stabilized closed forms.
        """
        device = res_head.device
        dtype = res_head.dtype

        Ar_n = int(mod.lora_A_r.numel())
        Br_n = int(mod.lora_B_r.numel())
        need = Ar_n + Br_n

        if res_head.numel() != need:
            res_head = self._pad_or_trim(res_head, need)

        dA_r = res_head[:Ar_n].view_as(mod.lora_A_r)
        dB_r = res_head[Ar_n : Ar_n + Br_n].view_as(mod.lora_B_r)

        # detach current factors (treat as constants for mapping)
        A_r = mod.lora_A_r.detach()
        B_r = mod.lora_B_r.detach()

        # ΔW residual (first-order)
        dW_res = (B_r @ dA_r) + (dB_r @ A_r)  # [out, in]

        A_hi = mod.lora_A_hi.detach()
        B_hi = mod.lora_B_hi.detach()
        hi = int(A_hi.shape[0])  # [hi, in]

        I_hi = torch.eye(hi, device=device, dtype=dtype)

        # Solve for dB_hi: dB_hi A_hi ≈ dW_res
        # dB_hi = dW_res A_hi^T (A_hi A_hi^T + lam I)^-1
        G_A = (A_hi @ A_hi.t()) + float(lam) * I_hi  # [hi,hi]
        RHS_B = (dW_res @ A_hi.t())  # [out,hi]
        dB_hi = torch.linalg.solve(G_A.t(), RHS_B.t()).t()  # [out,hi]

        # Solve for dA_hi: B_hi dA_hi ≈ dW_res
        # dA_hi = (B_hi^T B_hi + lam I)^-1 B_hi^T dW_res
        G_B = (B_hi.t() @ B_hi) + float(lam) * I_hi  # [hi,hi]
        RHS_A = (B_hi.t() @ dW_res)  # [hi,in]
        dA_hi = torch.linalg.solve(G_B, RHS_A)  # [hi,in]

        comp = torch.cat([dA_hi.reshape(-1), dB_hi.reshape(-1)], dim=0)
        return comp

    # -------------------------
    # Execution: in-place corrections
    # -------------------------
    @torch.no_grad()
    def apply_in_place_corrections(
        self,
        lora_modules: Dict[str, torch.nn.Module],
        stats: Dict[str, BlockStats],
        vote_sums: Dict[str, Dict[str, torch.Tensor]],
        debug: bool = False,
        grad_norm_trace: bool = False,
        debug_history: bool = False,
    ) -> Dict[str, Any]:
        ours = self.cfg.get("method", {}).get("ours", {}) or {}

        trig_cfg = ours.get("trigger_gate0", {}) or {}
        tau_N = float(trig_cfg.get("tau_N", 0.6))
        tau_D = float(trig_cfg.get("tau_D", 0.25))

        pull_cfg = ours.get("pulling", {}) or {}
        gamma_pull = float(pull_cfg.get("gamma_pull", 0.35))
        k_pull = float(pull_cfg.get("k_pull", 8.0))

        # compensation knobs
        comp_cfg = ours.get("compensation", None)
        if not isinstance(comp_cfg, dict):
            raise RuntimeError("[ShakeAlign] missing method.ours.compensation")
        lam = float(comp_cfg["ridge_lambda"])
        enable_comp = bool(comp_cfg["enabled"])

        info: Dict[str, Any] = {
            "tau_N": tau_N,
            "tau_D": tau_D,
            "triggered_blocks": 0.0,
        }
        if debug:
            info["per_block"] = {}
        if grad_norm_trace:
            info["per_block_grad_norm"] = {}
        if debug_history:
            info["per_block_history"] = {}

        triggered = 0

        for name, mod in lora_modules.items():
            if name not in stats or name not in vote_sums:
                continue
            if getattr(mod, "lora_A_r", None) is None:
                continue
            if (
                mod.lora_A_r.grad is None
                or mod.lora_B_r.grad is None
                or mod.lora_A_hi.grad is None
                or mod.lora_B_hi.grad is None
            ):
                continue

            s_raw = stats[name]
            s = self._update_history(name, s_raw) if self.hist_enabled else s_raw

            pack = vote_sums[name]
            vr = pack.get("votes_r", None)
            vhi = pack.get("votes_hi", None)
            if not isinstance(vr, torch.Tensor) or vr.numel() == 0:
                continue

            if not isinstance(vhi, torch.Tensor) or vhi.numel() == 0:
                # IMPORTANT: keep hi-dim consistent (zeros), not empty
                hi_dim = int(mod.lora_A_hi.numel() + mod.lora_B_hi.numel())
                vhi = torch.zeros((vr.shape[0], hi_dim), device=vr.device, dtype=vr.dtype)

            b_r, b_R = self.noise_gates(s)

            mean_r = vr.mean(dim=0)
            mean_hi = vhi.mean(dim=0)

            g_r_prime = mean_r
            g_R_prime = torch.cat([mean_r, mean_hi], dim=0)

            head_R = g_R_prime[: g_r_prime.numel()]

            D = 1.0 - self._cos(g_r_prime, head_R)
            N_summary = float(max(b_r, b_R))
            gate0 = (N_summary >= tau_N) or (D >= tau_D)

            chi_star = self.trusted_reference(s)

            deltaC = float(s.C_r - s.C_R)
            insuff = self._sigmoid(float(k_pull) * float(deltaC))
            over = 1.0 - insuff
            alpha_pull = float(gamma_pull) * float(over) if gate0 else 0.0

            g_r_exec = g_r_prime.clone()
            g_R_exec = g_R_prime.clone()

            # pull
            if chi_star == "r":
                # R head is pulled toward r
                g_R_exec[: g_r_exec.numel()] = (1.0 - alpha_pull) * g_R_exec[: g_r_exec.numel()] + alpha_pull * g_r_exec
            else:
                # r is pulled toward R head
                g_r_exec = (1.0 - alpha_pull) * g_r_exec + alpha_pull * g_R_exec[: g_r_exec.numel()]

            # Split head/tail
            hi_dim = int(mod.lora_A_hi.numel() + mod.lora_B_hi.numel())
            head_dim = int(g_r_exec.numel())

            head_R_exec = g_R_exec[:head_dim]
            tail_R_exec = g_R_exec[head_dim:]

            # safe align tail dim WITHOUT repeat
            tail_R_exec = self._pad_or_trim(tail_R_exec, hi_dim)

            # ✅ compensation residual (paper step)
            res_head = head_R_exec - g_r_exec
            comp_hi = torch.zeros((hi_dim,), device=tail_R_exec.device, dtype=tail_R_exec.dtype)

            if enable_comp and gate0:
                try:
                    comp_hi = self._compensation_P(mod, res_head, lam=lam)
                    comp_hi = self._pad_or_trim(comp_hi, hi_dim)
                except Exception:
                    # fail-open: no compensation if solver fails
                    comp_hi = torch.zeros((hi_dim,), device=tail_R_exec.device, dtype=tail_R_exec.dtype)

            # -------------------------
            # Apply grads back (scale-invariant decisions, REAL updates)
            # -------------------------
            gr = g_r_exec

            Ar_n = int(mod.lora_A_r.grad.numel())
            Br_n = int(mod.lora_B_r.grad.numel())
            Dr = Ar_n + Br_n
            if gr.numel() != Dr:
                gr = self._pad_or_trim(gr, Dr)

            # ✅ scale back to REAL optimizer grads (r branch)
            # scaling_r = float(getattr(mod, "scaling_r", None) or getattr(mod, "scaling", 1.0))
            # gr = gr * scaling_r

            # mod.lora_A_r.grad.copy_(gr[:Ar_n].view_as(mod.lora_A_r.grad))
            # mod.lora_B_r.grad.copy_(gr[Ar_n : Ar_n + Br_n].view_as(mod.lora_B_r.grad))

            # ghi = tail_R_exec + comp_hi

            # Ahi_n = int(mod.lora_A_hi.grad.numel())
            # Bhi_n = int(mod.lora_B_hi.grad.numel())
            # Dhi = Ahi_n + Bhi_n
            # if ghi.numel() != Dhi:
            #     ghi = self._pad_or_trim(ghi, Dhi)

            # # ✅ scale back to REAL optimizer grads (hi branch)
            # scaling_hi = float(getattr(mod, "scaling_hi", None) or getattr(mod, "scaling", 1.0))

            # if not hasattr(mod, "scaling_r") or not hasattr(mod, "scaling_hi"):
            #     raise RuntimeError(f"[ShakeAlign] missing scaling_r/scaling_hi for {name}")

            # scaling_r = float(getattr(mod, "scaling_r"))
            # scaling_hi = float(getattr(mod, "scaling_hi"))

            # ghi = ghi * scaling_hi

            # mod.lora_A_hi.grad.copy_(ghi[:Ahi_n].view_as(mod.lora_A_hi.grad))
            # mod.lora_B_hi.grad.copy_(ghi[Ahi_n : Ahi_n + Bhi_n].view_as(mod.lora_B_hi.grad))


            # --- require scaling ---
            if not hasattr(mod, "scaling_r") or not hasattr(mod, "scaling_hi"):
                raise RuntimeError(f"[ShakeAlign] missing scaling_r/scaling_hi for {name}")

            scaling_r = float(getattr(mod, "scaling_r"))
            scaling_hi = float(getattr(mod, "scaling_hi"))
            if abs(scaling_r) < 1e-12 or abs(scaling_hi) < 1e-12:
                raise RuntimeError(f"[ScalingInvalid][ShakeAlign] scaling_r={scaling_r} scaling_hi={scaling_hi}")

            # --- write back r branch grads ---
            Ar_n = int(mod.lora_A_r.grad.numel())
            Br_n = int(mod.lora_B_r.grad.numel())
            Dr = Ar_n + Br_n
            gr = g_r_exec
            if gr.numel() != Dr:
                gr = self._pad_or_trim(gr, Dr)
            gr = gr * scaling_r
            mod.lora_A_r.grad.copy_(gr[:Ar_n].view_as(mod.lora_A_r.grad))
            mod.lora_B_r.grad.copy_(gr[Ar_n:Ar_n + Br_n].view_as(mod.lora_B_r.grad))

            # --- build hi branch grads ---
            ghi = tail_R_exec + comp_hi
            Ahi_n = int(mod.lora_A_hi.grad.numel())
            Bhi_n = int(mod.lora_B_hi.grad.numel())
            Dhi = Ahi_n + Bhi_n
            if ghi.numel() != Dhi:
                ghi = self._pad_or_trim(ghi, Dhi)

            ghi = ghi * scaling_hi
            mod.lora_A_hi.grad.copy_(ghi[:Ahi_n].view_as(mod.lora_A_hi.grad))
            mod.lora_B_hi.grad.copy_(ghi[Ahi_n:Ahi_n + Bhi_n].view_as(mod.lora_B_hi.grad))


            if gate0:
                triggered += 1

            if debug:
                info["per_block"][name] = {
                    "C_r": float(s.C_r),
                    "C_R": float(s.C_R),
                    "A_b": float(s.A_b),
                    "deltaC": float(deltaC),
                    "b_r": float(b_r),
                    "b_R": float(b_R),
                    "N_summary": float(N_summary),
                    "D": float(D),
                    "gate0": bool(gate0),
                    "chi_star": str(chi_star),
                    "alpha_pull": float(alpha_pull),
                    "comp_enabled": bool(enable_comp),
                    "ridge_lambda": float(lam),
                }

            if grad_norm_trace:
                info["per_block_grad_norm"][name] = {
                    "||g_r'||": float(torch.norm(g_r_prime).item()),
                    "||head(g_R')||": float(torch.norm(head_R).item()),
                    "||tail(g_R_exec)||": float(torch.norm(tail_R_exec).item()),
                    "||P(res_head)||": float(torch.norm(comp_hi).item()),
                    "scaling_r": float(scaling_r),
                    "scaling_hi": float(scaling_hi),
                }

            if debug_history and self.hist_enabled:
                dq = self._hist.get(name, None)
                if dq is not None:
                    info["per_block_history"][name] = [
                        {"C_r": float(x.C_r), "C_R": float(x.C_R), "A_b": float(x.A_b)} for x in list(dq)
                    ]

        info["triggered_blocks"] = float(triggered)
        return info

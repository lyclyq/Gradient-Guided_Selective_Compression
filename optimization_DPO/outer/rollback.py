# ===== FILE: outer/rollback.py =====
from typing import Any, Dict, Optional

class RollbackController:
    """
    决策与回滚控制：
      - mode='reject'：不修改适配器，仅记录结果
      - mode='hard'  ：失败即回滚到 snapshot
      - mode='soft'  ：失败则做轻微回滚（当前实现与 hard 一致，可按需扩展）
    兼容没有 adaptor.kl_metric(...) 的情况（优雅降级为 KL=0.0）
    """
    def __init__(self, adaptor, kl_guard: float = 0.10, mode: str = "hard",
                 base_rho: float = 0.3, tau: float = 0.02):
        self.adaptor = adaptor
        self.kl_guard = float(kl_guard)
        self.mode = mode
        self.base_rho = float(base_rho)
        self.tau = float(tau)

        self.snap: Optional[Any] = None
        self.last: Dict[str, Any] = {}

    def take_snapshot(self):
        """保存当前适配器状态"""
        self.snap = self.adaptor.snapshot()

    def _safe_kl(self, snap) -> float:
        """若 adaptor 没有 kl_metric，则优雅降级为 0.0"""
        try:
            if hasattr(self.adaptor, "kl_metric"):
                return float(self.adaptor.kl_metric(snap))
        except Exception:
            pass
        return 0.0

    def decide_and_apply(self, accept_delta: float):
        """
        传入 accept_delta（>0 表示候选相对基线的正改进），
        本函数假设“当前 adaptor 状态”已是候选；若拒绝，将回滚到 snapshot。
        """
        KL = self._safe_kl(self.snap)
        action = "reject"
        ok = False

        if accept_delta > 0:
            # 通过
            action = "accept"
            ok = True
        else:
            # 未通过：根据模式处理
            if self.mode == "hard":
                # 直接回滚
                if self.snap is not None:
                    self.adaptor.load_snapshot_(self.snap)
                action = "rollback-hard"
            elif self.mode == "soft":
                # 目前与 hard 相同；如需“部分回滚”，可在 adaptor 内实现插值式恢复
                if self.snap is not None:
                    self.adaptor.load_snapshot_(self.snap)
                action = "rollback-soft"
            else:
                action = "reject"

        self.last = {
            "action": action,
            "kl": KL,
            "accept_delta": float(accept_delta),
        }
        return ok, action

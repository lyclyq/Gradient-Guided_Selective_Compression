# modules/dca_e_ctr.py
from dataclasses import dataclass
import torch
from .dca import DCA, DCACfg

@dataclass
class DCAEctrCfg(DCACfg):
    # β：耦合强度；如果 DCACfg 里没有这个字段，也没事，继承里加就好
    beta_couple: float = 1.0

class DCAEctr(DCA):
    """
    只做 E-CTR 主体：L_DCA = sum_l (beta * D * a_l)
    - 依赖父类 DCA 的 hooks / EMA / val_cache
    - 只返回标量 loss，便于对 DWConv / α 反传
    """
    def __init__(self, cfg: DCAEctrCfg, num_classes: int):
        super().__init__(cfg, num_classes)

    def compute_loss(self, val_logits: torch.Tensor, yv: torch.Tensor):
        # 父类里实现 compute_e_ctr_loss；这里只是转发
        loss, _ = self.compute_e_ctr_loss(val_logits, yv, return_scores=False)
        return loss

# # # modules/dca_e_ctr_resmix.py
# # from dataclasses import dataclass
# # import torch
# # from .dca import DCA, DCACfg

# # @dataclass
# # class DCAEctrResMixCfg(DCACfg):
# #     beta_couple: float = 1.0
# #     guide_eps: float = 1e-8  # s_l 的稳定项

# # class DCAEctrResMix(DCA):
# #     """
# #     E-CTR + ResMix 引导：
# #     - 返回 (loss, (layer_idxs, s_l[0..1]))，用于外部更新 ResMix.beta
# #     """
# #     def __init__(self, cfg: DCAEctrResMixCfg, num_classes: int):
# #         super().__init__(cfg, num_classes)

# #     def compute_loss_and_scores(self, val_logits: torch.Tensor, yv: torch.Tensor):
# #         # 父类返回 (loss, (idxs, scores))；这里直接透传
# #         return self.compute_e_ctr_loss(val_logits, yv, return_scores=True)
# # optimization/modules/dca_e_ctr_resmix.py
# import torch

# __all__ = ["build_e_ctr_resmix", "ECTRResMix"]

# class ECTRResMix:
#     """
#     Lightweight DCA alpha-loss builder for 'e_ctr_resmix'.
#     It is expected to be called as: build_e_ctr_resmix(dca).loss()
#     """
#     def __init__(self, dca, beta_entropy: float = 1.0, beta_energy: float = 0.0):
#         """
#         dca: modules.dca.DCA 实例，需能访问 arch_parameters() / controller_kernel_weights()
#         beta_entropy: 权重，约束 gate 的分布熵（鼓励“更确定/更稀疏”或按需要调方向）
#         beta_energy : 可选，对深度可分 DWConv 的高频能量做轻正则（需有 controller_kernel_weights）
#         """
#         self.dca = dca
#         self.be = float(beta_entropy)
#         self.bw = float(beta_energy)

#     def _alpha_entropy_loss(self) -> torch.Tensor:
#         # 针对所有 gate 的 alpha 做 softmax 后的熵（越小越确定）
#         alphas = []
#         for a in (self.dca.arch_parameters() if hasattr(self.dca, "arch_parameters") else []):
#             if isinstance(a, torch.Tensor):
#                 alphas.append(a)
#         if not alphas:
#             return torch.tensor(0.0, device=self._dev(), requires_grad=True)
#         loss = 0.0
#         for a in alphas:
#             p = torch.softmax(a, dim=-1)
#             # 避免 log(0)
#             loss = loss + (p * (p.clamp_min(1e-8).log())).sum()
#         # 熵取负号（熵大→惩罚小；我们希望熵小，因此用 -sum p log p 的相反数）
#         return -loss / len(alphas)

#     def _weight_energy_loss(self) -> torch.Tensor:
#         # 可选：对控制器(深度可分卷积)的频谱能量进行极轻正则
#         if not hasattr(self.dca, "controller_kernel_weights"):
#             return torch.tensor(0.0, device=self._dev(), requires_grad=True)
#         ws = self.dca.controller_kernel_weights()
#         if not ws:
#             return torch.tensor(0.0, device=self._dev(), requires_grad=True)
#         tot = None
#         for W in ws:
#             if not isinstance(W, torch.Tensor):
#                 continue
#             spec = torch.fft.rfft(W.squeeze(1), dim=-1) if W.ndim == 3 else \
#                    torch.fft.rfftn(W.squeeze(1), dim=(-2, -1)) if W.ndim == 4 else None
#             if spec is None:
#                 continue
#             mag2 = (spec.real**2 + spec.imag**2).mean()
#             tot = mag2 if tot is None else tot + mag2
#         if tot is None:
#             return torch.tensor(0.0, device=self._dev(), requires_grad=True)
#         return tot / len(ws)

#     def _dev(self):
#         # 尽量与现有参数同设备
#         for p in self.dca.parameters():
#             return p.device
#         return torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def loss(self) -> torch.Tensor:
#         # 组合损失（可按需要调权重）
#         L = torch.tensor(0.0, device=self._dev(), requires_grad=True)
#         if self.be != 0.0:
#             L = L + self.be * self._alpha_entropy_loss()
#         if self.bw != 0.0:
#             L = L + self.bw * self._weight_energy_loss()
#         return L

# def build_e_ctr_resmix(dca, beta_entropy: float = 1.0, beta_energy: float = 0.0) -> ECTRResMix:
#     return ECTRResMix(dca, beta_entropy=beta_entropy, beta_energy=beta_energy)



# optimization/modules/dca_e_ctr_resmix.py
import torch

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_e_ctr_resmix", "ECTRResMix"]

class ECTRResMix:
    """
    Lightweight DCA alpha-loss builder for 'e_ctr_resmix'.
    It is expected to be called as: build_e_ctr_resmix(dca).loss()
    """
    def __init__(self, dca, beta_entropy: float = 1.0, beta_energy: float = 0.0):
        """
        dca: modules.dca.DCA 实例，可选地提供：
             - arch_parameters() -> Iterable[Tensor]
             - controller_kernel_weights() -> Iterable[Tensor]
             - parameters() -> Iterable[Tensor]（若无也OK）
        beta_entropy: 权重，约束 gate 的分布熵（鼓励“更确定/更稀疏”或按需要调方向）
        beta_energy : 可选，对控制器核的频谱能量轻正则
        """
        self.dca = dca
        self.be = float(beta_entropy)
        self.bw = float(beta_energy)

    def _alpha_entropy_loss(self) -> torch.Tensor:
        # 针对所有 gate 的 alpha 做 softmax 后的熵（越小越确定）
        alphas = []
        try:
            if hasattr(self.dca, "arch_parameters"):
                for a in self.dca.arch_parameters():
                    if isinstance(a, torch.Tensor):
                        alphas.append(a)
        except Exception:
            pass
        if not alphas:
            return torch.tensor(0.0, device=self._dev(), requires_grad=True)

        loss = 0.0
        eps = 1e-8
        for a in alphas:
            p = torch.softmax(a, dim=-1)
            loss = loss - (p * (p.clamp_min(eps).log())).sum()  # -∑ p log p
        return loss / len(alphas)

    def _weight_energy_loss(self) -> torch.Tensor:
        # 对控制器(如DWConv)核的频谱能量做轻正则；若接口不存在，返回0
        ws = []
        try:
            if hasattr(self.dca, "controller_kernel_weights"):
                ws = list(self.dca.controller_kernel_weights()) or []
        except Exception:
            ws = []

        if not ws:
            return torch.tensor(0.0, device=self._dev(), requires_grad=True)

        tot = None
        for W in ws:
            if not isinstance(W, torch.Tensor) or W.numel() == 0:
                continue
            if W.ndim == 3:
                spec = torch.fft.rfft(W.squeeze(1), dim=-1)
            elif W.ndim == 4:
                spec = torch.fft.rfftn(W.squeeze(1), dim=(-2, -1))
            else:
                continue
            mag2 = (spec.real**2 + spec.imag**2).mean()
            tot = mag2 if tot is None else tot + mag2
        if tot is None:
            return torch.tensor(0.0, device=self._dev(), requires_grad=True)
        return tot / max(1, len(ws))

    def _dev(self):
        # 1) 若 DCA 有 parameters()，尝试从中取设备
        try:
            if hasattr(self.dca, "parameters"):
                for p in self.dca.parameters():
                    if isinstance(p, torch.Tensor):
                        return p.device
        except Exception:
            pass
        # 2) 其次尝试 arch_parameters()
        try:
            if hasattr(self.dca, "arch_parameters"):
                for a in self.dca.arch_parameters():
                    if isinstance(a, torch.Tensor):
                        return a.device
        except Exception:
            pass
        # 3) 退化：cuda可用则用cuda，否则cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loss(self) -> torch.Tensor:
        # 组合损失（可按需要调权重）
        L = torch.tensor(0.0, device=self._dev(), requires_grad=True)
        if self.be != 0.0:
            L = L + self.be * self._alpha_entropy_loss()
        if self.bw != 0.0:
            L = L + self.bw * self._weight_energy_loss()
        return L

def build_e_ctr_resmix(dca, beta_entropy: float = 1.0, beta_energy: float = 0.0) -> ECTRResMix:
    return ECTRResMix(dca, beta_entropy=beta_entropy, beta_energy=beta_energy)

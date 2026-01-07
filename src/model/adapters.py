import torch
import torch.nn as nn
import math

def _init_lora(A, B):
    # A: out x r, B: r x in
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    nn.init.zeros_(B)

class AdapterLinear(nn.Module):
    """Wrap a frozen nn.Linear with AB (rank r) and optional alt (rank r_alt) residual.
    Forward: y = xW^T + x(AB)^T + x(U Z V^T)^T
    We keep base weight frozen outside this module.
    """
    def __init__(self, base_linear: nn.Linear, r:int, r_alt:int=0, use_alt:bool=False, alt_style:str="uzv"):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = int(r)
        self.use_ab = self.r > 0
        self.r_alt = int(r_alt) if use_alt else 0
        self.use_alt = use_alt and self.r_alt > 0
        self.alt_style = alt_style

        # AB
        if self.use_ab:
            self.A = nn.Parameter(torch.empty(self.out_features, self.r))
            self.B = nn.Parameter(torch.empty(self.r, self.in_features))
            _init_lora(self.A, self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        # alt: U Z V^T (Z is r_alt x r_alt diagonal-like by default for stability)
        if self.use_alt:
            self.U = nn.Parameter(torch.empty(self.out_features, self.r_alt))
            self.V = nn.Parameter(torch.empty(self.in_features, self.r_alt))
            # Z as diagonal vector -> effective rank r_alt but compact; stable
            self.z = nn.Parameter(torch.zeros(self.r_alt))
            nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        else:
            self.register_parameter("U", None)
            self.register_parameter("V", None)
            self.register_parameter("z", None)

    def forward(self, x):
        y = self.base(x)
        if self.use_ab:
            # x: [B,*,in]
            # delta = x @ (A B)^T = (x @ B^T) @ A^T
            delta = torch.matmul(torch.matmul(x, self.B.t()), self.A.t())
            y = y + delta
        if self.use_alt:
            # delta_alt = x @ (U diag(z) V^T)^T = x @ (V diag(z) U^T)
            # compute xV: [B,*,r_alt]
            xV = torch.matmul(x, self.V)  # V: in x r_alt
            xVz = xV * self.z  # elementwise
            delta_alt = torch.matmul(xVz, self.U.t())
            y = y + delta_alt
        return y

    # ===== helpers for deltaW matrices =====
    def deltaW_ab(self):
        if not self.use_ab:
            return None
        return self.A @ self.B  # out x in

    def deltaW_alt(self):
        if not self.use_alt:
            return None
        # U diag(z) V^T
        return self.U @ torch.diag(self.z) @ self.V.t()

    def set_grads_from_G_ab(self, G):
        """Given target gradient in deltaW-space for AB (out x in), set grads for A,B."""
        if not self.use_ab:
            return
        # dL/dA = G @ B^T ; dL/dB = A^T @ G
        dA = G @ self.B.t()
        dB = self.A.t() @ G
        if self.A.grad is None: self.A.grad = torch.zeros_like(self.A)
        if self.B.grad is None: self.B.grad = torch.zeros_like(self.B)
        self.A.grad.copy_(dA)
        self.B.grad.copy_(dB)

    def set_grads_from_G_alt(self, G):
        """Given target gradient in deltaW-space for alt (out x in), set grads for U,V,z."""
        if not self.use_alt:
            return
        # deltaW = U diag(z) V^T
        # dU = G @ V diag(z)
        # dV = G^T @ U diag(z)
        # dz = diag(U^T G V)
        Z = torch.diag(self.z)
        dU = G @ (self.V @ Z)
        dV = G.t() @ (self.U @ Z)
        dz_full = self.U.t() @ G @ self.V  # r_alt x r_alt
        dz = torch.diag(dz_full)
        if self.U.grad is None: self.U.grad = torch.zeros_like(self.U)
        if self.V.grad is None: self.V.grad = torch.zeros_like(self.V)
        if self.z.grad is None: self.z.grad = torch.zeros_like(self.z)
        self.U.grad.copy_(dU)
        self.V.grad.copy_(dV)
        self.z.grad.copy_(dz)

    def absorb_alt_into_ab(self, r:int, alpha:float=1.0):
        """Route A: absorb compressible part of alt into AB using rank-r projection."""
        if (not self.use_alt) or (not self.use_ab):
            return
        with torch.no_grad():
            alt = self.deltaW_alt()
            # low-rank approx of alt with rank r
            U, S, Vh = torch.linalg.svd(alt, full_matrices=False)
            k = min(r, S.numel())
            Uk = U[:, :k]
            Sk = S[:k]
            Vk = Vh[:k, :]
            proj = (Uk * Sk) @ Vk  # out x in
            # add to AB: A B <- A B + alpha*proj, then re-factorize back to rank r
            new = self.deltaW_ab() + alpha * proj
            U2, S2, Vh2 = torch.linalg.svd(new, full_matrices=False)
            k2 = min(self.r, S2.numel())
            Uk2 = U2[:, :k2]
            Sk2 = S2[:k2]
            Vk2 = Vh2[:k2, :]
            # set A,B such that A@B = Uk diag(Sk) Vk
            self.A.copy_(Uk2 * Sk2)
            self.B.copy_(Vk2)
            # remove absorbed part from alt by subtracting proj
            resid = alt - alpha * proj
            # re-pack resid back into U,z,V via SVD rank r_alt (diagonal z)
            U3, S3, Vh3 = torch.linalg.svd(resid, full_matrices=False)
            k3 = min(self.r_alt, S3.numel())
            U3k = U3[:, :k3]
            S3k = S3[:k3]
            V3k = Vh3[:k3, :].t()
            self.U.zero_()
            self.V.zero_()
            self.z.zero_()
            self.U[:, :k3].copy_(U3k)
            self.V[:, :k3].copy_(V3k)
            self.z[:k3].copy_(S3k)

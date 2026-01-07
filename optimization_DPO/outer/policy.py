# ===== FILE: outer/policy.py =====
import math
import torch
import torch.nn as nn

# 保留原 PolicySampler（兼容旧脚本；可不用）
class PolicySampler(nn.Module):
    def __init__(self, feat_dim: int = 5, temperature: float = 4.0):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(feat_dim))
        self.temperature = float(temperature)
        self.entropy_coef = 1e-3
        self.kl_coef = 2e-2
        self.clip_eps = 0.1
        self.opt = torch.optim.Adam([self.w], lr=1e-3)

    @torch.no_grad()
    def probs(self, X: torch.Tensor) -> torch.Tensor:
        s = (X @ self.w).clamp(-20, 20)
        p = torch.softmax(self.temperature * s, dim=0)
        return p

    @torch.no_grad()
    def sample(self, X: torch.Tensor, k: int, sigma0: float = 0.05, kappa: float = 1.0):
        H = X.size(0)
        p = self.probs(X)
        k = int(max(1, min(H, k)))
        idx = torch.multinomial(p, num_samples=k, replacement=False)
        scores = (X @ self.w).detach()
        sig = sigma0 * (1.0 + kappa * torch.sigmoid(scores[idx]).detach())
        return idx.cpu().tolist(), sig.cpu().tolist(), p.detach()

    def update_ppo(self, old_logp: torch.Tensor, new_logp: torch.Tensor, advantage: torch.Tensor,
                   clip_eps: float = 0.2, entropy_coef: float = 0.0):
        if not hasattr(self, "opt"):
            self.opt = torch.optim.Adam(self.parameters(), lr=3e-4)

        self.opt.zero_grad(set_to_none=True)
        adv = advantage
        if isinstance(adv, torch.Tensor):
            adv = adv.detach()
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        else:
            adv = torch.tensor([float(adv)], dtype=torch.float32)

        use_ppo = (isinstance(new_logp, torch.Tensor) and new_logp.numel() > 0 and new_logp.requires_grad)
        if use_ppo and isinstance(old_logp, torch.Tensor) and old_logp.numel() == new_logp.numel():
            dev = next(self.parameters()).device
            old_logp = old_logp.to(dev)
            new_logp = new_logp.to(dev)
            adv = adv.to(dev)

            ratio = torch.exp((new_logp - old_logp).clamp(-20, 20))
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy_loss = torch.tensor(0.0, device=dev)
            if hasattr(self, "_last_entropy") and isinstance(self._last_entropy, torch.Tensor):
                entropy_loss = -entropy_coef * self._last_entropy.mean()

            loss = policy_loss + entropy_loss
            loss.backward()
            self.opt.step()
            return

        # 若 new_logp 无梯度，降级到对某个可训练参数的 surrogate（保持稳定）
        dev = next(self.parameters()).device
        adv = adv.to(dev)

        gamma_param = None
        if hasattr(self, "gamma"):
            gp = getattr(self, "gamma")
            if isinstance(gp, torch.Tensor) and getattr(gp, "requires_grad", False):
                gamma_param = gp
        if gamma_param is None:
            for p in self.parameters():
                if p.requires_grad:
                    gamma_param = p
                    break
        if gamma_param is None:
            return

        g = torch.sigmoid(gamma_param)
        loss = -(adv.mean()) * g.mean()

        l2 = torch.tensor(0.0, device=dev)
        for p in self.parameters():
            if p.requires_grad:
                l2 = l2 + (p.detach() ** 2).sum()
        loss = loss + 1e-8 * l2

        loss.backward()
        self.opt.step()


# ---------------------------
# 概率门策略（硬/软）
# ---------------------------
class ProbGatePolicy(nn.Module):
    """
    每维产生一个扰动概率 p_i，以及（可选）全局强度 gate g_t
    - gate_mode: "soft" (默认, Gumbel-Sigmoid) | "hard" (Bernoulli)
    - 输出:
        m: 0/1 或 (0,1) 的门，形状 [H]
        eps: 高斯幅度 ~ N(0, sigma)，形状 [H]
        logprob: PPO 用的总 logprob（门 + 高斯），标量（带梯度）
        p_detached: p 的副本（无梯度）方便记录
    """
    def __init__(self, feat_dim: int = 8, beta: float = 2.0, gate_mode: str = "soft",
                 learn_global_scale: bool = True, init_gamma: float = 0.0,
                 lr: float = 3e-4, tau: float = 0.5):
        super().__init__()
        self.beta = float(beta)
        self.gate_mode = gate_mode
        self.tau = float(tau)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, max(4, feat_dim)),
            nn.Tanh(),
            nn.Linear(max(4, feat_dim), 1)
        )
        self.learn_global_scale = bool(learn_global_scale)
        if self.learn_global_scale:
            self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))
        else:
            self.register_buffer("gamma", torch.tensor(float(init_gamma)))

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.clip_eps = 0.2

    def _scores_to_probs(self, s: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(self.beta * s.squeeze(-1))            # [H]
        g = torch.sigmoid(self.gamma).expand_as(p)               # 全局
        return (g * p).clamp(1e-6, 1 - 1e-6)

    def forward(self, X: torch.Tensor, sigma: float = 0.1):
        """
        X: [H, feat_dim]
        return: m, eps, logprob, p_detached
        """
        s = self.head(X)                         # [H,1]
        p = self._scores_to_probs(s)             # [H]
        eps = torch.randn_like(p) * float(sigma) # [H]

        if self.gate_mode == "hard":
            # Bernoulli
            m = (torch.rand_like(p) < p).float()                   # [H]
            logpm_vec = (m * torch.log(p) + (1 - m) * torch.log(1 - p))
            logpm = logpm_vec.sum()
        else:
            # soft: Gumbel-Sigmoid
            gumbel = -torch.log(-torch.log(torch.rand_like(p)))
            logits = torch.log(p) - torch.log(1 - p) + gumbel
            m = torch.sigmoid(logits / max(1e-6, self.tau))        # [H], (0,1)
            # 近似门的 logprob：对 PPO 足够稳定（m 用 detach 避免梯度路径破）
            logpm_vec = (m.detach() * torch.log(p) + (1 - m.detach()) * torch.log(1 - p))
            logpm = logpm_vec.sum()

        # 高斯幅度的 logprob（独立各维）
        logpe_vec = (-0.5 * (eps / float(sigma)) ** 2 - 0.5 * math.log(2 * math.pi * (sigma ** 2) + 1e-12))
        logpe = logpe_vec.sum()

        total_logp = logpm + logpe
        return m, eps, total_logp, p.detach()

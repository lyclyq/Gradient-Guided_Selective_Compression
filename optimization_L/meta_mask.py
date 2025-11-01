import torch

class BucketMetaMask:
    """
    K 桶掩码控制器：
    - 先把 per-sample 评分 S 标准化为 0 均值、1 方差；
    - 用分位数把 S 划分为 K 个桶；
    - 每个桶有一个可调“斜率” alpha_k，计算 p_i = sigmoid(alpha_{bucket(i)} * S_i)
      -> 权重 w_i = 1 - p_i （p 更像“噪声概率”，w 是保留权重）。
    - 只维护 alpha（低维、可解释），没有单样本参数，稳定高效。
    """
    def __init__(self, K=4, init_alpha=1.0, device="cpu"):
        self.K = int(K)
        self.device = device
        self.alpha = torch.full((self.K,), float(init_alpha), device=device)

    @torch.no_grad()
    def bucketize(self, S):
        """
        按分位数把 S 切成 K 桶，返回每个样本所在的桶 id（0..K-1）以及桶边界。
        """
        K = self.K
        qs = torch.linspace(0, 1, K+1, device=S.device)
        # 避免量化噪声：用 clamp 保证单调
        edges = torch.quantile(S, qs).clamp_min(S.min()).clamp_max(S.max())
        # 最右闭区间
        idx = torch.bucketize(S, edges[1:-1])  # 返回 0..K-1
        return idx, edges

    @torch.no_grad()
    def weights(self, S):
        """
        输入：S（标准化后的评分），shape [B]
        输出：w（样本权重，越大越被保留），以及一些统计量
        """
        # 标准化（稳定 & 使 alpha 可跨 batch 通用）
        Sm = (S - S.mean()) / (S.std(unbiased=False) + 1e-8)
        idx, edges = self.bucketize(Sm)
        alpha = self.alpha.clamp_min(-10.0).clamp_max(10.0)  # 避免过激

        # 逐样本读取所属桶的 alpha
        a = alpha[idx]
        pprob = torch.sigmoid(a * Sm)          # 认为“疑似噪声”的概率
        w = (1.0 - pprob).detach()             # 冻结到 S/alpha（不让梯度穿过掩码参数）

        # 统计
        with torch.no_grad():
            keep_rate = w.mean().item()
            bucket_keep = []
            for k in range(self.K):
                mask = (idx == k)
                if mask.any():
                    bucket_keep.append(float(w[mask].mean().item()))
                else:
                    bucket_keep.append(float('nan'))
        stats = {"keep_rate": keep_rate, "bucket_keep": bucket_keep, "edges": edges.detach().cpu().tolist()}
        return w, stats

    @torch.no_grad()
    def sample_perturbations(self, num=2, eps=0.2, symmetric=True, rng=None):
        """
        生成若干个低维微扰 Δ，作用在 alpha 上。
        - 每次只动一个桶（坐标方向），便于解释与稳态控制；
        - symmetric=True 则正负对称采样（antithetic），降低方差。
        """
        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(0)

        outs = []
        K = self.K
        for i in range(num):
            j = int(torch.randint(0, K, (1,), generator=rng).item())
            sign = 1.0
            if symmetric and (i % 2 == 1):
                sign = -1.0
            delta = torch.zeros(K, device=self.device)
            delta[j] = sign * eps
            outs.append(delta)
        return outs

    @torch.no_grad()
    def apply_delta(self, delta):
        self.alpha.add_(delta)

    @torch.no_grad()
    def clone_params(self):
        return self.alpha.clone()

    @torch.no_grad()
    def set_params(self, alpha_new):
        assert alpha_new.numel() == self.alpha.numel()
        self.alpha.copy_(alpha_new.to(self.alpha.device))

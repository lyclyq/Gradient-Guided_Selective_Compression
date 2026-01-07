import torch

def randomized_low_rank_proj(X: torch.Tensor, k: int, n_oversamples: int = 2, n_iter: int = 1):
    """Randomized SVD projection to rank k: returns proj(X).
    X: [out,in], float32
    """
    out, inn = X.shape
    kk = min(k, min(out, inn))
    if kk <= 0:
        return torch.zeros_like(X)
    device = X.device
    Omega = torch.randn(inn, kk + n_oversamples, device=device, dtype=X.dtype)
    Y = X @ Omega
    for _ in range(n_iter):
        Y = X @ (X.t() @ Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.t() @ X
    # small svd
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    Uk = Q @ Ub[:, :kk]
    Sk = S[:kk]
    Vk = Vh[:kk, :]
    proj = (Uk * Sk) @ Vk
    return proj

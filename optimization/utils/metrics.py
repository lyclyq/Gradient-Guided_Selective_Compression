# utils/metrics.py
import torch, torch.nn.functional as F

@torch.no_grad()
def ece_score(logits, y, n_bins=15):
    probs = F.softmax(logits, -1); conf, pred = probs.max(-1)
    correct = (pred==y).float(); bins = torch.linspace(0,1,n_bins+1, device=logits.device)
    ece = torch.tensor(0., device=logits.device)
    for i in range(n_bins):
        m = (conf>bins[i]) & (conf<=bins[i+1])
        if m.any(): ece += m.float().mean() * (correct[m].mean()-conf[m].mean()).abs()
    return float(ece.cpu())

class EMAMeter:
    def __init__(self, alpha=0.3): self.alpha, self.val = alpha, None
    def update(self, x: float):
        self.val = x if self.val is None else (self.alpha*x + (1-self.alpha)*self.val)
        return self.val

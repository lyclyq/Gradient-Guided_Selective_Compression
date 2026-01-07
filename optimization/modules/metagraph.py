# modules/metagraph.py
import torch, torch.nn as nn, torch.nn.functional as F
from utils.registry import register_meta

def _pairwise_knn(x, k):
    with torch.no_grad():
        dist = torch.cdist(x, x)         # [N,N]
        k = min(k, x.size(0)-1)
        vals, idx = torch.topk(-dist, k=k+1, dim=1)
        src = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand(-1,k+1)
        src = src[:,1:].reshape(-1); tgt = idx[:,1:].reshape(-1)
    return src, tgt

class _BettiSketchLite(nn.Module):
    def __init__(self, in_dim, levels=2, ratios=(0.1,0.05)):
        super().__init__(); self.levels=levels; self.ratios=ratios
        self.proj = nn.ModuleList([nn.Linear(in_dim, max(8,in_dim//(2**i)), bias=False) for i in range(levels)])
        for p in self.proj: nn.init.orthogonal_(p.weight)
    def forward(self, feats):      # feats:[B,C,H,W] or [B,C]
        if feats.dim()==4: feats = feats.mean(dim=(2,3))
        N,C = feats.shape
        agg_b0=agg_b1=0.0
        for i in range(self.levels):
            z = F.normalize(self.proj[i](feats), dim=-1)
            k = max(3, int(self.ratios[i]*N))
            src, tgt = _pairwise_knn(z, k)
            num_edges = src.numel(); num_nodes = N
            A = torch.zeros((N,N), device=feats.device, dtype=torch.bool)
            A[src, tgt] = True; A[tgt, src] = True
            visited = torch.zeros(N, dtype=torch.bool, device=feats.device)
            comps = 0
            for u in range(N):
                if not visited[u]:
                    comps += 1; q=[u]; visited[u]=True
                    while q:
                        v=q.pop()
                        nbrs = torch.nonzero(A[v], as_tuple=False).squeeze(-1)
                        for w in nbrs:
                            if not visited[w]:
                                visited[w]=True; q.append(int(w))
            b0 = comps; b1 = max(0, num_edges - num_nodes + comps)
            agg_b0 += b0; agg_b1 += b1
        return torch.tensor([agg_b0, agg_b1], device=feats.device, dtype=torch.float32)

class _ChangePointLite(nn.Module):
    def __init__(self, win=50, alpha=0.05):
        super().__init__(); from collections import deque
        self.win=win; self.alpha=alpha; self.buf=deque(maxlen=win); self.mu=None; self.var=None
    def forward(self, feats):
        if feats.dim()==4: feats=feats.mean(dim=(2,3))
        bmean = feats.mean(0).detach()
        if self.mu is None: self.mu=bmean; self.var=torch.ones_like(bmean)
        else:
            self.mu = 0.9*self.mu + 0.1*bmean
            self.var= 0.9*self.var+ 0.1*(bmean-self.mu).pow(2)
        stat = torch.norm(bmean-self.mu)/(self.var.mean().sqrt()+1e-6)
        self.buf.append(float(stat))
        if len(self.buf)<self.win//2: return {"change":False,"band":-1}
        import torch as T
        arr = T.tensor(list(self.buf))
        thr = arr.mean() + 1.0*arr.std()
        return {"change": bool(arr[-int(self.win*0.2):].gt(thr).float().mean()>self.alpha), "band":0}

class _CouplingLite(nn.Module):
    def __init__(self, dim, strength=0.5):
        super().__init__(); self.gate = nn.Parameter(torch.zeros(dim)); self.strength=strength
    def forward(self, clean, adv):
        if adv is None: return {"defended":clean}
        if clean.dim()==4: clean=clean.mean(dim=(2,3))
        if adv.dim()==4:   adv  =adv.mean(dim=(2,3))
        g = torch.sigmoid(self.gate).unsqueeze(0)
        return {"defended": clean + g*self.strength*(adv-clean)}

def _fgsm(x, y, model, eps=0.1):
    x = x.clone().detach().requires_grad_(True)
    logits = model(x); loss = F.cross_entropy(logits,y); model.zero_grad(); loss.backward()
    return (x + eps * x.grad.sign()).clamp(0,1).detach()

@register_meta("metagraph")
class MetaGraph:
    """
    单文件解耦：把 A(拓扑草图)、B(变更点)、D(耦合)、C(简易课程) 集成。
    对外接口：
      - maybe_adv(model, x, y) -> adv_x or None
      - forward_loss(model, x, y, adv_x=None) -> (loss, logits, extras)
      - on_batch_end(model, x, y, logits, loss) -> 更新旋钮（ls/rdrop/dropout/平滑核）
    """
    def __init__(self, use_betti=True, use_cp=True, use_coupling=True, use_curr=True,
                 adv_eps=0.1):
        self.use_betti=use_betti; self.use_cp=use_cp; self.use_coupling=use_coupling; self.use_curr=use_curr
        self._ls=0.0; self._rdrop=0.0; self.adv_eps=adv_eps
        self._betti=None; self._cp=None; self._coup=None

    def _ensure_modules(self, model, x):
        # 依据特征通道维度初始化内部模块
        with torch.no_grad():
            feats = model.features(x[:1])
        C = feats.size(1) if feats.dim()==4 else feats.size(-1)
        device = feats.device
        if self.use_betti   and self._betti is None: self._betti = _BettiSketchLite(in_dim=C).to(device)
        if self.use_cp      and self._cp    is None: self._cp    = _ChangePointLite().to(device)
        if self.use_coupling and self._coup is None: self._coup  = _CouplingLite(dim=C).to(device)

    def maybe_adv(self, model, x, y):
        return _fgsm(x, y, model, eps=self.adv_eps) if self.use_coupling else None

    def forward_loss(self, model, x, y, adv_x=None):
        self._ensure_modules(model, x)
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self._ls)
        # R-Drop
        if self._rdrop>0.0 and model.training:
            logits2 = model(x)
            ce = 0.5*(F.cross_entropy(logits,y,label_smoothing=self._ls) +
                      F.cross_entropy(logits2,y,label_smoothing=self._ls))
            kl = 0.5*(F.kl_div(F.log_softmax(logits,-1),  F.softmax(logits2,-1), reduction="batchmean")+
                      F.kl_div(F.log_softmax(logits2,-1), F.softmax(logits,-1),  reduction="batchmean"))
            loss = ce + self._rdrop*kl
            logits = 0.5*(logits+logits2)

        with torch.no_grad():
            feats = model.features(x)
            betti = self._betti(feats) if self.use_betti else None
            _     = self._cp(feats)    if self.use_cp    else None
            _     = self._coup(feats, model.features(adv_x)) if (self.use_coupling and adv_x is not None) else None

        return loss, logits, {"betti": betti}

    def on_batch_end(self, model, x, y, logits, loss):
        # 简易 curriculum：高精度/低损失时增大 LS/R-Drop/Dropout
        acc = (logits.argmax(-1)==y).float().mean().item()
        if not self.use_curr: return
        if acc>0.98:      self._ls  = min(0.2, self._ls+0.02)
        if loss.item()<0.1: self._rdrop = min(0.5, self._rdrop+0.02)
        if hasattr(model,'set_dropout'):
            model.set_dropout(min(0.5, getattr(model,'drop').p + (0.02 if acc>0.98 else 0.0)))

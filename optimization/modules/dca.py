# modules/dca.py
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DCACfg:
    enable: bool = True
    T: float = 2.0
    ema_beta: float = 0.95
    lambda_js: float = 0.2
    lambda_drift: float = 0.2
    beta_token_kl: float = 0.05
    entropy_w: float = 1e-3
    max_taps: int = 64
    verbose: bool = True

    # 两种贡献权重（a: 层漂移，b: 全局分布差对层能量的 proxy）
    w_local: float = 0.5
    w_global: float = 0.5

    # E-CTR 的 β 耦合强度
    beta_couple: float = 1.0


class DCA:
    """
    Distributional Consistency Attribution:
    - 训练：更新 per-class train 分布 + 每层 token 能量/轮廓的 EMA
    - 验证：基于 CE(val) + λ_js * JS(label-cond) + λ_drift * layer_drift (+ 熵正则) 更新 α
    - 另外返回每层引导分数（0..1），可用于外部微调 Soft-ResMix 的 beta
    """
    def __init__(self, cfg: DCACfg, num_classes: int):
        self.cfg = cfg
        self.C = num_classes

        # 训练侧 per-class 概率分布（length=C 的向量）EMA
        self.p_train_ema: List[Optional[torch.Tensor]] = [None for _ in range(self.C)]
        # 层统计 EMA：标量能量 mu、token 轮廓 r[T]
        self.ema_mu: Dict[int, torch.Tensor] = {}
        self.ema_r: Dict[int, torch.Tensor] = {}

        # 验证缓存：[(layer_idx, mu_val, r_val), ...]
        self.val_cache: List[Tuple[int, torch.Tensor, torch.Tensor]] = []

        # hook 保存
        self._hooks: List[Any] = []
        self._taps_idx: Dict[nn.Module, int] = {}

    # ---------- hooks ----------
    def attach_hooks(self, model: nn.Module):
        taps: List[nn.Module] = []
        # 优先选标记了 dca_tap 的模块
        for m in model.modules():
            if getattr(m, "dca_tap", False):
                taps.append(m)
        # 退化策略：按类名包含关键词挑选
        if len(taps) == 0:
            KEY = ("Block", "Attn", "Attention", "Transformer", "Encoder", "FFN", "Mlp", "Conv", "Gate")
            for m in model.modules():
                cn = m.__class__.__name__
                if any(k in cn for k in KEY):
                    taps.append(m)

        taps = taps[: self.cfg.max_taps]
        self._hooks = []
        self._taps_idx.clear()

        def _pick_tensor_from_output(out: Any) -> Optional[torch.Tensor]:
            """兼容 HF：有的层返回 tuple/list，把里面第一个 Tensor 拿出来。"""
            if isinstance(out, torch.Tensor):
                return out
            if isinstance(out, (tuple, list)):
                for o in out:
                    if isinstance(o, torch.Tensor):
                        return o
            return None

        for idx, m in enumerate(taps):
            self._taps_idx[m] = idx

            def _make_hook(layer_idx: int):
                def _hook(_m, _inp, out):
                    ten = _pick_tensor_from_output(out)
                    if ten is None:
                        return

                    # 统一到 [B, T, C] token 维度在中间
                    if ten.dim() == 4:
                        # [B,C,H,W] -> [B,HW,C]
                        B, C, H, W = ten.shape
                        ten = ten.view(B, C, H * W).transpose(1, 2).contiguous()
                    elif ten.dim() == 3:
                        # [B,T,C]，不处理
                        pass
                    else:
                        return

                    # token 能量（L2 范数）与归一化轮廓
                    m_token = ten.pow(2).sum(dim=-1).sqrt()                   # [B,T]
                    mu = m_token.mean()                                       # 标量
                    r = (m_token / (m_token.sum(dim=1, keepdim=True) + 1e-8)).mean(dim=0)  # [T] 平均轮廓

                    if _m.training:
                        # 训练端只更新 EMA（不参与梯度）
                        with torch.no_grad():
                            b = self.cfg.ema_beta
                            self.ema_mu[layer_idx] = mu if layer_idx not in self.ema_mu else b * self.ema_mu[layer_idx] + (1 - b) * mu
                            # self.ema_r[layer_idx] = r if layer_idx not in self.ema_r else b * self.ema_r[layer_idx] + (1 - b) * r
                            cur = r
                            if layer_idx in self.ema_r:
                                prev = self.ema_r[layer_idx]
                                if prev.shape == cur.shape:
                                    upd = b * prev + (1 - b) * cur
                                else:
                                    # 形状不一致，做一个保守 reset（也可以写成对齐裁剪/零填充）
                                    upd = cur
                            else:
                                upd = cur
                            self.ema_r[layer_idx] = upd
    
                    
                    else:
                        # 验证端给 α 反传：把 mu/r 原样缓存（不要 detach）
                        self.val_cache.append((layer_idx, mu, r))

                return _hook

            h = m.register_forward_hook(_make_hook(idx))
            self._hooks.append(h)

        if self.cfg.verbose:
            print(f"[DCA] hooks attached: {len(self._hooks)} taps")

    def clear_val_cache(self):
        self.val_cache.clear()

    def detach_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        self._taps_idx.clear()

    # ---------- train: 更新训练分布 EMA ----------
    @torch.no_grad()
    def update_train_distribution_ema(self, logits: torch.Tensor, y: torch.Tensor):
        T = self.cfg.T
        prob = torch.softmax(logits / T, dim=-1)  # [B,C]
        b = self.cfg.ema_beta
        for c in range(self.C):
            mask = (y == c)
            if mask.any():
                p_c = prob[mask].mean(dim=0)  # [C]
                prev = self.p_train_ema[c]
                self.p_train_ema[c] = p_c if prev is None else b * prev + (1 - b) * p_c

    # ---------- helper：计算层漂移 ----------
    def _layer_drift(self, device) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """
        返回 (mean_drift_scalar, layer_idxs, drift_vec)
        drift = |mu_val - mu_tr| + beta_token_kl * JS(r_val || r_tr)
        """
        drifts: List[torch.Tensor] = []
        idxs: List[int] = []
        eps = 1e-8

        for (layer_idx, mu_val, r_val) in self.val_cache:
            mu_tr = self.ema_mu.get(layer_idx, None)
            r_tr = self.ema_r.get(layer_idx, None)
            if (mu_tr is None) or (r_tr is None):
                continue

            # 标量能量 L1 偏差
            d_mu = (mu_val - mu_tr).abs()

            # token 轮廓 JS（两向 KL 的一半和）
            Tlen = min(r_val.numel(), r_tr.numel())
            rv = r_val[:Tlen]
            rt = r_tr[:Tlen]
            m = 0.5 * (rv + rt)
            js = 0.5 * (
                F.kl_div((rv + eps).log(), m, reduction="sum") +
                F.kl_div((rt + eps).log(), m, reduction="sum")
            )

            drift = d_mu + self.cfg.beta_token_kl * js
            drifts.append(drift)
            idxs.append(layer_idx)

        if len(drifts) == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, [], zero

        vec = torch.stack([d.to(device) for d in drifts])  # [L]
        return vec.mean(), idxs, vec

    # ---------- 验证：标准 DCA 架构损失 ----------
    def compute_arch_loss(
        self,
        val_logits: torch.Tensor,
        yv: torch.Tensor,
        class_prior: Optional[torch.Tensor] = None,
        controller_kernel_weights: Optional[List[torch.Tensor]] = None,
    ):
        device = val_logits.device

        # (1) CE(val)
        ce_val = F.cross_entropy(val_logits, yv)

        # (2) label-conditional JS
        T = self.cfg.T
        eps = 1e-8
        qv = torch.softmax(val_logits / T, dim=-1)  # [B,C]
        C = self.C
        if class_prior is None:
            with torch.no_grad():
                cnt = torch.bincount(yv, minlength=C).float()
                class_prior = cnt / (cnt.sum() + 1e-8)

        js_sum, pi_sum = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        for c in range(C):
            mask = (yv == c)
            if mask.any() and (self.p_train_ema[c] is not None):
                q_c = qv[mask].mean(dim=0)  # [C]
                p_c = self.p_train_ema[c].to(device)
                m = 0.5 * (p_c + q_c)
                js_c = 0.5 * (
                    F.kl_div((p_c + eps).log(), m, reduction="sum") +
                    F.kl_div((q_c + eps).log(), m, reduction="sum")
                )
                js_sum = js_sum + class_prior[c] * js_c
                pi_sum = pi_sum + class_prior[c]

        L_js = (js_sum / (pi_sum + eps)) if float(pi_sum.item()) > 0 else torch.tensor(0.0, device=device)

        # (3) 层漂移
        drift_mean, layer_idxs, drift_vec = self._layer_drift(device)
        L_drift = drift_mean

        # (4) 控制器核的熵正则（可选）
        L_ent = torch.tensor(0.0, device=device)
        if controller_kernel_weights:
            for w in controller_kernel_weights:
                if w is not None and w.numel() > 0:
                    p = torch.softmax(w, dim=-1)
                    L_ent = L_ent - (p * (p + eps).log()).sum()

        # 组合
        L = ce_val + self.cfg.lambda_js * L_js + self.cfg.lambda_drift * L_drift + self.cfg.entropy_w * L_ent

        # (5) 给 ResMix 的引导分数（0..1）
        resmix_scores = None
        if len(layer_idxs) > 0:
            local = drift_vec
            global_proxy = drift_vec  # 简单近似：先用 drift_vec 自身
            score = self.cfg.w_local * local + self.cfg.w_global * global_proxy
            score = score - score.min()
            denom = (score.max() - score.min() + 1e-8)
            score = (score / denom).clamp(0, 1)  # [L]
            resmix_scores = (layer_idxs, score.detach())

        return L, {"L_ce": ce_val.detach(),
                   "L_js": L_js.detach(),
                   "L_drift": L_drift.detach(),
                   "resmix_scores": resmix_scores}

    # ---------- 层本地能量相对改变量：a_l ----------
    def _compute_layer_energy_rel(self) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """
        读取 self.val_cache 的 (layer_idx, mu_val, r_val) 与 EMA(mu_tr)，计算
        a_l = |mu_val - mu_tr| / (|mu_tr| + eps)，返回 (a_mean, layer_idxs, a_vec)
        """
        if len(self.ema_mu) == 0:
            device = "cpu"
        else:
            device = next(iter(self.ema_mu.values())).device
        eps = 1e-8
        a_list: List[torch.Tensor] = []
        idxs: List[int] = []
        for (layer_idx, mu_val, _r_val) in self.val_cache:
            mu_tr = self.ema_mu.get(layer_idx, None)
            if mu_tr is None:
                continue
            a = (mu_val - mu_tr).abs() / (mu_tr.abs() + eps)
            a_list.append(a.to(device))
            idxs.append(layer_idx)
        if len(a_list) == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, [], zero
        avec = torch.stack(a_list)  # [L]
        return avec.mean(), idxs, avec

    # ---------- E-CTR ----------
    def compute_e_ctr_loss(self, val_logits: torch.Tensor, yv: torch.Tensor, return_scores: bool = False):
        """
        L_ECTR = sum_l g_l * (β * D * a_l)
        - D：类先验加权的 JS(p_tr[c] || q_val[c])
        - a_l：层能量相对改变量
        - g_l：敏感度门控；当前默认 1
        """
        device = val_logits.device
        T = self.cfg.T
        beta = getattr(self.cfg, "beta_couple", 1.0)
        eps = 1e-8
        C = self.C

        # (1) val 端类条件平均概率
        prob_val = torch.softmax(val_logits / T, dim=-1)  # [B,C]
        pi = torch.zeros(C, device=device)
        q_val = torch.zeros(C, C, device=device)  # [C,C]
        for c in range(C):
            mask = (yv == c)
            if mask.any():
                q_val[c] = prob_val[mask].mean(dim=0)
                pi[c] = mask.float().mean()

        # (2) D：加权 JS
        def _js(p, q):
            m = 0.5 * (p + q)
            return 0.5 * (
                F.kl_div((p + eps).log(), m, reduction="sum") +
                F.kl_div((q + eps).log(), m, reduction="sum")
            )

        D = torch.tensor(0.0, device=device)
        for c in range(C):
            p_tr = self.p_train_ema[c]
            if p_tr is None:
                continue
            D = D + pi[c] * _js(p_tr.to(device), q_val[c])

        # (3) a_l
        a_mean, layer_idxs, a_vec = self._compute_layer_energy_rel()

        if a_vec.numel() > 0:
            # 默认 g_l=1；D 只做“权重”，不让它反向到 logits（保持 E-CTR 纯控制 α）
            term = beta * D.detach() * a_vec
            L = term.sum()

            # s_l ∈ [0,1] 给 ResMix
            s = a_vec - a_vec.min()
            denom = (a_vec.max() - a_vec.min() + eps)
            s = (s / denom).clamp(0, 1)
            scores = (layer_idxs, s.detach())
        else:
            # 无统计时，保持计算图存在
            L = val_logits.sum() * 0.0
            scores = None

        return (L, scores) if return_scores else (L, None)


    def parameters(self, recurse: bool = True):
        """让它“看起来像个 nn.Module”，即使没有可训练张量也返回空迭代器。"""
        return iter(())

    def arch_parameters(self):
        """若没有显式的 alpha/arch 参数，返回空列表即可。"""
        return []

    def controller_kernel_weights(self):
        """若没有控制器核（例如频域滤波器权重），返回空列表即可。"""
        return []
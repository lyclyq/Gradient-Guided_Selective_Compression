# modules/gate_unfreezer.py
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple
import torch
import torch.nn as nn

@dataclass
class GateUnfreezeCfg:
    enable: bool = False
    mode: str = "epoch"        # "epoch" | "step" | "metric"
    every: int = 1             # 间隔（epoch/step）
    min_layers: int = 1        # 初始解冻的“最靠输出”的 gate 数量
    lr_scale: float = 1.0      # 新解冻 gate 的 lr 倍率（用于 opt_a）
    metric_name: str = "val_loss"
    metric_patience: int = 2
    verbose: bool = True

class GateLayerExtractor:
    """
    只收集 DARTS gate（例如 FilterGateDARTS1D/2D）的 alpha 参数：
      - 期望模块上有属性 pre_attn_gate.alpha_logits（small_transformer / small_bert 已满足）
      - 返回顺序：从输入->输出；外部会反转，以“从输出端开始解冻”
    """
    def __init__(self, model: nn.Module, path_resolver: Optional[Callable[[nn.Module], List[Tuple[nn.Module, nn.Parameter]]]] = None):
        self.model = model
        self.layers = self._default_collect(model) if path_resolver is None else path_resolver(model)

    def _default_collect(self, model: nn.Module) -> List[Tuple[nn.Module, nn.Parameter]]:
        pairs: List[Tuple[nn.Module, nn.Parameter]] = []
        for m in model.modules():
            if hasattr(m, "pre_attn_gate"):
                g = getattr(m, "pre_attn_gate")
                if hasattr(g, "alpha_logits") and isinstance(g.alpha_logits, torch.nn.Parameter):
                    pairs.append((g, g.alpha_logits))
        return pairs  # 约定顺序：网络定义顺序（输入->输出）

class GateUnfreezer:
    """
    只逐层解冻 DARTS 架构参数 α（不动普通权重θ）。
    - 需要传入“arch 优化器”（opt_a），它将按层逐步加入 gate.alpha_logits。
    - 一开始将所有 gate.alpha_logits 的 requires_grad=False 并从 opt_a 移除；
      然后从输出端起，逐层加入到 opt_a 的 param_groups。
    """
    def __init__(self, model: nn.Module, arch_optimizer: torch.optim.Optimizer, cfg: GateUnfreezeCfg,
                 path_resolver=None, base_lr=None):
        self.cfg = cfg
        self.opt_a = arch_optimizer
        self.extractor = GateLayerExtractor(model, path_resolver)
        # 反转为“从输出端开始”的顺序
        self.layers: List[Tuple[nn.Module, nn.Parameter]] = list(self.extractor.layers)[::-1]
        self.total = len(self.layers)
        self.unfrozen = 0
        self.base_lr = base_lr or self.opt_a.defaults.get('lr', 1e-3)
        self._metric_hist: List[float] = []

        # 冻住全部 gate α，并从 opt_a 清理
        self._freeze_all()
        # 最少解冻 min_layers 个
        self._ensure_unfrozen(self.cfg.min_layers)

    def _freeze_all(self):
        # 1) 先把所有 α 的 requires_grad 关掉
        for _, alpha in self.layers:
            alpha.requires_grad_(False)
        # 2) 再把 opt_a 里已有α移除（重建 param_groups）
        keep_groups = []
        for g in self.opt_a.param_groups:
            new_params = [p for p in g['params'] if p.requires_grad]  # 只留仍可训练的
            if new_params:
                ng = dict(g)  # 浅拷贝
                ng['params'] = new_params
                keep_groups.append(ng)
        # 用 keep_groups 替换原 groups
        self.opt_a.param_groups.clear()
        for g in keep_groups:
            self.opt_a.add_param_group(g)

    def _add_gate_to_opt(self, alpha: nn.Parameter, group_lr: float):
        # 防重复：跳过已在优化器里的参数
        existing = set()
        for g in self.opt_a.param_groups:
            for p in g['params']: existing.add(p)
        if alpha in existing:  # 已经在 opt_a 中，不重复加
            return
        # 独立 lr 的新组
        defaults = self.opt_a.defaults.copy()
        defaults['lr'] = group_lr
        self.opt_a.add_param_group({**defaults, 'params': [alpha]})

    def _ensure_unfrozen(self, k: int):
        k = min(k, self.total)
        while self.unfrozen < k:
            gate, alpha = self.layers[self.unfrozen]  # 最靠输出的优先
            alpha.requires_grad_(True)
            lr = self.base_lr * self.cfg.lr_scale
            self._add_gate_to_opt(alpha, lr)
            if self.cfg.verbose:
                print(f"[GateUnfreezer] unfreeze gate#{self.unfrozen} (lr={lr:.2e})")
            self.unfrozen += 1

    # ==== 触发接口 ====
    def on_epoch_end(self, epoch: int):
        if not self.cfg.enable or self.cfg.mode != "epoch": return
        if (epoch + 1) % max(1, self.cfg.every) == 0:
            self._ensure_unfrozen(self.unfrozen + 1)

    def on_step(self, step: int):
        if not self.cfg.enable or self.cfg.mode != "step": return
        if (step + 1) % max(1, self.cfg.every) == 0:
            self._ensure_unfrozen(self.unfrozen + 1)

    def on_metric(self, value: float):
        if not self.cfg.enable or self.cfg.mode != "metric": return
        self._metric_hist.append(float(value))
        if len(self._metric_hist) < self.cfg.metric_patience + 1: return
        recent = self._metric_hist[-(self.cfg.metric_patience + 1):]
        # 最近一次不优于前面最好 => 解冻一层
        if recent[-1] > min(recent[:-1]) - 1e-6:
            self._ensure_unfrozen(self.unfrozen + 1)

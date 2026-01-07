# ===== FILE: train_outerloop.py =====
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()  # 关掉 transformers 的大部分日志

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # 避免 tokenizers 并行 fork 警告

import argparse
import warnings
from collections import deque
from typing import Iterable, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 屏蔽 fast tokenizer 的刷屏提示
warnings.filterwarnings(
    "ignore",
    message=".*fast tokenizer.*using the `__call__` method is faster.*",
    category=UserWarning,
)

from utils.logging_utils import CSVLogger, SwanLogger

# --- dataset shim: 优先用项目内的 get_sst2；否则回退到 HF datasets ---
try:
    from utils.data_utils import get_sst2  # 项目内实现
except Exception:
    def get_sst2(batch_size: int = 64, model_name: str = "distilbert-base-uncased"):
        from datasets import load_dataset
        from transformers import AutoTokenizer, DataCollatorWithPadding

        ds = load_dataset("glue", "sst2")
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        def preprocess(examples):
            return tok(examples["sentence"], truncation=True)
        cols = ds["train"].column_names
        ds = ds.map(preprocess, batched=True, remove_columns=[c for c in cols if c not in ["label"]])
        ds = ds.rename_columns({"label": "labels"})
        ds.set_format(type="torch", columns=[c for c in ds["train"].column_names if c in
                                             ["input_ids", "attention_mask", "token_type_ids", "labels"]])

        # 官方 validation 拆成 val1/val2
        val = ds["validation"]
        idx_even = list(range(0, len(val), 2))
        idx_odd = list(range(1, len(val), 2))
        val1 = val.select(idx_even)
        val2 = val.select(idx_odd)
        test = val2  # GLUE test 无 label，直接用 val2 代替

        collate = DataCollatorWithPadding(tokenizer=tok, return_tensors="pt", pad_to_multiple_of=None)
        pin = torch.cuda.is_available()

        def make_loader(split, shuffle):
            return DataLoader(split, batch_size=batch_size, shuffle=shuffle,
                              collate_fn=collate, pin_memory=pin, num_workers=0)

        tr_ld = make_loader(ds["train"], shuffle=True)
        va1_ld = make_loader(val1, shuffle=False)
        va2_ld = make_loader(val2, shuffle=False)
        te_ld = make_loader(test, shuffle=False)
        return tr_ld, va1_ld, va2_ld, te_ld, tok

from models.transformer_text import TransformerTextClassifier
from models.distillbert_hf import DistillBertClassifier
from models.bert_hf import BertClassifier

from outer.adaptors import QMaskAdaptor, DoubleDiagAdaptor, GradSuppressor
from outer.policy import ProbGatePolicy
from outer.rollback import RollbackController

from outer.feature_stats import FeatureTracker
from outer.val2_metrics import (
    robust_eval_val2,
    attn_distribution_distance,
    grad_consistency,
)
from outer.ppo_buffer import PPOBuffer

# --- optional debug helpers from utils.debug_utils ---
try:
    from utils.debug_utils import (
        # 若你的 debug_utils 里名字不同，可在这里替换导入；下方均做了兜底
        fprint_tensor as tensor_fp,   # 打印张量指纹
        pairwise_cosines,             # 候选相似度矩阵
        align_check,                  # alpha vs attn T（仅供参考）
        max_abs_diff,                 # max |Δ|
        # 以下是训练脚本内部用到的可选接口（没有也行）
        # model 模式、attn RAW/QUANT 距离、门控轨迹、Δlogits 统计、平滑噪声
        model_mode_fp, attn_metrics, gate_trace, logits_delta_stats, vector_noise_like
    )
except Exception:
    def tensor_fp(*args, **kwargs): pass
    def pairwise_cosines(*args, **kwargs): pass
    def align_check(*args, **kwargs): pass
    def max_abs_diff(a, b): 
        try: return float((a.detach().float() - b.detach().float()).abs().max().item())
        except: return float('nan')
    def model_mode_fp(*args, **kwargs): pass
    def attn_metrics(*args, **kwargs): pass
    def gate_trace(*args, **kwargs): pass
    def logits_delta_stats(*args, **kwargs): pass
    def vector_noise_like(a0, kind="smooth", bw=2):
        z = torch.randn_like(a0)
        if kind == "smooth" and bw > 0 and a0.dim() == 1 and a0.numel() > 4 * bw:
            k = 2 * bw + 1
            pad = (bw, bw)
            z2 = F.pad(z.view(1, 1, -1), pad, mode="reflect")
            ker = torch.ones(1, 1, k, device=z.device, dtype=z.dtype) / k
            z = F.conv1d(z2, ker).view(-1)
        return z


# ---------- helpers ----------
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def _split_text_batch(batch: dict, device: torch.device):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attn = batch.get("attention_mask", None)
    if attn is None:
        attn = torch.ones_like(input_ids, device=device)
    else:
        attn = attn.to(device, non_blocking=True)
    labels = batch.get("labels", batch.get("label")).to(device, non_blocking=True)
    return input_ids, attn, labels


@torch.no_grad()
def eval_loader(model, loader: DataLoader, device: torch.device):
    # 评估时暂停探针，避免显存涨
    was_probe = getattr(model, "probe_runtime_enable", None)
    if was_probe is not None:
        model.set_probe(False)

    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for batch in loader:
        vi, am, yl = _split_text_batch(batch, device)
        logits = model(input_ids=vi, attention_mask=am)
        loss = F.cross_entropy(logits, yl)
        acc = accuracy_from_logits(logits, yl)
        b = yl.size(0)
        tot_loss += float(loss.item()) * b
        tot_acc += float(acc) * b
        n += b

    if was_probe is not None:
        model.set_probe(bool(was_probe))
        model.clear_attn_buffer()

    return tot_loss / max(1, n), tot_acc / max(1, n)


def build_model(name, tok=None, num_labels=2, hf_override=None, probe_args=None, layer_index: int = -1):
    name = name.lower()
    probe_args = probe_args or {}
    if name == "transformer":
        assert tok is not None
        return TransformerTextClassifier(
            vocab_size=tok.vocab_size,
            num_labels=num_labels,
            pad_id=tok.pad_token_id or 0,
            probe_layer_index=layer_index,
            attn_probe_enable=probe_args.get("attn_probe_enable", False),
            attn_probe_max_T=probe_args.get("attn_probe_max_T", 256),
            attn_probe_pool=probe_args.get("attn_probe_pool", "avg"),
            attn_buf_cap=probe_args.get("attn_buf_cap", 8),
            attn_quant_bits=probe_args.get("attn_quant_bits", 8),
        )
    if name == "distillbert":
        return DistillBertClassifier(
            num_labels=num_labels,
            model_name=hf_override or "distilbert-base-uncased",
            **probe_args,
        )
    if name == "bert":
        return BertClassifier(
            num_labels=num_labels,
            model_name=hf_override or "bert-base-uncased",
            **probe_args,
        )
    raise ValueError(f"Unknown model {name}")


# ---------- 收集注意力权重相关梯度（用于 V2-GRAD） ----------
ATTN_INCLUDE_PATTERNS = (
    "self_attn", ".attn", "multiheadattention",
    ".encoder.layer.", ".encoder.layers.", ".transformer.layers.",
)
SUFFIX_INCLUDE_PATTERNS = (
    "in_proj_weight", "q_proj.weight", "k_proj.weight", "v_proj.weight",
    "in_proj_bias", "out_proj.weight", "out_proj.bias",
)

def _name_match(n: str,
                include_heads: Tuple[str, ...] = ATTN_INCLUDE_PATTERNS,
                include_suffix: Tuple[str, ...] = SUFFIX_INCLUDE_PATTERNS) -> bool:
    n_low = n.lower()
    if not any(p in n_low for p in include_heads):
        return False
    if not any(n_low.endswith(suf) for suf in include_suffix):
        return False
    return True


@torch.no_grad()
def _flat_grad_collect(model: nn.Module) -> torch.Tensor:
    device = next((p.device for p in model.parameters() if p is not None), torch.device("cpu"))
    chunks: List[torch.Tensor] = []
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if not _name_match(n):
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            continue
        chunks.append(g.float().reshape(-1))

    if not chunks:
        return torch.randn(16, device=device) * 1e-8
    return torch.cat(chunks, dim=0)


def _zero_grad(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def _loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, loss_type: str = "ce") -> torch.Tensor:
    if loss_type == "ce":
        return F.cross_entropy(logits, labels)
    raise ValueError(f"Unsupported loss_type={loss_type}")


def _flat_grad_once(model: nn.Module,
                    batch: dict,
                    device: torch.device,
                    forward_kwargs: Optional[dict] = None,
                    loss_type: str = "ce") -> torch.Tensor:
    if forward_kwargs is None:
        forward_kwargs = {}
    model.train(False)
    _zero_grad(model)

    vi, am, yl = _split_text_batch(batch, device)
    logits = model(input_ids=vi, attention_mask=am, **forward_kwargs)
    loss = _loss_from_logits(logits, yl, loss_type=loss_type)
    loss.backward()

    return _flat_grad_collect(model)


def flat_grad_over_loader(model: nn.Module,
                          loader: Iterable[dict],
                          device: torch.device,
                          steps: int = 1,
                          forward_kwargs: Optional[dict] = None,
                          loss_type: str = "ce") -> torch.Tensor:
    vec = []
    it = iter(loader)
    for _ in range(max(1, int(steps))):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        g = _flat_grad_once(model, batch, device, forward_kwargs=forward_kwargs, loss_type=loss_type)
        vec.append(g)
    if len(vec) == 1:
        return vec[0]
    return torch.stack(vec, dim=0).mean(dim=0)


# ---------- 注意力形状调试 ----------
def _shape(x):
    if x is None: return None
    try:
        return tuple(x.shape)
    except Exception:
        return str(type(x))

def debug_attn_shapes(attn_base, attn_cand, attention_mask):
    print(f"[DEBUG-ATTN] base={_shape(attn_base)} cand={_shape(attn_cand)} mask={_shape(attention_mask)}")


# ---------- 诊断工具：alpha 展平与相对差 ----------
def _flatten_tensors(x):
    if isinstance(x, tuple):
        xs = []
        for t in x:
            if torch.is_tensor(t):
                xs.append(t.detach().float().flatten().cpu())
        if len(xs) == 0:
            return None
        return torch.cat(xs)
    elif torch.is_tensor(x):
        return x.detach().float().flatten().cpu()
    return None

def _alpha_rel_l2(a, b):
    va, vb = _flatten_tensors(a), _flatten_tensors(b)
    if va is None or vb is None:
        return None
    denom = vb.norm().item() + 1e-8
    return (va - vb).norm().item() / denom


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="sst2")
    ap.add_argument("--model", type=str, default="distillbert", choices=["distillbert", "bert", "transformer"])
    ap.add_argument("--hf_model", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)

    # --- outer options ---
    ap.add_argument("--mask_type", type=str, default="double", choices=["single", "double"])
    ap.add_argument("--layer_index", type=int, default=-1)

    # 概率门：软/硬
    ap.add_argument("--gate_mode", type=str, default="soft", choices=["soft", "hard"])
    ap.add_argument("--sigma_H", type=float, default=0.10)
    ap.add_argument("--sigma_Q", type=float, default=0.10)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gamma0", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.5)

    # 触发 & 节奏
    ap.add_argument("--use_train_acc_trigger", type=int, default=1)
    ap.add_argument("--trigger_acc", type=float, default=0.85)
    ap.add_argument("--trigger_window", type=int, default=400)
    ap.add_argument("--cooldown", type=int, default=1200)
    ap.add_argument("--val1_cooldown", type=int, default=None)
    ap.add_argument("--val2_every_rounds", type=int, default=1)

    # 候选、Top-K、Val2 M
    ap.add_argument("--R_candidates", type=int, default=16)
    ap.add_argument("--topK", type=int, default=8)
    ap.add_argument("--val2_topM", type=int, default=3)

    ap.add_argument("--val2_select", type=str, default="greedy", choices=["greedy", "softmerge"])
    ap.add_argument("--softmerge_topx", type=int, default=2)

    # 守门 & 回滚
    ap.add_argument("--kl_guard", type=float, default=0.10)
    ap.add_argument("--rollback_mode", type=str, default="hard", choices=["reject", "hard", "soft"])

    # 对应抑制（W_Q梯度）
    ap.add_argument("--grad_suppress_on", type=int, default=1)
    ap.add_argument("--grad_suppress_rho", type=float, default=0.5)

    # 日志
    ap.add_argument("--log_dir", type=str, default="logs_outer")
    ap.add_argument("--run_name", type=str, default="run_probmask")
    ap.add_argument("--swan_project_name", type=str, default="outerloop")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--debug", action="store_true")

    # === probe args ===
    ap.add_argument("--probe_attn", action="store_true")
    ap.add_argument("--probe_max_T", type=int, default=256)
    ap.add_argument("--probe_pool", type=str, default="avg", choices=["none", "avg"])
    ap.add_argument("--attn_buf_cap", type=int, default=8)
    ap.add_argument("--attn_quant_bits", type=int, default=8)

    # === V2 debug开关 ===
    ap.add_argument("--v2_debug", type=int, default=1)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset.lower() != "sst2":
        raise ValueError("Only sst2 is supported in this runner.")
    tr_ld, va1_ld, va2_ld, te_ld, tok = get_sst2(
        batch_size=args.batch,
        model_name=args.hf_model or "distilbert-base-uncased"
    )

    probe_args = dict(
        attn_probe_enable=args.probe_attn,
        attn_probe_max_T=args.probe_max_T,
        attn_probe_pool=args.probe_pool,
        attn_buf_cap=args.attn_buf_cap,
        attn_quant_bits=args.attn_quant_bits,
    )

    model = build_model(
        args.model,
        tok=tok,
        hf_override=args.hf_model,
        probe_args=probe_args,
        layer_index=args.layer_index,
    )

    model.to(device)

    if args.debug:
        print(f"[BOOT] debug={int(args.debug)} | v2_debug={int(args.v2_debug)} | "
              f"probe_attn={args.probe_attn} | probe_pool={args.probe_pool} | "
              f"probe_max_T={args.probe_max_T} | buf_cap={args.attn_buf_cap} | quant_bits={args.attn_quant_bits}")

    # --- adaptor ---
    adaptor = (QMaskAdaptor if args.mask_type == "single" else DoubleDiagAdaptor)(
        model, layer_index=args.layer_index
    )

    # 对齐检查：aH/aQ 长度 H 与 Wq 形状
    if args.debug:
        try:
            attn = adaptor.attn
            if hasattr(attn, "in_proj_weight") and attn.in_proj_weight is not None:
                Wq = attn.in_proj_weight[: adaptor.H, :]
            else:
                Wq = attn.q_proj.weight  # type: ignore
            print(f"[ALIGN-Wq] H={adaptor.H} | Wq.shape={tuple(Wq.shape)} | out_proj.shape={tuple(attn.out_proj.weight.shape)}")
        except Exception as e:
            print(f"[ALIGN-Wq] check skipped: {e}")

    if args.debug:
        print(f"[ADAPTOR] mask_type={args.mask_type} | layer_index={args.layer_index} | H={adaptor.H}")

    # --- 策略：H/Q 两套 ---
    feat_dim_H = 8
    feat_dim_Q = 8
    policy_H = ProbGatePolicy(
        feat_dim=feat_dim_H, beta=args.beta, gate_mode=args.gate_mode,
        learn_global_scale=True, init_gamma=args.gamma0, lr=3e-4, tau=args.tau
    ).to(device)
    policy_Q = ProbGatePolicy(
        feat_dim=feat_dim_Q, beta=args.beta, gate_mode=args.gate_mode,
        learn_global_scale=True, init_gamma=args.gamma0, lr=3e-4, tau=args.tau
    ).to(device)

    # 回滚守门
    rb = RollbackController(
        adaptor, kl_guard=args.kl_guard, mode=args.rollback_mode, base_rho=0.3, tau=0.02
    )

    # 优化器
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 对应抑制
    gs = GradSuppressor(model, layer_index=args.layer_index, rho=args.grad_suppress_rho) if args.grad_suppress_on else None

    # 特征追踪
    tracker = FeatureTracker(model, layer_index=args.layer_index,
                             feat_dim_H=feat_dim_H, feat_dim_Q=feat_dim_Q, momentum=0.9, debug=args.debug)

    # PPO 缓存
    ppo_buf_H = PPOBuffer(device=device)
    ppo_buf_Q = PPOBuffer(device=device)

    # 日志
    os.makedirs(args.log_dir, exist_ok=True)
    csv = CSVLogger(args.log_dir, args.run_name,
                    fieldnames=["step", "epoch", "split", "loss", "acc", "note"])
    swan = SwanLogger(args.swan_project_name, args.run_name, config=vars(args))

    # 触发窗口
    triggered = (args.use_train_acc_trigger == 0)
    acc_window = deque(maxlen=max(1, args.trigger_window))
    step = 0

    # Val1/Val2 时钟
    val1_cd = args.val1_cooldown or args.cooldown
    last_val1_step = -10 ** 9
    val1_round_in_cycle = 0
    cycle_base_snap = None
    last_accept_masks = None

    class TopKBuf:
        def __init__(self, K): self.K = K; self.items = []
        def push(self, item):
            self.items.append(item)
            self.items.sort(key=lambda t: t["score"], reverse=True)
            self.items = self.items[:self.K]
        def topM(self, M): return self.items[:M]

    topk_pool = TopKBuf(args.topK * max(1, args.val2_every_rounds))

    def _action_code(s: str) -> float:
        s = str(s).lower()
        if s == "accept": return 1.0
        if s in ("rollback-soft", "soft"): return 0.5
        return 0.0

    last_XH_for_ppo = None
    last_XQ_for_ppo = None

    # ================ 训练循环 ================
    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in tr_ld:
            step += 1
            vi, am, yl = _split_text_batch(batch, device)
            logits = model(input_ids=vi, attention_mask=am)
            loss = F.cross_entropy(logits, yl)

            # ---- backward ----
            optim.zero_grad(set_to_none=True)
            loss.backward()

            with torch.no_grad():
                tracker.update_grad_stats()

            if gs is not None and last_accept_masks is not None:
                mH_mask, mQ_mask = last_accept_masks
                if hasattr(gs, "apply"):
                    gs.apply(mask_H=mH_mask, mask_Q=mQ_mask)
                elif hasattr(gs, "step"):
                    gs.step()
                last_accept_masks = None
                if args.debug:
                    print("[GradSuppress] applied once for accepted candidate.")

            optim.step()

            # ---- train acc ----
            acc = accuracy_from_logits(logits.detach(), yl)
            if args.debug and (step % max(1, args.log_every // 2) == 0):
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                    print(f"[Train] step={step} | loss={float(loss.item()):.4f} | acc={acc:.4f} "
                          f"| mem_alloc={allocated:.1f}MB mem_reserved={reserved:.1f}MB")
                else:
                    print(f"[Train] step={step} | loss={float(loss.item()):.4f} | acc={acc:.4f}")

            if args.use_train_acc_trigger:
                acc_window.append(acc)
                if (not triggered) and len(acc_window) == acc_window.maxlen and (
                    sum(acc_window) / len(acc_window) >= args.trigger_acc
                ):
                    triggered = True
                    swan.log({"trigger/step": step})
                    if args.debug:
                        acc_mean = sum(acc_window) / len(acc_window)
                        print(f"[Trigger] step={step} | acc_win_mean={acc_mean:.4f} ≥ {args.trigger_acc:.2f} → TRIGGERED")
            elif args.debug and (step % max(1, args.log_every // 2) == 0):
                print("[Trigger] disabled (use_train_acc_trigger=0), training proceeds.")

            # ===== Val1/Val2 解耦 =====
            ready_for_val1 = (step - last_val1_step >= val1_cd)
            if triggered and ready_for_val1:
                if args.debug:
                    print(f"[Val1-START] step={step} | last_val1_step={last_val1_step} | cd={val1_cd}")
                if val1_round_in_cycle == 0:
                    rb.take_snapshot()
                    cycle_base_snap = adaptor.snapshot()

                # ---- Val1: 生成候选并打分 ----
                topk_local = TopKBuf(args.topK)
                adaptor.load_snapshot_(cycle_base_snap)

                XH = tracker.build_features_H().to(device)
                XQ = tracker.build_features_Q().to(device)
                last_XH_for_ppo = XH.detach()
                last_XQ_for_ppo = XQ.detach()

                cand_vecs_Q: List[torch.Tensor] = []
                cand_vecs_H: List[torch.Tensor] = []

                for ridx in range(args.R_candidates):
                    mH, epsH, logpH, pH = policy_H(XH, sigma=args.sigma_H)
                    mQ, epsQ, logpQ, pQ = policy_Q(XQ, sigma=args.sigma_Q)

                    if isinstance(cycle_base_snap, tuple):
                        aH0, aQ0 = cycle_base_snap
                        zH = vector_noise_like(aH0, kind="smooth", bw=2)
                        zQ = vector_noise_like(aQ0, kind="smooth", bw=2)
                        ampH = mH * epsH
                        ampQ = mQ * epsQ
                        aH_cand = aH0 + ampH * zH
                        aQ_cand = aQ0 + ampQ * zQ
                        cand_alpha = (aH_cand, aQ_cand)
                        logp_total = (logpH + logpQ)
                        cand_vecs_H.append(aH_cand.detach().cpu())
                        cand_vecs_Q.append(aQ_cand.detach().cpu())
                    else:
                        aQ0 = cycle_base_snap
                        zQ = vector_noise_like(aQ0, kind="smooth", bw=2)
                        ampQ = mQ * epsQ
                        aQ_cand = aQ0 + ampQ * zQ
                        cand_alpha = aQ_cand
                        logp_total = logpQ
                        mH = None
                        cand_vecs_Q.append(aQ_cand.detach().cpu())

                    # ======= 诊断区：base vs cand Δlogits =======
                    b1 = next(iter(va1_ld))
                    vi1, am1, yl1 = _split_text_batch(b1, device)

                    model.eval()
                    with torch.no_grad():
                        # base
                        adaptor.load_snapshot_(cycle_base_snap)
                        was_probe = getattr(model, "probe_runtime_enable", None)
                        if was_probe is not None: model.set_probe(False)
                        logits_base = model(input_ids=vi1, attention_mask=am1)
                        # cand
                        adaptor.load_snapshot_(cand_alpha)
                        logits_cand = model(input_ids=vi1, attention_mask=am1)
                        if was_probe is not None:
                            model.set_probe(bool(was_probe))
                            model.clear_attn_buffer()

                        d = (logits_cand - logits_base).abs()
                        dmax = float(d.max().item())
                        dmean = float(d.mean().item())

                        rel_l2 = _alpha_rel_l2(cand_alpha, cycle_base_snap)

                        score_base = accuracy_from_logits(logits_base, yl1)
                        score = accuracy_from_logits(logits_cand, yl1)
                        delta_acc = score - score_base

                    model.train()

                    logp_scalar = float(torch.as_tensor(logp_total).detach().mean().item())

                    it = {
                        "score": score,
                        "alpha": cand_alpha,
                        "logp": logp_scalar,
                        "mH": (mH.detach().cpu() if mH is not None else None),
                        "mQ": mQ.detach().cpu(),
                        "pH": (pH.detach().cpu() if mH is not None else None),
                        "pQ": pQ.detach().cpu()
                    }
                    topk_local.push(it)

                    if args.debug:
                        try:
                            if isinstance(cand_alpha, tuple):
                                aH, aQ = cand_alpha
                                aH_mean = float(aH.detach().abs().mean().item())
                                aQ_mean = float(aQ.detach().abs().mean().item())
                            else:
                                aH_mean = None
                                aQ = cand_alpha
                                aQ_mean = float(aQ.detach().abs().mean().item())
                        except Exception:
                            aH_mean, aQ_mean = None, None

                        rel_l2_txt = "n/a" if rel_l2 is None else f"{rel_l2:.3e}"
                        mHmag = float(it["mH"].abs().mean().item()) if it["mH"] is not None else None
                        mQmag = float(it["mQ"].abs().mean().item())
                        print(f"[Val1-CAND] idx={ridx} | base={score_base:.4f} cand={score:.4f} Δacc={delta_acc:+.4f} | logp={logp_scalar:.4f} "
                              f"| mean|mH|={mHmag} | mean|mQ|={mQmag}")
                        print(f"[DEBUG-EFF] ridx={ridx} | Δlogits_max={dmax:.3e} Δlogits_mean={dmean:.3e} | "
                              f"alpha_relL2={rel_l2_txt} | mean|aH|={aH_mean} | mean|aQ|={aQ_mean}")

                    if args.v2_debug:
                        try:
                            if it["mH"] is not None:
                                tensor_fp("VAL1.cand[mH]", it["mH"])
                            tensor_fp("VAL1.cand[mQ]", it["mQ"])
                        except Exception:
                            pass

                # 候选独立性：打印余弦相似度矩阵（前 8×8）
                if args.debug:
                    if len(cand_vecs_Q) > 1:
                        pairwise_cosines(cand_vecs_Q, name="V1.cand.cos(Q)")
                    if len(cand_vecs_H) > 1:
                        pairwise_cosines(cand_vecs_H, name="V1.cand.cos(H)")

                for it in topk_local.topM(args.topK):
                    topk_pool.push(it)

                if args.debug:
                    tops = ", ".join([f"{i}:{it['score']:.4f}" for i, it in enumerate(topk_pool.items)])
                    print(f"[Val1-TOPK] pool_size={len(topk_pool.items)} | scores={tops}")

                v1_loss_cd, v1_acc_cd = eval_loader(model, va1_ld, device)
                csv.log(step=step, epoch=ep, split="val1_cd", loss=v1_loss_cd, acc=v1_acc_cd, note=f"cd={val1_cd}")
                swan.log({"outer/step": step, "val1_cd/loss": v1_loss_cd, "val1_cd/acc":  v1_acc_cd})

                val1_round_in_cycle += 1
                last_val1_step = step
                swan.log({"outer/val1_round": val1_round_in_cycle, "outer/step": step})
                if args.debug:
                    print(f"[Val1-END] step={step} | round_in_cycle={val1_round_in_cycle} | pool_size={len(topk_pool.items)}")

                # ---- 是否执行 Val2 ----
                if (val1_round_in_cycle % args.val2_every_rounds) == 0:
                    evald = []
                    adaptor.load_snapshot_(cycle_base_snap)

                    if args.v2_debug:
                        try:
                            _topM = topk_pool.topM(args.val2_topM)
                            _scores = [float(t['score']) for t in _topM]
                            print(f"[V2-PREP] step={step} | M={len(_topM)} | scores={_scores}")
                            model_mode_fp(model, note="V2-pre")
                        except Exception:
                            pass

                    for k, item in enumerate(topk_pool.topM(args.val2_topM)):
                        if args.v2_debug:
                            try:
                                if isinstance(item["alpha"], tuple):
                                    aH, aQ = item["alpha"]
                                    tensor_fp("V2.alpha[aH]", aH)
                                    tensor_fp("V2.alpha[aQ]", aQ)
                                else:
                                    tensor_fp("V2.alpha[aQ]", item["alpha"])
                                if item.get("mH", None) is not None:
                                    tensor_fp("V2.cand[mH]", item["mH"])
                                if item.get("mQ", None) is not None:
                                    tensor_fp("V2.cand[mQ]", item["mQ"])
                                model_mode_fp(model, note=f"V2-evald-k{k}")
                            except Exception:
                                pass

                        acc0_mean, acc1_mean, delta_mean, se = robust_eval_val2(
                            model, adaptor, va2_ld,
                            base_snap=cycle_base_snap,
                            cand_snap=item["alpha"],
                            seeds=5
                        )
                        LCB = delta_mean - 1.64 * se

                        # 注意力域距离（取同一小批）
                        try:
                            b_small = next(iter(va2_ld))
                            vi2, am2, yl2 = _split_text_batch(b_small, device)

                            adaptor.load_snapshot_(cycle_base_snap)
                            model.set_probe(True)
                            _ = model(input_ids=vi2, attention_mask=am2)
                            attn_base = model.pop_attn_logits()

                            adaptor.load_snapshot_(item["alpha"])
                            _ = model(input_ids=vi2, attention_mask=am2)
                            attn_cand = model.pop_attn_logits()

                            if attn_base is not None and attn_cand is not None:
                                B_attn = attn_base.shape[0]
                                mask_for_attn = am2
                                if mask_for_attn.dim() in (2, 3) and mask_for_attn.size(0) != B_attn:
                                    mask_for_attn = mask_for_attn[:B_attn]
                                if args.debug:
                                    debug_attn_shapes(attn_base, attn_cand, mask_for_attn)

                                if args.v2_debug:
                                    try:
                                        attn_metrics(attn_base, attn_cand,
                                                     bits=args.attn_quant_bits,
                                                     pool=args.probe_pool,
                                                     name=f"V2-k{k}")
                                    except Exception:
                                        pass

                                D_att = attn_distribution_distance(
                                    attn_cand, ref_scores=attn_base, mask=mask_for_attn, metric="js"
                                )
                                if args.debug:
                                    mb = float(attn_base.exp().mean().item())
                                    mc = float(attn_cand.exp().mean().item())
                                    print(f"[V2-ATTN] k={k} | path=JS | mean|prob| base={mb:.5f} cand={mc:.5f} | D_att(JS)={D_att:.6f}")
                            else:
                                adaptor.load_snapshot_(cycle_base_snap)
                                out0 = model(input_ids=vi2, attention_mask=am2)
                                adaptor.load_snapshot_(item["alpha"])
                                out1 = model(input_ids=vi2, attention_mask=am2)
                                D_att = float(torch.norm(out1 - out0, p='fro').item())
                                if args.debug:
                                    m0 = float(out0.abs().mean().item()); m1 = float(out1.abs().mean().item())
                                    print(f"[V2-ATTN] k={k} | path=FRO | mean|out| base={m0:.5f} cand={m1:.5f} | D_att(Fro)={D_att:.6f}")
                        except Exception as e:
                            D_att = 0.0
                            if args.debug:
                                print(f"[V2-ATTN] ERROR fallback D_att=0.0 | err={e}")
                        finally:
                            model.clear_attn_buffer()
                            model.set_probe(bool(args.probe_attn))

                        # 梯度一致性
                        try:
                            adaptor.load_snapshot_(cycle_base_snap)
                            g_base = flat_grad_over_loader(model, va2_ld, device, steps=1)
                            adaptor.load_snapshot_(item["alpha"])
                            g_cand = flat_grad_over_loader(model, va2_ld, device, steps=1)
                            cos = grad_consistency(g_base, g_cand)
                            if args.debug:
                                nb = float(g_base.norm(p=2).item())
                                nc = float(g_cand.norm(p=2).item())
                                print(f"[V2-GRAD] k={k} | cos={cos:.6f} | ||g_base||={nb:.5e} ||g_cand||={nc:.5e} | size={g_base.numel()}")
                        except Exception as e:
                            cos = 1.0
                            if args.debug:
                                print(f"[V2-GRAD] ERROR fallback cos=1.0 | err={e}")

                        if args.debug:
                            print(f"[V2-ROBUST] k={k} | acc_base_mean={acc0_mean:.4f} | acc_cand_mean={acc1_mean:.4f} | "
                                  f"delta_mean={delta_mean:.4f} | se={se:.6f} | LCB={LCB:.6f}")

                        # alpha round-trip：确认 adaptor 未改写候选
                        try:
                            adaptor.load_snapshot_(item["alpha"])
                            snap_back = adaptor.snapshot()
                            if isinstance(snap_back, tuple) and isinstance(item["alpha"], tuple):
                                madH = max_abs_diff(snap_back[0], item["alpha"][0])
                                madQ = max_abs_diff(snap_back[1], item["alpha"][1])
                                print(f"[V2-ALPHA-CHECK] k={k} | max|ΔaH|={madH:.3e} | max|ΔaQ|={madQ:.3e}")
                                # 同时打印相对 base 的 L∞
                                baseH, baseQ = cycle_base_snap
                                dbH = max_abs_diff(item["alpha"][0], baseH)
                                dbQ = max_abs_diff(item["alpha"][1], baseQ)
                                print(f"[V2-ALPHA-DELTA] k={k} | vs BASE: max|aH_cand-aH0|={dbH:.3e} | max|aQ_cand-aQ0|={dbQ:.3e}")
                            elif not isinstance(snap_back, tuple) and not isinstance(item["alpha"], tuple):
                                madQ = max_abs_diff(snap_back, item["alpha"])
                                dbQ = max_abs_diff(item["alpha"], cycle_base_snap)
                                print(f"[V2-ALPHA-CHECK] k={k} | max|ΔaQ|={madQ:.3e}")
                                print(f"[V2-ALPHA-DELTA] k={k} | vs BASE: max|aQ_cand-aQ0|={dbQ:.3e}")
                            else:
                                print(f"[V2-ALPHA-CHECK] k={k} | shape-kind mismatch between saved and loaded alpha")
                        except Exception as e:
                            print(f"[V2-ALPHA-CHECK] k={k} | skipped: {e}")

                        evald.append((item, LCB, D_att, cos))

                    # 选择策略
                    if args.val2_select == "greedy":
                        evald.sort(key=lambda t: t[1], reverse=True)
                        chosen = [evald[0]]
                    else:
                        evald.sort(key=lambda t: t[1], reverse=True)
                        chosen = evald[:max(1, args.softmerge_topx)]
                        lcbs = torch.tensor([t[1] for t in chosen], dtype=torch.float32, device=device)
                        w = torch.softmax(lcbs, dim=0)
                        if isinstance(cycle_base_snap, tuple):
                            aH_new = 0; aQ_new = 0
                            for (it, _, _, _), ww in zip(chosen, w):
                                aH, aQ = it["alpha"]
                                aH_new = aH_new + ww.item() * aH
                                aQ_new = aQ_new + ww.item() * aQ
                            merged_alpha = (aH_new, aQ_new)
                        else:
                            aQ_new = 0
                            for (it, _, _, _), ww in zip(chosen, w):
                                aQ_new = aQ_new + ww.item() * it["alpha"]
                            merged_alpha = aQ_new
                        chosen = [({"alpha": merged_alpha,
                                    "mH": chosen[0][0]["mH"],
                                    "mQ": chosen[0][0]["mQ"],
                                    "logp": chosen[0][0]["logp"]},
                                   float(lcbs.max().item()), 0.0, 0.0)]

                    # ---- 判定/回滚 ----
                    item, LCB, D_att, cos = chosen[0]
                    adaptor.load_snapshot_(cycle_base_snap)
                    b2 = next(iter(va2_ld))
                    vi2, am2, yl2 = _split_text_batch(b2, device)
                    model.eval()
                    with torch.no_grad():
                        adaptor.load_snapshot_(cycle_base_snap)
                        logits_base = model(input_ids=vi2, attention_mask=am2); acc_base = accuracy_from_logits(logits_base, yl2)
                        adaptor.load_snapshot_(item["alpha"])
                        logits_cand = model(input_ids=vi2, attention_mask=am2); acc_cand = accuracy_from_logits(logits_cand, yl2)

                        if args.v2_debug:
                            try:
                                logits_delta_stats(logits_base, logits_cand, name="V2")
                            except Exception:
                                pass
                    model.train()

                    eps_tol = 1e-3
                    delta_final = acc_cand - acc_base
                    r = max(0.0, LCB - eps_tol)
                    if acc_cand < acc_base:
                        r -= 0.01
                    pass_gates = (r > 0.0) and (D_att <= args.kl_guard) and (cos >= 0.0)

                    if args.v2_debug:
                        try:
                            gate_trace({
                                "LCB": float(LCB),
                                "r": float(r),
                                "tol": float(eps_tol),
                                "D_att": float(D_att),
                                "kl_guard": float(args.kl_guard),
                                "cos": float(cos),
                                "cos_hi": 0.0,
                                "delta_final": float(delta_final),
                                "pass_flag": bool(pass_gates),
                            })
                        except Exception:
                            pass

                    ok, _ = rb.decide_and_apply(delta_final if pass_gates else -1e9)

                    if args.debug:
                        print(f"[V2-GATES] r={r:.6f} | tol={eps_tol} | D_att={D_att:.6f} ≤ {args.kl_guard:.6f} | "
                              f"cos={cos:.6f} ≥ 0  → pass={pass_gates} | delta_final={delta_final:.6f}")
                        print(f"[ROLLBACK] action={rb.last['action']} | kl={rb.last.get('kl', None)} | "
                              f"accept_delta={rb.last.get('accept_delta', None)} | ok={ok}")

                    ppo_buf_H.add(item.get("logp", 0.0), r)
                    ppo_buf_Q.add(item.get("logp", 0.0), r)

                    if ok and gs is not None:
                        mH = item.get("mH", None)
                        mQ = item.get("mQ", None)
                        mH = torch.tensor(mH, device=device, dtype=torch.float32) if mH is not None else None
                        mQ = torch.tensor(mQ, device=device, dtype=torch.float32) if mQ is not None else None
                        last_accept_masks = (mH, mQ)
                        if args.debug:
                            h_mag2 = float(mH.abs().mean().item()) if mH is not None else None
                            q_mag2 = float(mQ.abs().mean().item()) if mQ is not None else None
                            print(f"[Accept] next GradSuppress | mean|mH|={h_mag2} | mean|mQ|={q_mag2}")

                    note = f"V2 LCB={LCB:.4f}, D_att={D_att:.4f}, cos={cos:.3f}, delta={delta_final:.4f}, {rb.last['action']}"
                    csv.log(step=step, epoch=ep, split="qv2", loss=0.0, acc=acc_cand, note=note)
                    swan.log({
                        "outer/step": step,
                        "outer/LCB": LCB,
                        "outer/D_att": D_att,
                        "outer/cos": cos,
                        "outer/delta": delta_final,
                        "outer/action_code": _action_code(rb.last["action"]),
                        "outer/val1_round": val1_round_in_cycle
                    })
                    if args.debug:
                        print(f"[Val2] step={step} | {note}")

                    v2m_loss, v2m_acc = eval_loader(model, va2_ld, device)
                    csv.log(step=step, epoch=ep, split="val2_mid", loss=v2m_loss, acc=v2m_acc,
                            note=f"every_rounds={args.val2_every_rounds}")
                    swan.log({"outer/step": step, "val2_mid/loss": v2m_loss, "val2_mid/acc":  v2m_acc})

                    # PPO surrogate（稳妥）
                    try:
                        advH = ppo_buf_H.advantages().mean().to(device)
                        advQ = ppo_buf_Q.advantages().mean().to(device)
                        if last_XH_for_ppo is None or last_XQ_for_ppo is None:
                            XH_sur = tracker.build_features_H().to(device)
                            XQ_sur = tracker.build_features_Q().to(device)
                        else:
                            XH_sur = last_XH_for_ppo.to(device)
                            XQ_sur = last_XQ_for_ppo.to(device)

                        _, _, logpH_curr, _ = policy_H(XH_sur, sigma=args.sigma_H)
                        lossH = -(advH * logpH_curr.mean())
                        optH = getattr(policy_H, "opt", None) or getattr(policy_H, "optimizer", None)
                        if optH is None: raise RuntimeError("policy_H has no optimizer.")
                        optH.zero_grad(set_to_none=True); lossH.backward(); optH.step()

                        _, _, logpQ_curr, _ = policy_Q(XQ_sur, sigma=args.sigma_Q)
                        lossQ = -(advQ * logpQ_curr.mean())
                        optQ = getattr(policy_Q, "opt", None) or getattr(policy_Q, "optimizer", None)
                        if optQ is None: raise RuntimeError("policy_Q has no optimizer.")
                        optQ.zero_grad(set_to_none=True); lossQ.backward(); optQ.step()
                    except Exception as e:
                        print(f"[PPO-SURROGATE] update skipped due to error: {e}")

                    # 清空本周期
                    topk_pool = TopKBuf(args.topK * max(1, args.val2_every_rounds))
                    val1_round_in_cycle = 0
                    cycle_base_snap = None

            # ---- 周期性日志 ----
            if step % args.log_every == 0:
                tr_loss = float(loss.item()); tr_acc = float(acc)
                v1_loss, v1_acc = eval_loader(model, va1_ld, device)
                csv.log(step=step, epoch=ep, split="train", loss=tr_loss, acc=tr_acc, note="")
                csv.log(step=step, epoch=ep, split="val1", loss=v1_loss, acc=v1_acc, note="")
                swan.log({"step": step, "train/loss": tr_loss, "train/acc": tr_acc,
                          "val1/loss": v1_loss, "val1/acc": v1_acc})

        # epoch end
        v2_loss, v2_acc = eval_loader(model, va2_ld, device)
        te_loss, te_acc = eval_loader(model, te_ld, device)
        csv.log(step=step, epoch=ep, split="val2", loss=v2_loss, acc=v2_acc, note="")
        csv.log(step=step, epoch=ep, split="test", loss=te_loss, acc=te_acc, note="")

        swan.log({"epoch": ep, "val2/loss": v2_loss, "val2/acc":  v2_acc,
                  "test/loss": te_loss, "test/acc":  te_acc, "step": step})

        print(f"[Epoch {ep}] val2_acc={v2_acc:.4f} | test_acc={te_acc:.4f}")


if __name__ == "__main__":
    main()

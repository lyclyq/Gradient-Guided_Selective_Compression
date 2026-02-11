# /home/lyclyq/Optimization/grad-shake-align/src/trainer.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Deque, Optional
from collections import deque
import copy
from contextlib import contextmanager, nullcontext

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .loggingx import RunLogger
from .shake_align import ShakeAlignController, BlockStats
from .lora_layers import debug_check_dualrank_init


def _cfg_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _named_dualrank_lora_modules(model) -> Dict[str, torch.nn.Module]:
    out: Dict[str, torch.nn.Module] = {}
    for name, m in model.named_modules():
        if hasattr(m, "lora_A_r") and hasattr(m, "lora_A_hi"):
            out[name] = m
    return out


def _flatten_branch_grads(mod: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return scale-invariant gradient vectors (votes):
      g_r = concat(vec(dA_r), vec(dB_r)) / scaling_r
      g_hi = concat(vec(dA_hi), vec(dB_hi)) / scaling_hi

    IMPORTANT:
      - This is the ONLY place outside ShakeAlign that touches scaling,
        and it only DIVIDES to remove scaling for decision-making.
      - Writing scaled grads back happens ONLY inside ShakeAlign.
    """
    device = next(mod.parameters()).device

    if not hasattr(mod, "scaling_r") or not hasattr(mod, "scaling_hi"):
        raise RuntimeError(
            f"[ScalingMissing] {type(mod).__name__} missing scaling_r/scaling_hi. Module={mod}"
        )

    sr = float(getattr(mod, "scaling_r"))
    shi = float(getattr(mod, "scaling_hi"))
    if abs(sr) < 1e-12 or abs(shi) < 1e-12:
        raise RuntimeError(f"[ScalingInvalid] sr={sr} shi={shi} (must be nonzero)")

    def g_or_zeros(p: torch.nn.Parameter, s: float) -> torch.Tensor:
        if p.grad is None:
            return torch.zeros_like(p, device=device).flatten()
        g = p.grad.detach()
        g = g / s
        return g.flatten()

    g_r = torch.cat(
        [g_or_zeros(mod.lora_A_r, sr), g_or_zeros(mod.lora_B_r, sr)],
        dim=0,
    )
    g_hi = torch.cat(
        [g_or_zeros(mod.lora_A_hi, shi), g_or_zeros(mod.lora_B_hi, shi)],
        dim=0,
    )
    return g_r, g_hi


@contextmanager
def _set_dualrank_use_hi(model, use_hi: bool):
    touched = []
    for m in model.modules():
        if hasattr(m, "use_hi") and hasattr(m, "lora_A_hi") and hasattr(m, "lora_B_hi"):
            touched.append((m, bool(getattr(m, "use_hi"))))
            setattr(m, "use_hi", bool(use_hi))
    try:
        yield
    finally:
        for m, old in touched:
            setattr(m, "use_hi", old)


@torch.no_grad()
def evaluate_acc(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
    *,
    use_hi: Optional[bool] = None,
) -> float:
    was_training = bool(model.training)
    model.eval()
    correct = 0
    total = 0

    ctx = nullcontext() if use_hi is None else _set_dualrank_use_hi(model, use_hi)

    with ctx:
        for i, batch in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            preds = out.logits.argmax(dim=-1)
            labels = batch["labels"]
            correct += (preds == labels).sum().item()
            total += labels.numel()

    model.train(was_training)
    return correct / max(total, 1)


def _dbg_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = cfg.get("debug", {}) or {}
    return {
        "enabled": bool(d.get("enabled", False)),
        "print_every_steps": int(d.get("print_every_steps", 50)),
        "max_blocks_to_print": int(d.get("max_blocks_to_print", 3)),
        "dump_init": bool(d.get("dump_init", True)),
        "dump_votes": bool(d.get("dump_votes", True)),
        "dump_gates": bool(d.get("dump_gates", True)),
        "dump_grad_norms": bool(d.get("dump_grad_norms", True)),
        "dump_history": bool(d.get("dump_history", False)),
        "assert_hi_zero_init": bool(d.get("assert_hi_zero_init", True)),
    }


def _vote_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ours = cfg.get("method", {}).get("ours", {}) or {}
    voting = ours.get("voting", {}) or {}
    return {
        "samples_per_vote": int(voting.get("samples_per_vote", 8)),
        "keep_single_votes": bool(voting.get("keep_single_votes", True)),
        "allow_tail": bool(voting.get("allow_tail", True)),
    }


def _vote_history_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ours = cfg.get("method", {}).get("ours", {}) or {}
    hist = ours.get("history", {}) or {}
    return {
        "enabled": bool(hist.get("enabled", False)),
        "steps": int(hist.get("window_steps", 4)),
    }


def _split_indices(n: int, chunk: int, allow_tail: bool) -> List[Tuple[int, int]]:
    out = []
    s = 0
    while s < n:
        e = min(s + chunk, n)
        if (e - s) < chunk and (not allow_tail):
            break
        out.append((s, e))
        s = e
    return out


def _avg_last_k(seq: List[float], k: int) -> float:
    if not seq:
        return float("-inf")
    k = max(1, int(k))
    take = seq[-k:] if len(seq) >= k else seq
    return float(sum(take) / len(take))


def train_one(
    cfg: dict,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: RunLogger,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])
    warmup_ratio = float(cfg["train"]["warmup_ratio"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    lora_modules = _named_dualrank_lora_modules(model)

    dbg = _dbg_cfg(cfg)
    vote_cfg = _vote_cfg(cfg)
    vh_cfg = _vote_history_cfg(cfg)

    is_ours = (cfg.get("method", {}).get("name", "") == "ours")

    # -------- eval config --------
    stage = str(cfg.get("stage", "") or "").strip().lower()

    eval_strategy = str(_cfg_get(cfg, "train.eval.strategy", "per_epoch")).strip().lower()
    if eval_strategy == "epoch":
        eval_strategy = "per_epoch"

    if stage != "final" and eval_strategy == "dense_early":
        eval_strategy = "per_epoch"

    dense_eval_per_epoch = _as_int(_cfg_get(cfg, "train.eval.dense_early_per_epoch", 8), 8)
    dense_early_epochs = _as_int(_cfg_get(cfg, "train.eval.dense_early_epochs", 2), 2)

    eval_every_steps = _as_int(_cfg_get(cfg, "train.eval.every_steps", 50), 50)
    eval_first_step = _as_bool(_cfg_get(cfg, "train.eval.first_step", False), False)
    eval_max_batches = _as_int(_cfg_get(cfg, "train.eval.max_batches", 0), 0)

    compute_train_acc = _as_bool(_cfg_get(cfg, "train.eval.compute_train_acc", True), True)
    train_acc_max_batches = _as_int(_cfg_get(cfg, "train.eval.train_max_batches", 0), 0)

    eval_r_only = _as_bool(_cfg_get(cfg, "train.eval.log_r_only", True), True)

    if eval_strategy not in {"dense_early", "per_epoch", "steps", "none"}:
        print(f"[WARN] Unknown train.eval.strategy={eval_strategy!r}, fallback to per_epoch")
        eval_strategy = "per_epoch"

    def _should_eval(
        *,
        global_step: int,
        step_in_epoch: int,
        steps_in_epoch: int,
        is_epoch_end: bool,
        ep: int,
    ) -> bool:
        if eval_strategy == "none":
            return False

        if is_epoch_end:
            return True

        if eval_strategy == "per_epoch":
            if eval_first_step and ep == 1 and step_in_epoch == 1:
                return True
            return False

        if eval_strategy == "dense_early":
            if ep > dense_early_epochs:
                return False
            k = max(1, int(dense_eval_per_epoch))
            every = max(1, steps_in_epoch // k)
            if step_in_epoch >= steps_in_epoch:
                return False
            return (step_in_epoch % every) == 0

        if eval_first_step and global_step == 1:
            return True
        if eval_every_steps <= 0:
            return False
        return (global_step % eval_every_steps) == 0

    # -------- controller config: disable double smoothing --------
    cfg_ctrl = copy.deepcopy(cfg)
    cfg_ctrl.setdefault("method", {})
    cfg_ctrl["method"].setdefault("ours", {})
    cfg_ctrl["method"]["ours"]["ema_H"] = 1
    cfg_ctrl["method"]["ours"].setdefault("history", {})
    cfg_ctrl["method"]["ours"]["history"]["enabled"] = False

    controller = ShakeAlignController(cfg_ctrl) if is_ours else None
    if controller is not None:
        controller.set_lora_modules(lora_modules)

    if controller is not None and dbg["enabled"] and dbg["dump_init"]:
        debug_check_dualrank_init(
            model,
            assert_hi_zero=dbg["assert_hi_zero_init"],
            max_blocks_to_print=dbg["max_blocks_to_print"],
        )

    best_val = -1.0
    best_epoch = -1

    val_history_epoch: List[float] = []
    val_r_only_history_epoch: List[float] = []
    train_history_epoch: List[float] = []
    train_r_only_history_epoch: List[float] = []

    global_step = 0
    last_eval_global_step = -1

    # -------- vote history buffers --------
    vote_hist_enabled = bool(vh_cfg["enabled"])
    vote_hist_steps = max(1, int(vh_cfg["steps"]))
    vote_hist_r: Dict[str, Deque[torch.Tensor]] = {n: deque(maxlen=vote_hist_steps) for n in lora_modules.keys()}
    vote_hist_hi: Dict[str, Deque[torch.Tensor]] = {n: deque(maxlen=vote_hist_steps) for n in lora_modules.keys()}

    def _do_eval_and_log(step: int, ep: int) -> None:
        nonlocal best_val, best_epoch, last_eval_global_step
        if step == last_eval_global_step:
            return

        val_acc = evaluate_acc(model, val_loader, device, max_batches=eval_max_batches, use_hi=True)
        payload: Dict[str, Any] = {"val/acc": float(val_acc), "epoch": int(ep)}

        val_acc_r = None
        if is_ours and eval_r_only:
            val_acc_r = evaluate_acc(model, val_loader, device, max_batches=eval_max_batches, use_hi=False)
            payload["val/acc_r_only"] = float(val_acc_r)

        if compute_train_acc:
            tr_acc = evaluate_acc(model, train_loader, device, max_batches=train_acc_max_batches, use_hi=True)
            payload["train/acc"] = float(tr_acc)
            payload["gap/train_minus_val"] = float(tr_acc - val_acc)

            if is_ours and eval_r_only:
                tr_acc_r = evaluate_acc(model, train_loader, device, max_batches=train_acc_max_batches, use_hi=False)
                payload["train/acc_r_only"] = float(tr_acc_r)
                if val_acc_r is not None:
                    payload["gap_r_only/train_minus_val_r_only"] = float(tr_acc_r - float(val_acc_r))

        logger.log(step, payload)
        last_eval_global_step = step

        if val_acc > best_val:
            best_val = float(val_acc)
            best_epoch = int(ep)

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{epochs}")

        steps_in_epoch = len(train_loader)

        for step_in_epoch, batch in enumerate(pbar, start=1):
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            step_loss_val = 0.0

            if controller is None:
                out = model(**batch)
                loss = out.loss
                loss.backward()
                step_loss_val = float(loss.item())
                logger.log(global_step, {"train/loss": step_loss_val})

            else:
                bs = int(batch["input_ids"].shape[0])
                spv = int(vote_cfg["samples_per_vote"])
                allow_tail = bool(vote_cfg["allow_tail"])

                windows = _split_indices(bs, spv, allow_tail=allow_tail)
                if len(windows) == 0:
                    windows = [(0, bs)]

                step_votes_r: Dict[str, List[torch.Tensor]] = {n: [] for n in lora_modules.keys()}
                step_votes_hi: Dict[str, List[torch.Tensor]] = {n: [] for n in lora_modules.keys()}

                total_grads: Dict[torch.nn.Parameter, torch.Tensor] = {}
                loss_mean_batch = 0.0

                for (s, e) in windows:
                    sub = {k: v[s:e] for k, v in batch.items()}
                    win_weight = float(e - s) / float(bs)
                    if win_weight <= 0.0:
                        raise RuntimeError(f"[trainer] invalid window weight: s={s} e={e} bs={bs}")

                    opt.zero_grad(set_to_none=True)
                    out = model(**sub)
                    win_loss = out.loss
                    loss_mean_batch += float(win_loss.detach().item()) * win_weight
                    win_loss.backward()

                    # scale-invariant votes (divide scaling here)
                    for name, mod in lora_modules.items():
                        g_r, g_hi = _flatten_branch_grads(mod)
                        step_votes_r[name].append(g_r)
                        step_votes_hi[name].append(g_hi)

                    # accumulate REAL grads (still scaled) for optimizer step later,
                    # but will be overwritten by ShakeAlign write-back anyway.
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is None:
                                continue
                            if p not in total_grads:
                                total_grads[p] = p.grad.detach().clone() * win_weight
                            else:
                                total_grads[p].add_(p.grad.detach(), alpha=win_weight)

                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for p, g in total_grads.items():
                        p.grad = g
                step_loss_val = float(loss_mean_batch)

                packed_step_r: Dict[str, torch.Tensor] = {}
                packed_step_hi: Dict[str, torch.Tensor] = {}
                for name in lora_modules.keys():
                    if not step_votes_r[name]:
                        continue
                    packed_step_r[name] = torch.stack(step_votes_r[name], dim=0)
                    packed_step_hi[name] = torch.stack(step_votes_hi[name], dim=0)

                if vote_hist_enabled:
                    for name in lora_modules.keys():
                        if name in packed_step_r:
                            vote_hist_r[name].append(packed_step_r[name].detach())
                            vote_hist_hi[name].append(packed_step_hi[name].detach())

                votes_r: Dict[str, torch.Tensor] = {}
                votes_hi: Dict[str, torch.Tensor] = {}
                for name in lora_modules.keys():
                    if vote_hist_enabled:
                        if len(vote_hist_r[name]) == 0:
                            continue
                        votes_r[name] = torch.cat(list(vote_hist_r[name]), dim=0)
                        votes_hi[name] = torch.cat(list(vote_hist_hi[name]), dim=0)
                    else:
                        if name not in packed_step_r:
                            continue
                        votes_r[name] = packed_step_r[name]
                        votes_hi[name] = packed_step_hi[name]

                stats: Dict[str, BlockStats] = {}
                vote_sums: Dict[str, Dict[str, torch.Tensor]] = {}
                single_vote_blocks = 0

                for name in lora_modules.keys():
                    if name not in votes_r:
                        continue
                    vr = votes_r[name]
                    vhi = votes_hi[name]
                    if vr.shape[0] < 2:
                        single_vote_blocks += 1
                        continue

                    fresh = controller.compute_stats_from_votes(vr, vhi)
                    smooth = controller.ema_update(name, fresh)
                    stats[name] = smooth
                    vote_sums[name] = {"votes_r": vr, "votes_hi": vhi}

                if dbg["enabled"] and dbg["dump_votes"] and (global_step % dbg["print_every_steps"] == 0):
                    any_name = next(iter(votes_r.keys()), None)
                    v_total = int(votes_r[any_name].shape[0]) if any_name else 0
                    print(f"[DBG][Step={global_step}] vote_hist={vote_hist_enabled} H={vote_hist_steps} V_total≈{v_total}")
                    if single_vote_blocks > 0:
                        print(f"[DBG][Step={global_step}] single-vote blocks skipped={single_vote_blocks}")

                # IMPORTANT: write-back scaling happens ONLY inside ShakeAlign
                info = controller.apply_in_place_corrections(
                    lora_modules=lora_modules,
                    stats=stats,
                    vote_sums=vote_sums,
                    debug=bool(dbg["enabled"] and dbg["dump_gates"]),
                    grad_norm_trace=bool(dbg["enabled"] and dbg["dump_grad_norms"]),
                    debug_history=bool(dbg["enabled"] and dbg["dump_history"]),
                )

                logger.log(
                    global_step,
                    {
                        "train/loss": float(step_loss_val),
                        "train/gate0_triggered_blocks": info.get("triggered_blocks", 0.0),
                        "train/single_vote_skipped_blocks": float(single_vote_blocks),
                        "train/tau_N": info.get("tau_N", 0.0),
                        "train/tau_D": info.get("tau_D", 0.0),
                    },
                )

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": f"{float(step_loss_val):.4f}"})

            if _should_eval(
                global_step=global_step,
                step_in_epoch=step_in_epoch,
                steps_in_epoch=steps_in_epoch,
                is_epoch_end=False,
                ep=ep,
            ):
                _do_eval_and_log(global_step, ep)

        if _should_eval(
            global_step=global_step,
            step_in_epoch=steps_in_epoch,
            steps_in_epoch=steps_in_epoch,
            is_epoch_end=True,
            ep=ep,
        ):
            _do_eval_and_log(global_step, ep)

            # epoch-end summary snapshots
            val_acc = evaluate_acc(model, val_loader, device, max_batches=eval_max_batches, use_hi=True)
            val_history_epoch.append(float(val_acc))

            if is_ours and eval_r_only:
                val_acc_r = evaluate_acc(model, val_loader, device, max_batches=eval_max_batches, use_hi=False)
                val_r_only_history_epoch.append(float(val_acc_r))

            if compute_train_acc:
                tr_acc = evaluate_acc(model, train_loader, device, max_batches=train_acc_max_batches, use_hi=True)
                train_history_epoch.append(float(tr_acc))

                if is_ours and eval_r_only:
                    tr_acc_r = evaluate_acc(model, train_loader, device, max_batches=train_acc_max_batches, use_hi=False)
                    train_r_only_history_epoch.append(float(tr_acc_r))

    # -------- summary metrics --------
    if len(val_history_epoch) == 0:
        val_max = float(best_val)
        val_final = float(best_val)
        val_avg_last3 = float(best_val)
    else:
        val_max = float(max(val_history_epoch))
        val_final = float(val_history_epoch[-1])
        val_avg_last3 = _avg_last_k(val_history_epoch, 3)

    out: Dict[str, float] = {
        "best_val_acc": float(best_val),
        "best_epoch": float(best_epoch),
        "val_max": float(val_max),
        "val_final": float(val_final),
        "val_avg": float(val_avg_last3),
        "val_avg_last3ep": float(val_avg_last3),
    }

    if is_ours and eval_r_only and len(val_r_only_history_epoch) > 0:
        out["val_r_only_max"] = float(max(val_r_only_history_epoch))
        out["val_r_only_final"] = float(val_r_only_history_epoch[-1])
        out["val_r_only_avg_last3ep"] = float(_avg_last_k(val_r_only_history_epoch, 3))

    if len(train_history_epoch) > 0:
        out["train_final"] = float(train_history_epoch[-1])
        out["train_max"] = float(max(train_history_epoch))

    if is_ours and eval_r_only and len(train_r_only_history_epoch) > 0:
        out["train_r_only_final"] = float(train_r_only_history_epoch[-1])
        out["train_r_only_max"] = float(max(train_r_only_history_epoch))

    return out

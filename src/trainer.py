# src/trainer.py
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .loggingx import RunLogger
from .shake_align import ShakeAlignController, BlockStats


def _named_dualrank_lora_modules(model) -> Dict[str, torch.nn.Module]:
    out: Dict[str, torch.nn.Module] = {}
    for name, m in model.named_modules():
        # DualRankLoRALinear has these attrs in your lora_layers.py
        if hasattr(m, "lora_A_r") and hasattr(m, "lora_A_hi"):
            out[name] = m
    return out


def _flatten_branch_grads(mod: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (g_r_flat, g_hi_flat) from DualRankLoRALinear grads.
    If grad missing, returns zeros of matching shape.
    """
    device = next(mod.parameters()).device

    def g_or_zeros(p: torch.nn.Parameter) -> torch.Tensor:
        if p.grad is None:
            return torch.zeros_like(p, device=device).flatten()
        return p.grad.detach().flatten()

    g_r = torch.cat([g_or_zeros(mod.lora_A_r), g_or_zeros(mod.lora_B_r)], dim=0)
    g_hi = torch.cat([g_or_zeros(mod.lora_A_hi), g_or_zeros(mod.lora_B_hi)], dim=0)
    return g_r, g_hi


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out.logits.argmax(dim=-1)
        labels = batch["labels"]
        correct += (preds == labels).sum().item()
        total += labels.numel()
    model.train()
    return correct / max(total, 1)


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
    weight_decay = float(cfg["train"]["weight_decay"])
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    lora_modules = _named_dualrank_lora_modules(model)
    controller = ShakeAlignController(cfg) if cfg["method"]["name"] == "ours" else None

    best_val = -1.0
    best_epoch = -1
    val_history: List[float] = []

    global_step = 0
    eval_strategy = cfg["train"]["eval"]["strategy"]
    dense_eval_per_epoch = int(cfg["train"]["eval"].get("dense_early_per_epoch", 8))
    dense_early_epochs = int(cfg["train"]["eval"].get("dense_early_epochs", 2))

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{epochs}")

        for batch in pbar:
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            if controller is None:
                out = model(**batch)
                loss = out.loss
                loss.backward()
                logger.log(global_step, {"train/loss": float(loss.item())})
            else:
                V = int(cfg["method"]["ours"]["votes"])
                bs = batch["input_ids"].shape[0]
                micro = max(1, bs // V)

                # vote buffers per block
                votes_r: Dict[str, List[torch.Tensor]] = {n: [] for n in lora_modules.keys()}
                votes_hi: Dict[str, List[torch.Tensor]] = {n: [] for n in lora_modules.keys()}

                # accumulate total grads over votes
                total_grads: Dict[torch.nn.Parameter, torch.Tensor] = {}
                loss = None

                for v_idx in range(V):
                    s = v_idx * micro
                    e = min((v_idx + 1) * micro, bs)
                    if s >= e:
                        continue
                    sub = {k: v[s:e] for k, v in batch.items()}

                    opt.zero_grad(set_to_none=True)
                    out = model(**sub)
                    loss = out.loss
                    loss.backward()

                    # snapshot per-block vote grads
                    for name, mod in lora_modules.items():
                        g_r, g_hi = _flatten_branch_grads(mod)
                        votes_r[name].append(g_r)
                        votes_hi[name].append(g_hi)

                    # accumulate grads into total_grads
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is None:
                                continue
                            if p not in total_grads:
                                total_grads[p] = p.grad.detach().clone()
                            else:
                                total_grads[p].add_(p.grad.detach())

                # restore accumulated grad
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for p, g in total_grads.items():
                        p.grad = g

                # compute stats per block (+ EMA)
                stats: Dict[str, BlockStats] = {}
                vote_sums: Dict[str, Dict[str, torch.Tensor]] = {}
                for name in lora_modules.keys():
                    if len(votes_r[name]) == 0:
                        continue
                    vr = torch.stack(votes_r[name], dim=0)        # [V, d_r]
                    vhi = torch.stack(votes_hi[name], dim=0)      # [V, d_hi]
                    fresh = controller.compute_stats_from_votes(vr, vhi)
                    smooth = controller.ema_update(name, fresh)
                    stats[name] = smooth
                    vote_sums[name] = {"sum_r": vr.sum(dim=0)}     # Σ_r

                # apply correction
                info = controller.apply_in_place_corrections(lora_modules, stats, vote_sums)

                if loss is None:
                    # should not happen
                    loss_val = 0.0
                else:
                    loss_val = float(loss.item())

                logger.log(
                    global_step,
                    {
                        "train/loss": loss_val,
                        "train/triggered_blocks": info.get("triggered_blocks", 0.0),
                        "train/align_thr": info.get("alignment_threshold", 0.0),
                    },
                )

            # step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": f"{float(loss.item()) if loss is not None else 0.0:.4f}"})

            # dense early eval
            if eval_strategy == "dense_early" and ep <= dense_early_epochs:
                every = max(1, len(train_loader) // dense_eval_per_epoch)
                if (global_step % every) == 0:
                    val_acc = evaluate(model, val_loader, device)
                    val_history.append(val_acc)
                    logger.log(global_step, {"val/acc": val_acc})

                    if val_acc > best_val:
                        best_val = val_acc
                        best_epoch = ep

        # per-epoch eval
        if eval_strategy == "per_epoch":
            val_acc = evaluate(model, val_loader, device)
            val_history.append(val_acc)
            logger.log(global_step, {"val/acc": val_acc, "epoch": ep})

            if val_acc > best_val:
                best_val = val_acc
                best_epoch = ep

    # summarize val stats
    if len(val_history) == 0:
        val_max = best_val
        val_final = best_val
        val_avg = best_val
    else:
        val_max = float(max(val_history))
        val_final = float(val_history[-1])
        val_avg = float(sum(val_history) / len(val_history))

    return {
        "best_val_acc": float(best_val),
        "best_epoch": float(best_epoch),
        "val_max": val_max,
        "val_final": val_final,
        "val_avg": val_avg,
    }

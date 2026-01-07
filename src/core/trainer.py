import os, math, time
import torch
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd

from src.utils.metrics import compute_acc
from src.signals.consistency import compute_consistency_votes
from src.signals.soft_router import soft_router
from src.signals.projection import randomized_low_rank_proj
from src.signals.persistence import PersistenceEMA
from src.distill.absorb import absorb_if_mature


def _collect_adapters(model):
    adapters = []
    for m in model.modules():
        if m.__class__.__name__ == "AdapterLinear":
            adapters.append(m)
    return adapters


def _to_device(batch, device):
    """
    Supports:
      - dict[str, Tensor]
      - transformers.BatchEncoding (has .to)
    """
    if hasattr(batch, "to") and callable(getattr(batch, "to")):
        return batch.to(device)
    # plain dict
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


def _get_labels(batch):
    """
    DataCollatorWithPadding usually returns 'labels'.
    Some datasets/collate return 'label'.
    """
    if hasattr(batch, "get") and callable(getattr(batch, "get")):
        lab = batch.get("labels", None)
        if lab is None:
            lab = batch.get("label", None)
        return lab
    # very defensive fallback
    if "labels" in batch:
        return batch["labels"]
    if "label" in batch:
        return batch["label"]
    return None


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = []
    labels = []
    for batch in loader:
        batch = _to_device(batch, device)
        lab = _get_labels(batch)
        if lab is None:
            raise KeyError("Batch has no 'labels' or 'label' key. Please check your dataloader/collator.")

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None),
            labels=None,
        )
        logits = out.logits
        p = torch.argmax(logits, dim=-1)
        preds.extend(p.detach().cpu().tolist())
        labels.extend(lab.detach().cpu().tolist())
    return compute_acc(preds, labels)


def train_one(
    cfg,
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    exp_name: str,
    out_csv_path: str,
    router_mode: bool = False,
    alt_mode: str = "none",
):
    """
    alt_mode: none|ab_only|alt_only|offline_project
    router_mode: use soft routing & manual grads for adapters (ours)
    """
    model.to(device)
    model.train()

    adapters = _collect_adapters(model)

    # optimizer: adapter params + classifier head params
    params = []
    for a in adapters:
        if a.use_ab:
            params += [a.A, a.B]
        if a.use_alt:
            params += [a.U, a.V, a.z]
    for p in model.classifier.parameters():
        if p.requires_grad:
            params.append(p)

    opt = AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    ema = PersistenceEMA(beta=0.9)

    rows = []
    step = 0

    for ep in range(int(cfg["epochs"])):
        model.train()
        pbar = tqdm(train_loader, desc=f"{exp_name} ep{ep+1}", leave=False)

        for batch in pbar:
            step += 1
            batch = _to_device(batch, device)
            lab = _get_labels(batch)
            if lab is None:
                raise KeyError("Batch has no 'labels' or 'label' key. Please check your dataloader/collator.")

            # forward for logging/loss (classifier head needs loss too)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids", None),
                labels=lab,
            )
            loss = out.loss

            opt.zero_grad(set_to_none=True)

            if router_mode:
                # ===== microbatch votes: split batch into V chunks =====
                V = int(cfg.get("vote_chunks", 4))
                bsz = batch["input_ids"].shape[0]
                chunk = max(1, bsz // V)

                G_votes_by_adapter = {id(a): [] for a in adapters}

                for vi in range(V):
                    s = vi * chunk
                    e = bsz if vi == V - 1 else (vi + 1) * chunk
                    if s >= bsz:
                        break

                    # build sub-batch (preserve keys; labels key may be 'labels' or 'label')
                    sub = {k: v[s:e] for k, v in batch.items()}

                    model.zero_grad(set_to_none=True)
                    sub_lab = _get_labels(sub)
                    if sub_lab is None:
                        raise KeyError("Sub-batch has no 'labels' or 'label' key. Please check batching logic.")

                    o = model(
                        input_ids=sub["input_ids"],
                        attention_mask=sub["attention_mask"],
                        token_type_ids=sub.get("token_type_ids", None),
                        labels=sub_lab,
                    )
                    l = o.loss
                    l.backward()

                    # estimate deltaW-space gradients per adapter
                    for a in adapters:
                        G_total = None

                        # AB gradient estimate
                        if a.use_ab and (a.A.grad is not None) and (a.B.grad is not None):
                            Bt_pinv = torch.linalg.pinv(a.B.t())
                            At_pinv = torch.linalg.pinv(a.A.t())
                            G1 = a.A.grad @ Bt_pinv
                            G2 = At_pinv @ a.B.grad
                            G_ab = 0.5 * (G1 + G2)
                            G_total = G_ab

                        # alt gradient estimate
                        if a.use_alt and (a.U.grad is not None) and (a.V.grad is not None) and (a.z.grad is not None):
                            Z = torch.diag(a.z.detach())
                            VZ = a.V.detach() @ Z
                            UZ = a.U.detach() @ Z
                            VZt_pinv = torch.linalg.pinv(VZ.t())
                            UZt_pinv = torch.linalg.pinv(UZ.t())
                            G1 = a.U.grad @ VZt_pinv
                            G2 = (UZt_pinv @ a.V.grad.t()).t()
                            G_alt = 0.5 * (G1 + G2)
                            G_total = G_alt if G_total is None else (G_total + G_alt)

                        if G_total is None:
                            continue

                        G_votes_by_adapter[id(a)].append(G_total.detach())

                # Now compute routing and set adapter grads.
                # We will backprop loss only for classifier head; then overwrite adapter grads.
                model.zero_grad(set_to_none=True)

                # classifier head grads from loss
                loss.backward(retain_graph=False)

                # clear any accidental adapter grads from that backward
                for a in adapters:
                    if a.use_ab:
                        a.A.grad = None
                        a.B.grad = None
                    if a.use_alt:
                        a.U.grad = None
                        a.V.grad = None
                        a.z.grad = None

                for a in adapters:
                    votes = G_votes_by_adapter.get(id(a), [])
                    if len(votes) == 0:
                        continue

                    G_star = torch.stack(votes, dim=0).mean(dim=0)
                    r = int(cfg["r"])
                    # Use r_alt as the "R" probe rank if available; else fallback
                    R = int(cfg.get("r_alt", max(r * 2, r + 1)))

                    C_ab, C_perp, C_R = compute_consistency_votes(votes, r=r, R=R)

                    signal = float(torch.linalg.norm(G_star).item())
                    P = min(1.0, ema.update(key=id(a), signal_strength=signal / (signal + 1.0)))

                    w = soft_router(
                        C_ab,
                        C_perp,
                        C_R,
                        P,
                        tau_ab=cfg.get("tau_ab", 0.7),
                        tau_perp=cfg.get("tau_s", 0.5),
                        tau_delta=cfg.get("tau_delta", 0.12),
                        k=cfg.get("k_sigmoid", 10.0),
                    )

                    # projections
                    G_r = randomized_low_rank_proj(G_star, r)
                    G_res = G_star - G_r

                    # AB gets comp part of low-rank
                    if a.use_ab:
                        G_ab_tgt = w["comp"] * G_r
                        a.set_grads_from_G_ab(G_ab_tgt)

                    # alt gets (alt + noise + overfit) portion of residual; suppress AB-subspace drift if overfit
                    if a.use_alt:
                        G_alt_tgt = (w["alt"] + w["noise"] + w["overfit"]) * G_res
                        beta = float(cfg.get("beta_suppress", 0.7))
                        if w["overfit"] > 0:
                            G_alt_tgt = G_alt_tgt - beta * w["overfit"] * randomized_low_rank_proj(G_alt_tgt, r)
                        a.set_grads_from_G_alt(G_alt_tgt)

                    # periodic absorb if mature (route A)
                    if cfg.get("absorb_every_steps", 0) > 0 and (step % int(cfg["absorb_every_steps"]) == 0):
                        _ = absorb_if_mature(
                            a,
                            r=r,
                            tau_cap=float(cfg.get("tau_cap", 0.7)),
                            alpha=float(cfg.get("absorb_alpha", 0.7)),
                        )

                    # soft decay on alt params (leaky memory)
                    rho = float(cfg.get("rho_alt", 0.99))
                    if a.use_alt:
                        with torch.no_grad():
                            a.z.mul_(rho)

            else:
                # baseline training: regular backward
                loss.backward()

                if alt_mode == "alt_only":
                    # update only alt params
                    for a in adapters:
                        if a.use_ab:
                            if a.A.grad is not None:
                                a.A.grad.zero_()
                            if a.B.grad is not None:
                                a.B.grad.zero_()

                if alt_mode == "ab_only":
                    # update only AB params
                    for a in adapters:
                        if a.use_alt:
                            if a.U.grad is not None:
                                a.U.grad.zero_()
                            if a.V.grad is not None:
                                a.V.grad.zero_()
                            if a.z.grad is not None:
                                a.z.grad.zero_()

                # offline_project: train alt only, absorb later (handled outside in rte_suite)
                if alt_mode == "offline_project":
                    pass

            opt.step()

            if step % 50 == 0:
                val_acc = evaluate(model, val_loader, device)
                test_acc = evaluate(model, test_loader, device)
                rows.append(
                    {
                        "step": step,
                        "epoch": ep + 1,
                        "loss": float(loss.item()),
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                    }
                )
                pbar.set_postfix({"val": f"{val_acc:.3f}", "test": f"{test_acc:.3f}"})

    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)

    # final eval
    val_acc = evaluate(model, val_loader, device)
    test_acc = evaluate(model, test_loader, device)
    return val_acc, test_acc

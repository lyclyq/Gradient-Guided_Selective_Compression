# ===== FILE: trainbase.py (baseline with adjustable Val1/Val2 frequency) =====
import os
import argparse
import torch
import torch.nn.functional as F

from utils.logging_utils import CSVLogger, SwanLogger
from utils.data_utils import get_sst2
from models.transformer_text import TransformerTextClassifier
from models.distillbert_hf import DistillBertClassifier
from models.bert_hf import BertClassifier


# ---------- helpers ----------
def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=-1) == y).float().mean().item()


def _split_text_batch(batch, device):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attn = batch.get("attention_mask", None)
    if attn is None:
        attn = torch.ones_like(input_ids, device=device)
    else:
        attn = attn.to(device, non_blocking=True)
    labels = batch.get("labels", batch.get("label")).to(device, non_blocking=True)
    return input_ids, attn, labels


@torch.no_grad()
def eval_loader(model, loader, device):
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
    return tot_loss / max(1, n), tot_acc / max(1, n)


def build_model(name, tok=None, num_labels=2, hf_override=None, probe_args=None):
    name = name.lower()
    probe_args = probe_args or {}
    if name == "transformer":
        assert tok is not None
        return TransformerTextClassifier(
            vocab_size=tok.vocab_size,
            num_labels=num_labels,
            pad_id=tok.pad_token_id or 0
        )
    if name == "distillbert":
        return DistillBertClassifier(
            num_labels=num_labels,
            model_name=hf_override or "distilbert-base-uncased",
            **probe_args
        )
    if name == "bert":
        return BertClassifier(
            num_labels=num_labels,
            model_name=hf_override or "bert-base-uncased",
            **probe_args
        )
    raise ValueError(f"Unknown model {name}")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Pure baseline trainer (no masks / no policy / no outer loop) with adjustable Val1/Val2 frequency")
    ap.add_argument("--dataset", type=str, default="sst2")
    ap.add_argument("--model", type=str, default="transformer", choices=["transformer", "distillbert", "bert"])
    ap.add_argument("--hf_model", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    # logging
    ap.add_argument("--log_dir", type=str, default="logs_base")
    ap.add_argument("--run_name", type=str, default="baseline_run")
    ap.add_argument("--swan_project_name", type=str, default="baseline")
    ap.add_argument("--log_every", type=int, default=50)

    # probe (kept for interface parity; off by default)
    ap.add_argument("--probe_attn", action="store_true")
    ap.add_argument("--probe_max_T", type=int, default=256)
    ap.add_argument("--probe_pool", type=str, default="avg", choices=["none", "avg"])

    # NEW: decoupled Val1/Val2 frequency (parity with outerloop knobs)
    ap.add_argument("--val1_cooldown", type=int, default=None,
                    help="Trigger Val1 every N steps (if None, defaults to log_every).")
    ap.add_argument("--val2_every_rounds", type=int, default=1,
                    help="Run a mid-epoch Val2 every K Val1 rounds (1 = every Val1).")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset.lower() != "sst2":
        raise ValueError("Only sst2 is supported in this baseline runner.")
    tr_ld, va1_ld, va2_ld, te_ld, tok = get_sst2(
        batch_size=args.batch,
        model_name=args.hf_model or "distilbert-base-uncased"
    )

    probe_args = dict(
        attn_probe_enable=args.probe_attn,
        attn_probe_max_T=args.probe_max_T,
        attn_probe_pool=args.probe_pool,
    )

    model = build_model(args.model, tok=tok, hf_override=args.hf_model, probe_args=probe_args)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # logging init
    os.makedirs(args.log_dir, exist_ok=True)
    csv = CSVLogger(args.log_dir, args.run_name,
                    fieldnames=["step", "epoch", "split", "loss", "acc", "note"])
    swan = SwanLogger(args.swan_project_name, args.run_name, config=vars(args))

    # decoupled clocks (baseline)
    step = 0
    val1_cd = args.val1_cooldown or args.log_every
    last_val1_step = -10**9
    val1_round_in_cycle = 0

    # ================ training loop ================
    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in tr_ld:
            step += 1
            vi, am, yl = _split_text_batch(batch, device)

            logits = model(input_ids=vi, attention_mask=am)
            loss = F.cross_entropy(logits, yl)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            tr_acc = accuracy_from_logits(logits.detach(), yl)

            # periodic train + (optionally) val1 via log_every (kept for parity)
            if step % args.log_every == 0:
                v1_loss_le, v1_acc_le = eval_loader(model, va1_ld, device)
                csv.log(step=step, epoch=ep, split="train", loss=float(loss.item()), acc=float(tr_acc), note="")
                csv.log(step=step, epoch=ep, split="val1",  loss=v1_loss_le, acc=v1_acc_le, note="")
                swan.log({
                    "step": step,
                    "train/loss": float(loss.item()),
                    "train/acc":  float(tr_acc),
                    "val1/loss":  v1_loss_le,
                    "val1/acc":   v1_acc_le
                })

            # decoupled Val1: every val1_cd steps
            if (step - last_val1_step) >= val1_cd:
                v1_loss_cd, v1_acc_cd = eval_loader(model, va1_ld, device)
                csv.log(step=step, epoch=ep, split="val1_cd", loss=v1_loss_cd, acc=v1_acc_cd, note=f"cd={val1_cd}")
                swan.log({
                    "outer/step": step,           # 用 outer/step key 方便与你外环面板并列
                    "val1_cd/loss": v1_loss_cd,
                    "val1_cd/acc":  v1_acc_cd
                })
                val1_round_in_cycle += 1
                last_val1_step = step
                try:
                    print(f"[Val1] step={step} | round_in_cycle={val1_round_in_cycle}")
                except Exception:
                    pass

                # mid-epoch Val2 after K Val1 rounds
                if (val1_round_in_cycle % max(1, args.val2_every_rounds)) == 0:
                    v2m_loss, v2m_acc = eval_loader(model, va2_ld, device)
                    csv.log(step=step, epoch=ep, split="val2_mid", loss=v2m_loss, acc=v2m_acc, note=f"every_rounds={args.val2_every_rounds}")
                    swan.log({
                        "outer/step": step,
                        "val2_mid/loss": v2m_loss,
                        "val2_mid/acc":  v2m_acc
                    })
                    try:
                        print(f"[Val2-mid] step={step} | val2_mid_acc={v2m_acc:.4f}")
                    except Exception:
                        pass

        # epoch end
        v2_loss, v2_acc = eval_loader(model, va2_ld, device)
        te_loss, te_acc = eval_loader(model, te_ld, device)
        csv.log(step=step, epoch=ep, split="val2", loss=v2_loss, acc=v2_acc, note="")
        csv.log(step=step, epoch=ep, split="test", loss=te_loss, acc=te_acc, note="")
        swan.log({
            "epoch": ep,
            "val2/loss": v2_loss,
            "val2/acc":  v2_acc,
            "test/loss": te_loss,
            "test/acc":  te_acc
        })
        print(f"[Epoch {ep}] val2_acc={v2_acc:.4f} | test_acc={te_acc:.4f}")


if __name__ == "__main__":
    main()

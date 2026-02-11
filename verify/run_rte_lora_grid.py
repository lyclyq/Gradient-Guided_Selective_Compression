import csv
import time
import math
import random
from typing import Dict, List, Optional
from collections import deque

from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model


# -----------------------
# Experiment Config
# -----------------------
SEEDS = [2,3,5]
LR = 1e-3
EPOCHS = 5

MODEL_NAME = "bert-base-uncased"
TASK_NAME = "glue"
SUBSET_NAME = "rte"   # ✅ SST-2

# microbatch config
MICROBATCH_SIZE = 4
ACCUM_STEPS = 16
EFFECTIVE_BATCH = MICROBATCH_SIZE * ACCUM_STEPS  # 32

MAX_LEN = 256
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.09

# eval cadence (and also conflict interval)
EVAL_EVERY_STEPS = 1

# ✅ history window for conflict vote aggregation (exactly 2 steps)
CONFLICT_HIST_STEPS = 1

VAL_PROBE_SIZE = 512
VAL_BATCH_SIZE = 128

# early stop
MAX_OPT_STEPS = 4000
EARLY_STOP_HIGH_ACC = 0.80
EARLY_STOP_MIN_HIGH_EVALS = 300

# stage splits
STAGE_LOW_MAX = 0.60
STAGE_MID_MAX = 0.80

# LoRA
LORA_R = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["query", "value"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean().item()


def tokenize_batch(examples, tokenizer):
    # Pair tasks
    if "sentence1" in examples and "sentence2" in examples:
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )

    # QNLI
    if "question" in examples and "sentence" in examples:
        return tokenizer(
            examples["question"],
            examples["sentence"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )

    # MNLI
    if "premise" in examples and "hypothesis" in examples:
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )

    # SST-2 single sentence
    if "sentence" in examples:
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )

    raise KeyError(f"Unknown text fields: {list(examples.keys())}")


def prepare_datasets(tokenizer):
    ds = load_dataset(TASK_NAME, SUBSET_NAME)
    ds = ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True)

    if "label" in ds["train"].column_names:
        ds = ds.rename_column("label", "labels")

    keep = ["input_ids", "attention_mask", "labels"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    ds.set_format(type="torch")
    return ds["train"], ds["validation"]


def build_model():
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=TARGET_MODULES,
    )
    return get_peft_model(base, lora_cfg)


def lora_params(model):
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def eval_loss_acc(model, loader, device) -> (float, float):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        bs = batch["labels"].size(0)
        total_loss += out.loss.item() * bs
        total_acc += accuracy_from_logits(out.logits, batch["labels"]) * bs
        n += bs
    n = max(n, 1)
    return total_loss / n, total_acc / n


def grad_vector_from_params(params: List[torch.nn.Parameter]) -> torch.Tensor:
    vecs = []
    for p in params:
        if p.grad is None:
            vecs.append(torch.zeros_like(p).view(-1))
        else:
            vecs.append(p.grad.detach().view(-1))
    return torch.cat(vecs)


def cancel_score(grads: List[torch.Tensor], eps: float = 1e-12) -> float:
    # Cancel = 1 - ||mean(g)|| / mean(||g_i||)
    if len(grads) == 0:
        return 0.0
    with torch.no_grad():
        norms = torch.stack([g.norm() for g in grads])
        mean_norm = norms.mean()
        g_mean = torch.stack(grads).mean(dim=0)
        score = 1.0 - (g_mean.norm() / (mean_norm + eps))
        return float(score.clamp(min=0.0, max=2.0).item())


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def export_stage_csvs(master_csv_path: str, rows: List[Dict]):
    eval_rows = [r for r in rows if r.get("event") == "eval" and r.get("val_probe_acc") is not None]

    low, mid, high = [], [], []
    for r in eval_rows:
        acc = float(r["val_probe_acc"])
        if acc <= STAGE_LOW_MAX:
            low.append(r)
        elif acc < STAGE_MID_MAX:
            mid.append(r)
        else:
            high.append(r)

    base = master_csv_path.replace(".csv", "")
    write_csv(base + "_stage_low.csv", low)
    write_csv(base + "_stage_mid.csv", mid)
    write_csv(base + "_stage_high.csv", high)

    print(f"[stage export] low={len(low)} mid={len(mid)} high={len(high)}")
    print("Saved:", base + "_stage_low.csv")
    print("Saved:", base + "_stage_mid.csv")
    print("Saved:", base + "_stage_high.csv")


def train_one(seed: int, out_rows: List[Dict], out_csv_path: str):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ds, val_ds = prepare_datasets(tokenizer)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=MICROBATCH_SIZE, shuffle=True, generator=g)

    probe_n = min(VAL_PROBE_SIZE, len(val_ds))
    val_probe_ds = val_ds.select(list(range(probe_n)))
    val_probe_loader = DataLoader(val_probe_ds, batch_size=VAL_BATCH_SIZE, shuffle=False)

    model = build_model().to(device)
    params = lora_params(model)
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = math.ceil(len(train_ds) / EFFECTIVE_BATCH)
    total_opt_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_opt_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    global_step = 0
    t0 = time.time()

    last_probe_loss: Optional[float] = None
    last_probe_acc: Optional[float] = None
    last_delta_probe_loss: Optional[float] = None
    last_delta_probe_acc: Optional[float] = None

    # ✅ conflict history over last 2 optimizer steps
    cancel_hist2 = deque(maxlen=CONFLICT_HIST_STEPS)

    it = iter(train_loader)

    high_eval_count = 0
    stop_now = False

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(range(steps_per_epoch), desc=f"[seed={seed}] epoch {epoch}/{EPOCHS}", leave=True)

        for step_in_epoch in pbar:
            global_step += 1

            grad_sums = [torch.zeros_like(p, device=device) for p in params]
            micro_grads: List[torch.Tensor] = []

            # one optimizer step = 8 votes (8 microbatches)
            for _ in range(ACCUM_STEPS):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it)

                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                out = model(**batch)
                out.loss.backward()

                micro_grads.append(grad_vector_from_params(params))

                for j, p in enumerate(params):
                    if p.grad is not None:
                        grad_sums[j].add_(p.grad.detach())

            optimizer.zero_grad(set_to_none=True)
            scale = 1.0 / float(len(micro_grads))
            for j, p in enumerate(params):
                p.grad = grad_sums[j] * scale

            optimizer.step()
            scheduler.step()

            # step-level conflict (8 votes)
            conf_cancel_step = cancel_score(micro_grads)
            cancel_hist2.append(conf_cancel_step)

            # for convenience: hist2 mean conflict (if only 1 step so far, mean over 1)
            conf_cancel_hist2 = float(np.mean(list(cancel_hist2)))

            out_rows.append({
                "seed": seed,
                "lr": LR,
                "epoch": epoch,
                "global_step": global_step,
                "step_in_epoch": step_in_epoch,
                "microbatch_size": MICROBATCH_SIZE,
                "accum_steps": ACCUM_STEPS,
                "conf_cancel_step": conf_cancel_step,
                "conf_cancel_hist2": conf_cancel_hist2,
                "event": "step",
            })

            pbar.set_postfix({
                "gstep": global_step,
                "c_step": f"{conf_cancel_step:.3f}",
                "c_h2": f"{conf_cancel_hist2:.3f}",
                "vL": "None" if last_probe_loss is None else f"{last_probe_loss:.4f}",
                "dL": "None" if last_delta_probe_loss is None else f"{last_delta_probe_loss:+.4f}",
                "vA": "None" if last_probe_acc is None else f"{last_probe_acc:.3f}",
                "dA": "None" if last_delta_probe_acc is None else f"{last_delta_probe_acc:+.4f}",
                "highN": high_eval_count,
            })

            # eval every 2 steps (aligned with hist2)
            if global_step % EVAL_EVERY_STEPS == 0:
                # ✅ x = mean conflict of the last 2 optimizer steps
                avg_conf_interval = conf_cancel_hist2

                probe_loss, probe_acc = eval_loss_acc(model, val_probe_loader, device)

                delta_probe_loss = None
                if last_probe_loss is not None:
                    delta_probe_loss = probe_loss - last_probe_loss
                last_probe_loss = probe_loss
                last_delta_probe_loss = delta_probe_loss

                delta_probe_acc = None
                if last_probe_acc is not None:
                    delta_probe_acc = probe_acc - last_probe_acc
                last_probe_acc = probe_acc
                last_delta_probe_acc = delta_probe_acc

                if probe_acc >= EARLY_STOP_HIGH_ACC:
                    high_eval_count += 1

                out_rows.append({
                    "seed": seed,
                    "lr": LR,
                    "epoch": epoch,
                    "global_step": global_step,
                    "eval_every_steps": EVAL_EVERY_STEPS,
                    "conflict_hist_steps": CONFLICT_HIST_STEPS,
                    "val_probe_size": probe_n,
                    "avg_conf_cancel_interval": avg_conf_interval,
                    "val_probe_loss": probe_loss,
                    "delta_val_probe_loss": delta_probe_loss,
                    "val_probe_acc": probe_acc,
                    "delta_val_probe_acc": delta_probe_acc,
                    "event": "eval",
                })

                pbar.write(
                    f"[eval] seed={seed} ep={epoch} step={global_step} "
                    f"avg_cancel(hist{CONFLICT_HIST_STEPS})={avg_conf_interval:.3f} "
                    f"valLoss={probe_loss:.4f} dLoss={('None' if delta_probe_loss is None else f'{delta_probe_loss:+.4f}')} "
                    f"valAcc={probe_acc:.4f} dAcc={('None' if delta_probe_acc is None else f'{delta_probe_acc:+.4f}')} "
                    f"highN={high_eval_count}"
                )

                write_csv(out_csv_path, out_rows)

                if high_eval_count >= EARLY_STOP_MIN_HIGH_EVALS:
                    pbar.write(
                        f"[early-stop] reached high-stage eval points: {high_eval_count} (acc>={EARLY_STOP_HIGH_ACC})"
                    )
                    stop_now = True

            if global_step >= MAX_OPT_STEPS:
                pbar.write(f"[stop] reached MAX_OPT_STEPS={MAX_OPT_STEPS}")
                stop_now = True

            if stop_now:
                break

        if stop_now:
            break

    elapsed = time.time() - t0
    out_rows.append({
        "seed": seed,
        "lr": LR,
        "epoch": "SUMMARY",
        "global_step": global_step,
        "elapsed_sec": round(elapsed, 2),
        "event": "summary",
    })


def main():
    out_csv = f"{SUBSET_NAME}_lora_r{LORA_R}_lr{LR}_cancel_valloss_log.csv".replace(".", "p")
    rows: List[Dict] = []

    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    for seed in SEEDS:
        print(
            f"\n=== {SUBSET_NAME.upper()} LoRA(r={LORA_R}) seed={seed} lr={LR} epochs={EPOCHS} "
            f"microbatch={MICROBATCH_SIZE} accum={ACCUM_STEPS} (eff={EFFECTIVE_BATCH}) "
            f"eval_every={EVAL_EVERY_STEPS} conflict_hist={CONFLICT_HIST_STEPS} val_probe_size={VAL_PROBE_SIZE} "
            f"MAX_OPT_STEPS={MAX_OPT_STEPS} early_stop_high_acc={EARLY_STOP_HIGH_ACC} "
            f"min_high_evals={EARLY_STOP_MIN_HIGH_EVALS} ==="
        )
        train_one(seed, rows, out_csv)
        write_csv(out_csv, rows)

    print("\nDone. Saved:", out_csv)
    export_stage_csvs(out_csv, rows)


if __name__ == "__main__":
    main()

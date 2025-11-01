# train_eval_L_and_gva.py
import os, argparse, random, time
import numpy as np
import torch
import torch.nn.functional as F

from utils.logging_utils import CSVLogger, SwanLogger
from modules.gva import GVAProjector
from modules.laplace_spec import SpectralAlign, collect_group_flat_grads



def _split_text_batch(batch, device):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attn = batch.get("attention_mask", batch.get("attention_masks", None))
    if attn is None:
        # 兜底：没有 attention_mask 就给全1
        attn = torch.ones_like(input_ids, device=device)
    else:
        attn = attn.to(device, non_blocking=True)

    # 兼容 label / labels 两种键名
    if "labels" in batch:
        labels = batch["labels"]
    elif "label" in batch:
        labels = batch["label"]
    else:
        raise KeyError("Neither 'labels' nor 'label' found in text batch.")
    labels = labels.to(device, non_blocking=True)
    return input_ids, attn, labels

# ---------------- Model registry ----------------
def build_model(name, num_classes=10, text_num_labels=2):
    name = name.lower()
    if name == "small_transformer":
        from models.small_transformer import SmallViT
        return SmallViT(num_classes=num_classes)
    if name == "transformer":
        from models.transformer_baseline import TransformerBaseline
        return TransformerBaseline(num_classes=num_classes)
    if name == "small_bert":
        from models.small_bert import SmallBertClassifier
        return SmallBertClassifier(num_labels=text_num_labels)
    if name == "distillbert":
        from models.distillbert_hf import DistillBertClassifier
        return DistillBertClassifier(num_labels=text_num_labels)
    raise ValueError(f"Unknown model {name}")

def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()

def run_epoch_vision(model, loader, device, opt=None):
    model.train(mode=opt is not None)
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        if opt is None:
            with torch.no_grad():
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
        else:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
        b = xb.size(0)
        tot_loss += loss.item() * b
        tot_acc  += accuracy_from_logits(logits, yb) * b
        n += b
    return tot_loss / n, tot_acc / n

def run_epoch_text(model, loader, device, opt=None):
    model.train(mode=opt is not None)
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    # for batch in loader:
    #     input_ids = batch["input_ids"].to(device, non_blocking=True)
    #     attn = batch["attention_mask"].to(device, non_blocking=True)
    #     labels = batch["labels"].to(device, non_blocking=True)
    #     if opt is None:
    #         with torch.no_grad():
    #             logits = model(input_ids=input_ids, attention_mask=attn)
    #             loss = F.cross_entropy(logits, labels)
    #     else:
    #         opt.zero_grad(set_to_none=True)
    #         logits = model(input_ids=input_ids, attention_mask=attn)
    #         loss = F.cross_entropy(logits, labels)
    #         loss.backward()
    #         opt.step()

    for batch in loader:
        input_ids, attn, labels = _split_text_batch(batch, device)
        if opt is None:
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attn)
                loss = F.cross_entropy(logits, labels)
        else:
            opt.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attn)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()

        b = input_ids.size(0)
        tot_loss += loss.item() * b
        tot_acc  += accuracy_from_logits(logits, labels) * b
        n += b
    return tot_loss / n, tot_acc / n

def tiny_batch_from(loader):
    for x in loader:
        return x
    return None

def slice_first_k(batch, k, modality):
    if k <= 0:
        return batch
    if modality == "vision":
        xb, yb = batch
        return xb[:k], yb[:k]
    else:
        b = {}
        for key, val in batch.items():
            b[key] = val[:k]
        return b

def backward_on_batch(model, batch, device, modality):
    if modality == "vision":
        xb, yb = batch
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        return float(loss.item())
    # else:
    #     input_ids = batch["input_ids"].to(device, non_blocking=True)
    #     attn = batch["attention_mask"].to(device, non_blocking=True)
    #     labels = batch["labels"].to(device, non_blocking=True)
    #     logits = model(input_ids=input_ids, attention_mask=attn)
    #     loss = F.cross_entropy(logits, labels)
    #     loss.backward()
    #     return float(loss.item())
    else:
        input_ids, attn, labels = _split_text_batch(batch, device)
        logits = model(input_ids=input_ids, attention_mask=attn)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        return float(loss.item())


# ---- utils for best snapshot
def _save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def _copy_csv_as(src_csv, dst_csv):
    if os.path.isfile(src_csv):
        os.makedirs(os.path.dirname(dst_csv), exist_ok=True)
        with open(src_csv, "rb") as fsrc, open(dst_csv, "wb") as fdst:
            fdst.write(fsrc.read())

# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["vision","text"], required=True)
    p.add_argument("--dataset", choices=["mnist","sst2"], required=True)
    p.add_argument("--model", choices=["small_transformer","transformer","distillbert","small_bert"], required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42, help="Global seed (data split, shuffles, QV1 random).")

    # logging
    p.add_argument("--log_dir", type=str, default="logs/L_and_gva")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--swan_project_name", type=str, default="optimization_L", help="Custom SwanLab project name")
    p.add_argument("--log_every", type=int, default=50, help="每隔多少步写一条 train 日志（CSV & SwanLab）。<=0 表示每步。")
    p.add_argument("--speed_every", type=int, default=100, help="每隔多少步打印一次速度/显存/当前lr。")

    # split ratios
    p.add_argument("--val_ratio", type=float, default=0.1, help="Total ratio for QV1 if val2_ratio is None; else QV1 ratio.")
    p.add_argument("--val2_ratio", type=float, default=None, help="QV2 ratio; if None, val_ratio is split half/half.")

    # GVA options
    p.add_argument("--align_every", type=int, default=300)
    p.add_argument("--align_tau", type=float, default=0.0)
    p.add_argument("--align_eta", type=float, default=0.2)
    p.add_argument("--align_delta", type=float, default=0.3)
    p.add_argument("--align_beta", type=float, default=0.9)

    # Spectral metric
    p.add_argument("--spec_m", type=int, default=32)
    p.add_argument("--spec_scale_gva", action="store_true", help="Use spectral mismatch to modulate align_eta")

    # QV1 sampling controls
    p.add_argument("--qv1_cycle", action="store_true", help="Round-robin QV1 batches instead of always first batch.")
    p.add_argument("--qv1_rand",  action="store_true", help="Randomly pick a QV1 batch (seeded & reproducible). If set, overrides qv1_cycle.")
    p.add_argument("--qv1_k", type=int, default=0, help="If >0, take first k samples from the chosen QV1 batch (micro-batch).")
    p.add_argument("--qv1_full_every", type=int, default=0, help="Every N steps, run full QV1 aggregation (0=off).")

    # Noise weighting & QV2 meta-calibration (baseline)
    p.add_argument("--qv2_every", type=int, default=500, help="Steps between QV2 meta-calibration checks.")
    p.add_argument("--noise_prob", action="store_true", help="Enable sample-level noise weighting via temperature tau.")
    p.add_argument("--noise_tau", type=float, default=0.5, help="Initial temperature for noise probability.")
    p.add_argument("--noise_alpha", type=float, default=1.0, help="(reserved) weight for geometric term in score.")
    p.add_argument("--noise_beta", type=float, default=0.0, help="(reserved) bias from spectral mismatch into score.")

    # === New: L_spec -> tau (fast loop) & meta gain (slow loop)
    p.add_argument("--tau_from_spec", action="store_true",
                   help="Use L_spec trend to adjust tau frequently (fast loop).")
    p.add_argument("--tau_step0", type=float, default=0.02,
                   help="Base step size for tau update per L_spec trigger (multiplicative).")
    p.add_argument("--lspec_ema_beta", type=float, default=0.9,
                   help="EMA beta for L_spec smoothing.")
    p.add_argument("--meta_gain0", type=float, default=1.0,
                   help="Initial meta-gain for tau step.")
    p.add_argument("--meta_up", type=float, default=1.10,
                   help="When QV2 improves, amplify tau step by this ratio.")
    p.add_argument("--meta_down", type=float, default=0.90,
                   help="When QV2 degrades, shrink tau step by this ratio.")

    # === New: noise score variant
    p.add_argument("--noise_score", choices=["loss", "loss+conf"], default="loss",
                   help="Use per-sample loss or loss+confidence as noise score.")
    p.add_argument("--noise_beta_conf", type=float, default=0.5,
                   help="Weight for (1-confidence) when noise_score=loss+conf.")

    # === New: QV2 gates projection (optional supervisor for projection length)
    p.add_argument("--qv2_proj", action="store_true",
                   help="Let QV2 meta signal gate projection strength (if align_eta>0).")
    p.add_argument("--qv2_proj_up", type=float, default=1.10,
                   help="Gate multiplier when QV2 improves.")
    p.add_argument("--qv2_proj_down", type=float, default=0.90,
                   help="Gate multiplier when QV2 degrades.")
    p.add_argument("--qv2_proj_beta", type=float, default=0.9,
                   help="EMA for projection gate smoothing.")
    p.add_argument("--proj_cos_scale", action="store_true",
                   help="Scale projection by mean cosine (only when align_eta>0).")

    # === New: QV2 policy module selector
    p.add_argument("--qv2_mode", choices=["shrink", "dir_gate", "instant_reverse"], default="shrink",
                   help="shrink=Δacc≤0只缩小步长；dir_gate=Δacc≤0下次快环改变方向；instant_reverse=Δacc≤0当场反向小步")
    p.add_argument("--qv2_meta_eta", type=float, default=0.01,
                   help="instant_reverse模式下，当场反向的小步（对log(τ)的增量幅度）")

    # === New: cosine-meta + magnitude fusion + Adam gate ===
    p.add_argument("--use_cos_meta", action="store_true",
                   help="把 cos(sign) 并入方向判定：cos<0 视作几何相反，驱动换向。")
    p.add_argument("--tau_mag_mode", choices=["lspec","cos","prod"], default="prod",
                   help="τ 步长幅度来自 |ΔL_spec|、|cos| 或两者乘积。")
    p.add_argument("--tau_cos_pow", type=float, default=1.2,
                   help="|cos| 的幂次，控制对齐强弱的敏感度。")
    p.add_argument("--tau_lspec_scale", type=float, default=120.0,
                   help="把 |ΔL_spec| 送进 tanh 的尺度（越大越保守）。")

    p.add_argument("--adam_gate", choices=["off","wd","lr"], default="off",
                   help="把(几何×谱)信号用来门控 Adam：wd=调 weight_decay；lr=调学习率。")
    p.add_argument("--adam_gate_k", type=float, default=0.10,
                   help="Adam 门控强度系数 k。")
    p.add_argument("--adam_gate_beta", type=float, default=0.9,
                   help="Adam 门控信号 EMA 平滑系数。")

    args = p.parse_args()

    # ---- CUDA/设备信息 ----
    print(f"[Device] requested={args.device} | torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[CUDA] device_count={torch.cuda.device_count()} | current={torch.cuda.current_device()} | "
              f"name={torch.cuda.get_device_name(torch.cuda.current_device())}")
        torch.cuda.empty_cache()
    else:
        print("[CUDA] not available -> will run on CPU")

    # ----- global seeds -----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Data ----
    if args.task == "vision":
        from utils.data_utils import get_mnist
        tr, va1, va2, te = get_mnist(batch_size=args.batch,
                                     val_ratio=args.val_ratio,
                                     val2_ratio=args.val2_ratio,
                                     seed=args.seed)
        modality = "vision"; num_classes = 10
    else:
        from utils.data_utils import get_sst2
        model_name = "distilbert-base-uncased" if args.model=="distillbert" else "prajjwal1/bert-tiny"
        tr, va1, va2, te, tok = get_sst2(batch_size=max(16, args.batch//4),   # 注意：这里如果要“真实 batch”，把 max(...) 改为 args.batch
                                         val_ratio=args.val_ratio if args.val_ratio>0 else 0.1,
                                         model_name=model_name,
                                         seed=args.seed)
        modality = "text"; num_classes = 2

    # 实际 batch size 探测（从第一个 batch 读）
    first_tr = tiny_batch_from(tr)
    if modality == "vision":
        real_bs = first_tr[0].size(0)
    else:
        real_bs = first_tr["input_ids"].size(0)
    print(f"[Data] task={args.task} | dataset={args.dataset} | model={args.model}")
    print(f"[Data] loader sizes: |train|={len(tr)} |val1|={len(va1)} |val2|={len(va2)} |test|={len(te)}")
    print(f"[Data] args.batch={args.batch} -> real_batch_in_loader={real_bs}")
    print(f"[Q-config] align_every={args.align_every} | qv1_k={args.qv1_k} | qv1_rand={args.qv1_rand} | "
          f"qv2_every={args.qv2_every} | log_every={args.log_every} | speed_every={args.speed_every}")

    # ---- Model/opt ----
    model = build_model(args.model, num_classes=num_classes, text_num_labels=num_classes).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Adam gate baselines & EMA
    base_lrs = [pg["lr"] for pg in opt.param_groups]
    base_wds = [pg.get("weight_decay", 0.0) for pg in opt.param_groups]
    adam_gate_ema = 0.0

    # Groups
    groups = model.groups() if hasattr(model, "groups") else [("all", [p for p in model.parameters() if p.requires_grad])]

    # ---- Loggers ----
    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, f"{args.dataset}_{args.model}.csv")
    csv = CSVLogger(csv_path, fieldnames=[
        "step","epoch","split","loss","acc",
        "gva_cos","gva_numproj","lspec","eta_eff",
        "qv2_acc","tau","qv1_full","meta_gain","qv2_gate","qv2_dir",
        "cos_est","adam_gate_ema"
    ])
    swan = SwanLogger(project=args.swan_project_name,
                      run_name=args.run_name or f"{args.dataset}_{args.model}",
                      config=vars(args))
    print(f"[Init] CSV -> {csv_path}")
    print(f"[Init] SwanLab project='{args.swan_project_name}' run='{args.run_name or f'{args.dataset}_{args.model}'}'")

    # ---- Tools ----
    gva = GVAProjector(beta=args.align_beta, tau=args.align_tau, eta_proj=args.align_eta, delta_max=args.align_delta)
    spec = SpectralAlign(m=args.spec_m)

    # ---- States ----
    tau = float(args.noise_tau)
    last_qv2_acc = None
    printed_once_proj = False

    lspec_ema = None
    meta_gain = args.meta_gain0
    last_tau_update_dir = 0     # +1/-1
    qv2_gate = 1.0
    qv2_dir = +1                # for dir_gate mode
    cos_est = -1.0              # keep as float for logging

    # Best model tracking (for auto checkpoint & best.csv)
    best_val_acc = -1.0
    best_test_acc = -1.0
    best_dir = os.path.join(args.log_dir, "best_snapshots", f"{args.dataset}_{args.model}")
    best_model_path = os.path.join(best_dir, "model_best.pt")
    best_csv_path   = os.path.join(best_dir, "best.csv")

    # QV1 iter helpers
    qv1_iter = iter(va1)
    rng_qv1 = np.random.RandomState(args.seed + 1337)
    qv1_len = len(va1)

    # ---- 速度计时器 ----
    t_speed = None
    speed_count = 0

    step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        for batch in iter(tr):
            step += 1
            if t_speed is None:
                # 同步一次，开始测速窗
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_speed = time.time()
                speed_count = 0

            opt.zero_grad(set_to_none=True)

            # ---- train forward
            if modality == "vision":
                xb, yb = batch[0].to(args.device, non_blocking=True), batch[1].to(args.device, non_blocking=True)
                logits = model(xb)
                losses_vec = F.cross_entropy(logits, yb, reduction='none')

                # noise score
                if args.noise_score == "loss+conf":
                    with torch.no_grad():
                        prob = torch.softmax(logits, dim=-1)
                        conf = prob.gather(1, yb.view(-1,1)).squeeze(1)
                        geo_proxy = (1.0 - conf)
                    S_loss = (losses_vec - losses_vec.mean()) / (losses_vec.std(unbiased=False) + 1e-8)
                    S_geo  = (geo_proxy  - geo_proxy.mean())  / (geo_proxy.std(unbiased=False)  + 1e-8)
                    S = S_loss + args.noise_beta_conf * S_geo
                else:
                    mean = losses_vec.mean()
                    std  = losses_vec.std(unbiased=False) + 1e-8
                    S = (losses_vec - mean) / std

                if args.noise_prob:
                    pprob = torch.sigmoid(S / max(1e-4, tau))
                    weights = (1.0 - pprob).detach()
                    loss = (weights * losses_vec).mean()
                else:
                    loss = losses_vec.mean()

                loss.backward()
                train_acc = (logits.argmax(-1) == yb).float().mean().item()
            # else:
            #     # input_ids = batch["input_ids"].to(args.device, non_blocking=True)
            #     # attn = batch["attention_mask"].to(args.device, non_blocking=True)
            #     # labels = batch["labels"].to(args.device, non_blocking=True)
            #     # logits = model(input_ids=input_ids, attention_mask=attn)
            #     # loss = F.cross_entropy(logits, labels)
            #     # loss.backward()
            #     # train_acc = (logits.argmax(-1) == labels).float().mean().item()
            #     input_ids, attn, labels = _split_text_batch(batch, args.device)
            #     logits = model(input_ids=input_ids, attention_mask=attn)
            #     loss = F.cross_entropy(logits, labels)
            #     loss.backward()
            #     train_acc = (logits.argmax(-1) == labels).float().mean().item()
            else:
                # --- TEXT branch with sample-wise mask (noise_prob) ---
                input_ids, attn, labels = _split_text_batch(batch, args.device)
                logits = model(input_ids=input_ids, attention_mask=attn)

                # per-sample loss
                losses_vec = F.cross_entropy(logits, labels, reduction='none')

                # noise score S（与 vision 分支一致）
                if args.noise_score == "loss+conf":
                    with torch.no_grad():
                        prob = torch.softmax(logits, dim=-1)
                        conf = prob.gather(1, labels.view(-1,1)).squeeze(1)
                        geo_proxy = (1.0 - conf)
                    S_loss = (losses_vec - losses_vec.mean()) / (losses_vec.std(unbiased=False) + 1e-8)
                    S_geo  = (geo_proxy  - geo_proxy.mean())  / (geo_proxy.std(unbiased=False)  + 1e-8)
                    S = S_loss + args.noise_beta_conf * S_geo
                else:
                    mean = losses_vec.mean()
                    std  = losses_vec.std(unbiased=False) + 1e-8
                    S = (losses_vec - mean) / std

                # 概率掩码（冻结到 τ：detach）
                if args.noise_prob:
                    pprob   = torch.sigmoid(S / max(1e-4, tau))
                    weights = (1.0 - pprob).detach()     # 关键：切断到 τ / S 的梯度
                    loss    = (weights * losses_vec).mean()
                else:
                    loss = losses_vec.mean()

                loss.backward()
                train_acc = (logits.argmax(-1) == labels).float().mean().item()


            # ---- save masked TRAIN grads snapshot (after loss.backward) ----
            _saved_tr_grads = []
            for p in model.parameters():
                if p.grad is None:
                    _saved_tr_grads.append(None)
                else:
                    _saved_tr_grads.append(p.grad.detach().clone())


            gva_stats = {"cos_mean": None, "num_proj": 0}
            eta_eff = args.align_eta
            lspec_val = None
            qv1_full_flag = 0




            # ---- QV1: spectral (+ optional projection) ----
            if (args.align_every > 0) and (step % args.align_every == 0) and (qv1_len > 0):
                # choose a QV1 batch
                if args.qv1_rand:
                    target_idx = int(rng_qv1.randint(0, qv1_len))
                    for bi, b in enumerate(va1):
                        if bi == target_idx:
                            vb = b; break
                elif args.qv1_cycle:
                    try:
                        vb = next(qv1_iter)
                    except StopIteration:
                        qv1_iter = iter(va1)
                        vb = next(qv1_iter)
                else:
                    vb = tiny_batch_from(va1)
                vb = slice_first_k(vb, args.qv1_k, modality)

                do_full = (args.qv1_full_every > 0 and step % args.qv1_full_every == 0)

                was_train = model.training
                model.eval()

                if do_full:
                    # aggregate all QV1
                    opt.zero_grad(set_to_none=True)
                    total_batches = 0
                    for bfull in va1:
                        _ = backward_on_batch(model, bfull, args.device, modality)
                        gva.cache_val_grad(groups)
                        opt.zero_grad(set_to_none=True)
                        total_batches += 1
                    print(f"[QV1 step {step}] FULL aggregation over {total_batches} batches.")
                    qv1_full_flag = 1
                else:
                    # quick one-batch val
                    opt.zero_grad(set_to_none=True)
                    _ = backward_on_batch(model, vb, args.device, modality)
                    gva.cache_val_grad(groups)

                # collect val/train grads snapshots
                # gsrc_val = collect_group_flat_grads(groups)
                # opt.zero_grad(set_to_none=True)
                # if modality == "vision":
                #     logits2 = model(xb)
                #     loss2 = F.cross_entropy(logits2, yb)
                # # else:
                # #     logits2 = model(input_ids=input_ids, attention_mask=attn)
                # #     loss2 = F.cross_entropy(logits2, labels)
                # else:
                #     input_ids2, attn2, labels2 = _split_text_batch(batch, args.device)
                #     logits2 = model(input_ids=input_ids2, attention_mask=attn2)
                #     loss2 = F.cross_entropy(logits2, labels2)

                # loss2.backward()
                # gsrc_tr = collect_group_flat_grads(groups)

                # collect VAL grads snapshot (from the quick/full val backward above)
                gsrc_val = collect_group_flat_grads(groups)

                # === 清除 val 梯度，再恢复“已加权”的 TRAIN 梯度 ===
                opt.zero_grad(set_to_none=True)
                for p, g in zip(model.parameters(), _saved_tr_grads):
                    if g is None:
                        p.grad = None
                    else:
                        # 赋予一份独立张量，避免与缓存共享存储
                        p.grad = g.clone()

                # 现在 .grad 就是带掩码的 train 梯度；据此收集 gsrc_tr
                gsrc_tr = collect_group_flat_grads(groups)
                # spectral mismatch
                L_spec, scale = spec.compute(groups, gsrc_tr, gsrc_val)
                lspec_val = L_spec

                # ---- estimate mean cosine BEFORE projection (for meta)
                cos_vals = []
                for vt, vv in zip(gsrc_tr, gsrc_val):
                    if vt.numel()==0 or vv.numel()==0:
                        continue
                    c = torch.nn.functional.cosine_similarity(vt, vv, dim=0).clamp(-1.0, 1.0)
                    cos_vals.append(c)
                cos_est = float(torch.stack(cos_vals).mean().item()) if len(cos_vals)>0 else 0.0

                # Fast loop: L_spec -> tau  (with cos-meta + magnitude fusion)
                if args.tau_from_spec and args.noise_prob:
                    # EMA(L_spec)
                    if lspec_ema is None:
                        lspec_ema = L_spec
                    else:
                        lspec_ema = args.lspec_ema_beta * lspec_ema + (1 - args.lspec_ema_beta) * L_spec

                    delta = L_spec - lspec_ema

                    # direction
                    dir_sign = 1.0 if delta > 0 else -1.0
                    if args.use_cos_meta:
                        dir_sign *= (1.0 if cos_est >= 0.0 else -1.0)
                    if args.qv2_mode == "dir_gate":
                        dir_sign *= (1.0 if qv2_dir >= 0 else -1.0)

                    # magnitude fusion
                    lspec_mag = float(np.tanh(abs(delta) / max(1e-6, args.tau_lspec_scale)))
                    cos_mag   = float((abs(cos_est)) ** args.tau_cos_pow)
                    if args.tau_mag_mode == "lspec":
                        mag = lspec_mag
                    elif args.tau_mag_mode == "cos":
                        mag = cos_mag
                    else:
                        mag = lspec_mag * cos_mag

                    step_tau = args.tau_step0 * meta_gain * mag
                    tau = float(np.clip(tau * np.exp(dir_sign * step_tau), 0.2, 2.0))
                    last_tau_update_dir = 1 if dir_sign > 0 else -1

                    print(f"[τ-by-Lspec+cos step {step}] L={L_spec:.2f} ema={lspec_ema:.2f} Δ={delta:+.2f} "
                          f"cos={cos_est:+.3f} dir={'+' if dir_sign>0 else '-'} mag={mag:.3f} "
                          f"meta_gain={meta_gain:.3f} -> tau={tau:.4f}")

                # diag for extremely small L_spec
                if (L_spec is not None) and (L_spec < 1e-12):
                    flat_tr = torch.cat([g.flatten() for g in gsrc_tr if g.numel() > 0])
                    flat_v  = torch.cat([g.flatten() for g in gsrc_val if g.numel() > 0])
                    print(f"[Diag] L_spec≈0. var_tr={float(flat_tr.var().item()):.3e} var_val={float(flat_v.var().item()):.3e} "
                          f"groups={len(gsrc_tr)} m={args.spec_m}")

                # projection strength scheduling
                eta_eff = args.align_eta
                if args.spec_scale_gva:
                    eta_eff = eta_eff * (0.5 + 0.5 * scale)
                if args.qv2_proj:
                    eta_eff = eta_eff * qv2_gate
                if args.proj_cos_scale:
                    cos_list = []
                    for vt, vv in zip(gsrc_tr, gsrc_val):
                        if vt.numel()==0 or vv.numel()==0:
                            continue
                        c = torch.nn.functional.cosine_similarity(vt, vv, dim=0).clamp(-1.0, 1.0)
                        cos_list.append(c)
                    if len(cos_list) > 0:
                        cos_est_proj = float(torch.stack(cos_list).mean().item())
                        eta_eff = eta_eff * max(0.0, cos_est_proj)

                gva.eta_proj = eta_eff
                gva_stats = gva.project_train_grads(groups)

                # pretty print once
                if do_full:
                    cm = gva_stats.get('cos_mean')
                    msg = f"[QV1 step {step}] FULL GVA proj={gva_stats.get('num_proj',0)} "
                    msg += f"cos_mean={cm:.4f} " if cm is not None else "cos_mean=None "
                    msg += f"eta_eff={eta_eff:.4f} L_spec={L_spec:.6f}"
                    print(msg)
                elif gva_stats.get("num_proj", 0) > 0 and not printed_once_proj:
                    cm = gva_stats.get('cos_mean')
                    msg = f"[QV1 step {step}] GVA projected {gva_stats['num_proj']} groups; "
                    msg += f"cos_mean={cm:.4f} " if cm is not None else "cos_mean=None "
                    msg += f"eta_eff={eta_eff:.4f} L_spec={L_spec:.6f}"
                    print(msg)
                    printed_once_proj = True

                model.train(mode=was_train)

                # === Adam gate（插拔）：用同一 mag/dir_sign，门控 lr 或 weight_decay ===
                if args.adam_gate != "off":
                    gate_raw = mag if 'mag' in locals() else 0.0
                    adam_gate_ema = args.adam_gate_beta * adam_gate_ema + (1 - args.adam_gate_beta) * gate_raw

                    scale_up   = (1.0 + args.adam_gate_k * adam_gate_ema)
                    scale_down = 1.0 / max(1e-6, scale_up)
                    apply_scale = scale_up if ('dir_sign' in locals() and dir_sign > 0) else scale_down

                    for gi, pg in enumerate(opt.param_groups):
                        if args.adam_gate == "wd":
                            pg["weight_decay"] = base_wds[gi] * apply_scale
                        elif args.adam_gate == "lr":
                            # 几何差(dir>0)->lr降；几何好(dir<0)->lr升
                            pg["lr"] = base_lrs[gi] * (1.0 / apply_scale)

            # ---- Optimizer step
            opt.step()

            # ---- 速度/显存/学习率 打印 ----
            speed_count += 1
            if args.speed_every > 0 and (speed_count % args.speed_every == 0):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                now = time.time()
                dt = now - t_speed
                its = args.speed_every / max(1e-9, dt)
                mem_alloc = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
                mem_rsvd  = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0
                cur_lr = opt.param_groups[0]["lr"]
                print(f"[Speed] step={step} | {its:.1f} it/s | lr={cur_lr:.2e} | "
                      f"cuda_mem={mem_alloc:.0f}MiB (reserved {mem_rsvd:.0f}MiB)")
                t_speed = now

            # ---- QV2: meta policy ----
            qv2_acc = None
            if (step % max(1, args.qv2_every) == 0):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_qv2 = time.time()

                model.eval()
                v2_loss, v2_acc = (run_epoch_vision(model, va2, args.device, opt=None)
                                   if modality=="vision" else run_epoch_text(model, va2, args.device, opt=None))
                qv2_acc = v2_acc

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt_qv2 = time.time() - t_qv2
                print(f"[QV2] step={step} | eval_time={dt_qv2:.2f}s | acc={v2_acc:.4f}")

                if last_qv2_acc is not None:
                    delta_acc = v2_acc - last_qv2_acc
                    # meta gain
                    if delta_acc > 0:
                        meta_gain = min(5.0, meta_gain * args.meta_up)
                    else:
                        meta_gain = max(0.2, meta_gain * args.meta_down)

                    # policy branch
                    if args.qv2_mode == "shrink":
                        pass
                    elif args.qv2_mode == "dir_gate":
                        qv2_dir = +1 if delta_acc > 0 else -1
                    elif args.qv2_mode == "instant_reverse":
                        if args.tau_from_spec and args.noise_prob:
                            rev = - last_tau_update_dir * float(args.qv2_meta_eta)
                            tau = float(np.clip(tau * np.exp(rev), 0.2, 2.0))

                    # optional: gate projection
                    if args.qv2_proj:
                        if delta_acc > 0:
                            qv2_gate = args.qv2_proj_beta * qv2_gate + (1 - args.qv2_proj_beta) * min(2.0, qv2_gate * args.qv2_proj_up)
                        else:
                            qv2_gate = args.qv2_proj_beta * qv2_gate + (1 - args.qv2_proj_beta) * max(0.0, qv2_gate * args.qv2_proj_down)

                    msg = (f"[QV2 step {step}] acc={v2_acc:.4f} Δ={delta_acc:+.4f} "
                           f"mode={args.qv2_mode} -> meta_gain={meta_gain:.3f}")
                    if args.qv2_proj:
                        msg += f" (proj_gate={qv2_gate:.3f})"
                    if args.qv2_mode == "dir_gate":
                        msg += f" (qv2_dir={qv2_dir:+d})"
                    if args.qv2_mode == "instant_reverse":
                        msg += f" (tau={tau:.4f})"
                    print(msg)
                else:
                    print(f"[QV2 step {step}] acc={v2_acc:.4f} (init) meta_gain={meta_gain:.3f}")
                last_qv2_acc = v2_acc
                model.train()

            # ---- Log train row（降频）----
            do_log = (args.log_every <= 0) or (step % args.log_every == 0)
            if do_log:
                csv.log({
                    "step": step, "epoch": epoch, "split": "train",
                    "loss": float(loss.item()), "acc": train_acc,
                    "gva_cos": gva_stats.get("cos_mean"),
                    "gva_numproj": gva_stats.get("num_proj"),
                    "lspec": lspec_val,
                    "eta_eff": eta_eff,
                    "qv2_acc": qv2_acc,
                    "tau": tau if args.noise_prob else None,
                    "qv1_full": int(qv1_full_flag),
                    "meta_gain": meta_gain,
                    "qv2_gate": qv2_gate if args.qv2_proj else None,
                    "qv2_dir": qv2_dir if args.qv2_mode=="dir_gate" else None,
                    "cos_est": float(cos_est),
                    "adam_gate_ema": float(adam_gate_ema),
                })
                swan.log({
                    "train/loss": float(loss.item()), "train/acc": train_acc,
                    "gva/cos": gva_stats.get("cos_mean") if gva_stats.get("cos_mean") is not None else -1.0,
                    "gva/numproj": gva_stats.get("num_proj", 0),
                    "gva/eta_eff": eta_eff,
                    "spec/lspec": lspec_val if lspec_val is not None else -1.0,
                    "qv2/acc": qv2_acc if qv2_acc is not None else -1.0,
                    "noise/tau": tau if args.noise_prob else -1.0,
                    "qv1/full": int(qv1_full_flag),
                    "meta/gain": meta_gain,
                    "proj/gate": qv2_gate if args.qv2_proj else -1.0,
                    "qv2/dir": qv2_dir if args.qv2_mode=="dir_gate" else 0.0,
                    "meta/cos_est": float(cos_est),
                    "meta/adam_gate_ema": float(adam_gate_ema),
                }, step=step)

        # ---- epoch-end: eval on QV2 and TEST ----
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_val = time.time()

        model.eval()
        vloss, vacc = (run_epoch_vision(model, va2, args.device, opt=None)
                       if modality=="vision" else run_epoch_text(model, va2, args.device, opt=None))
        tloss, tacc = (run_epoch_vision(model, te, args.device, opt=None)
                       if modality=="vision" else run_epoch_text(model, te, args.device, opt=None))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[Epoch {epoch}] QV2: loss={vloss:.4f}, acc={vacc:.4f} | TEST: loss={tloss:.4f}, acc={tacc:.4f} "
              f"| epoch_eval_time={time.time()-t_val:.2f}s")

        csv.log({"step": step, "epoch": epoch, "split": "val", "loss": vloss, "acc": vacc,
                 "gva_cos": None, "gva_numproj": None, "lspec": None, "eta_eff": None,
                 "qv2_acc": None, "tau": tau if args.noise_prob else None, "qv1_full": None,
                 "meta_gain": meta_gain, "qv2_gate": qv2_gate if args.qv2_proj else None,
                 "qv2_dir": qv2_dir if args.qv2_mode=='dir_gate' else None})
        csv.log({"step": step, "epoch": epoch, "split": "test", "loss": tloss, "acc": tacc,
                 "gva_cos": None, "gva_numproj": None, "lspec": None, "eta_eff": None,
                 "qv2_acc": None, "tau": tau if args.noise_prob else None, "qv1_full": None,
                 "meta_gain": meta_gain, "qv2_gate": qv2_gate if args.qv2_proj else None,
                 "qv2_dir": qv2_dir if args.qv2_mode=='dir_gate' else None})
        swan.log({"val/loss": vloss, "val/acc": vacc, "test/loss": tloss, "test/acc": tacc}, step=step)

        # ---- Save best by validation; if later test gets better, override best ----
        override = False
        if vacc > best_val_acc:
            best_val_acc = vacc
            override = True
        elif tacc > best_test_acc:
            override = True

        if override:
            best_test_acc = max(best_test_acc, tacc)
            _save_model(model, best_model_path)
            _copy_csv_as(csv_path, best_csv_path)
            print(f"[BEST] saved -> {best_model_path} & {best_csv_path} | "
                  f"best_val={best_val_acc:.4f}, best_test={best_test_acc:.4f}")

    csv.close()
    swan.finish()

if __name__ == "__main__":
    main()

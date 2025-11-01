import os, argparse, time, math, copy, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from simple_logger import CSVLogger
from meta_mask import BucketMetaMask
from spectral_utils import collect_flat_grads, cosine_similarity_flat, CountSketchLspec

# ------------------- Data -------------------
def get_sst2(batch_size=32, model_name="distilbert-base-uncased", seed=42):
    ds = load_dataset("glue", "sst2")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def enc(ex):
        out = tok(ex["sentence"], truncation=True, padding="max_length", max_length=128)
        out["labels"] = ex["label"]
        return out

    ds_enc = ds.map(enc, batched=True, remove_columns=ds["train"].column_names)
    ds_enc.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    # 官方 validation 当作 test；train 再切出 qv1/qv2
    rng = np.random.RandomState(seed)
    n = len(ds_enc["train"])
    idx = np.arange(n); rng.shuffle(idx)
    v1 = int(round(n * 0.05))
    v2 = int(round(n * 0.05))
    val1_idx = idx[:v1]; val2_idx = idx[v1:v1+v2]; tr_idx = idx[v1+v2:]

    tr_ds  = ds_enc["train"].select(tr_idx.tolist())
    va1_ds = ds_enc["train"].select(val1_idx.tolist())
    va2_ds = ds_enc["train"].select(val2_idx.tolist())
    te_ds  = ds_enc["validation"]

    g = torch.Generator(); g.manual_seed(seed)
    def _dl(ds, shuffle, bs):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, pin_memory=True, generator=g)

    return _dl(tr_ds, True, batch_size), _dl(va1_ds, False, batch_size), _dl(va2_ds, False, batch_size), _dl(te_ds, False, batch_size), tok

def batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

# ------------------- Scoring S -------------------
@torch.no_grad()
def compute_S(logits, labels, mode="loss", beta_conf=0.5):
    """
    返回标准化评分 S：
    - "loss": per-sample CE
    - "loss+conf": z(loss) + beta_conf * z(1 - p_true)
    """
    losses = F.cross_entropy(logits, labels, reduction='none')
    if mode == "loss":
        S = (losses - losses.mean()) / (losses.std(unbiased=False) + 1e-8)
        return S, {"loss_mean": float(losses.mean().item())}
    else:
        prob = logits.softmax(dim=-1)
        conf = prob.gather(1, labels.view(-1,1)).squeeze(1)
        geo  = 1.0 - conf
        z1   = (losses - losses.mean()) / (losses.std(unbiased=False) + 1e-8)
        z2   = (geo    - geo.mean())    / (geo.std(unbiased=False)    + 1e-8)
        S = z1 + beta_conf * z2
        return S, {"loss_mean": float(losses.mean().item()), "conf_mean": float(conf.mean().item())}

# ------------------- Eval helpers -------------------
@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for batch in loader:
        b = batch_to_device(batch, device)
        logits = model(**b).logits
        loss = F.cross_entropy(logits, b["labels"])
        pred = logits.argmax(-1)
        acc = (pred == b["labels"]).float().mean().item()
        bs = b["labels"].size(0)
        tot_loss += loss.item() * bs
        tot_acc  += acc * bs
        n += bs
    return tot_loss/n, tot_acc/n

# ------------------- Virtual step (for tournament on Val2) -------------------
def virtual_train_step(model, train_batch, device, mask_ctrl: BucketMetaMask, s_mode="loss", beta_conf=0.5, grad_scale=1.0):
    """
    对一个 train_batch 执行一次“虚拟”训练步（只改模型参数，忽略优化器状态），返回更新前后的参数快照以便回滚。
    """
    # 保存参数快照
    with torch.no_grad():
        snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.zero_grad(set_to_none=True)
    b = batch_to_device(train_batch, device)
    out = model(**b)
    logits = out.logits
    S, _ = compute_S(logits, b["labels"], mode=s_mode, beta_conf=beta_conf)
    w, _ = mask_ctrl.weights(S)
    losses = F.cross_entropy(logits, b["labels"], reduction='none')
    loss = (w * losses).mean()
    loss.backward()

    # 采用简单 SGD 步（grad_scale 相当于学习率）
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None: continue
            p.add_( - grad_scale * p.grad )

    return snapshot

@torch.no_grad()
def restore_snapshot(model, snapshot: dict):
    sd = model.state_dict()
    for k, v in snapshot.items():
        sd[k].copy_(v)

# ------------------- Main -------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model_name", default="distilbert-base-uncased", help="HF model id, e.g., distilbert-base-uncased / prajjwal1/bert-tiny")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)  # 注意：SST2 有效 batch 会受长度影响
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=2025)

    # 快/慢环
    p.add_argument("--qv1_every", type=int, default=300, help="多少个训练 step 进行一次快环（Val1）")
    p.add_argument("--qv2_every", type=int, default=1200, help="多少个训练 step 进行一次慢环（Val2 锦标赛）")
    p.add_argument("--cand_per_qv1", type=int, default=2, help="每次快环生成多少个候选扰动（推荐2~4）")
    p.add_argument("--perturb_eps", type=float, default=0.2, help="每次对某个桶的 alpha 的改变幅度")
    p.add_argument("--tourney_max", type=int, default=6, help="最多累计多少个候选进入锦标赛窗口")
    p.add_argument("--virt_lr", type=float, default=5e-4, help="锦标赛评估时的虚拟一步学习率")

    # 掩码控制器
    p.add_argument("--buckets", type=int, default=4)
    p.add_argument("--alpha0", type=float, default=1.0)
    p.add_argument("--score_mode", choices=["loss","loss+conf"], default="loss+conf")
    p.add_argument("--beta_conf", type=float, default=0.5)

    # 奖励/正则
    p.add_argument("--lambda_spec", type=float, default=0.0, help="对 L_spec 的惩罚系数（快环）")
    p.add_argument("--lambda_budget", type=float, default=0.0, help="对 keep_rate 偏离的惩罚系数（快环）")
    p.add_argument("--target_keep", type=float, default=0.7, help="期望的平均保留率，用于预算约束")

    # 日志
    p.add_argument("--logdir", type=str, default="logs/d2mm_minimal")
    p.add_argument("--run_name", type=str, default=None)

    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = args.device
    tr, va1, va2, te, tok = get_sst2(batch_size=max(16, args.batch//4), model_name=args.model_name, seed=args.seed)

    # model & opt
    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(tr)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    # controllers
    mask_ctrl = BucketMetaMask(K=args.buckets, init_alpha=args.alpha0, device=device)
    lspec = CountSketchLspec(m=64)

    # logging
    name = args.run_name or f"sst2_{args.model_name.replace('/','-')}"
    csv = CSVLogger(os.path.join(args.logdir, f"{name}.csv"),
                    fieldnames=["step","epoch","split","loss","acc","keep_rate","cos_v1","lspec_v1","alpha"])

    # helpers
    def train_step(batch):
        model.train()
        opt.zero_grad(set_to_none=True)
        b = batch_to_device(batch, device)
        out = model(**b)
        logits = out.logits
        S, _ = compute_S(logits, b["labels"], args.score_mode, args.beta_conf)
        w, stats = mask_ctrl.weights(S)
        losses = F.cross_entropy(logits, b["labels"], reduction='none')
        loss = (w * losses).mean()
        loss.backward()
        opt.step(); sched.step()
        with torch.no_grad():
            acc = (logits.argmax(-1) == b["labels"]).float().mean().item()
        return float(loss.item()), float(acc), float(stats["keep_rate"])

    # candidate window for tournament
    cand_bank = []   # list of dict: {"alpha": tensor, "r_fast": float}
    last_qv2_acc = None

    step = 0
    for epoch in range(1, args.epochs+1):
        for batch in tr:
            step += 1
            loss, acc, keep = train_step(batch)
            if step % 100 == 0:
                csv.log({"step":step,"epoch":epoch,"split":"train","loss":loss,"acc":acc,"keep_rate":keep,"alpha":mask_ctrl.alpha.detach().cpu().tolist()})

            # ---------- QV1: 快环生成候选 & 快速奖励 ----------
            if (step % args.qv1_every) == 0:
                # 1) 取一个 Val1 批算 g_val1
                try:
                    vb = next(_va1_iter)
                except Exception:
                    _va1_iter = iter(va1)
                    vb = next(_va1_iter)
                model.eval()
                model.zero_grad(set_to_none=True)
                vbd = batch_to_device(vb, device)
                vout = model(**vbd); vloss = F.cross_entropy(vout.logits, vbd["labels"]); vloss.backward()
                g_val = collect_flat_grads(model)
                model.zero_grad(set_to_none=True)

                # 2) 生成若干扰动（坐标方向），评估在当前 train 批上的对齐奖励
                deltas = mask_ctrl.sample_perturbations(num=args.cand_per_qv1, eps=args.perturb_eps, symmetric=True)
                base_alpha = mask_ctrl.clone_params()

                for d in deltas:
                    # baseline grads with current alpha
                    model.train()
                    model.zero_grad(set_to_none=True)
                    btr = batch_to_device(batch, device)
                    outA = model(**btr); logitsA = outA.logits
                    SA, _ = compute_S(logitsA, btr["labels"], args.score_mode, args.beta_conf)
                    wA, _ = mask_ctrl.weights(SA)
                    lossA = (F.cross_entropy(logitsA, btr["labels"], reduction='none') * wA).mean()
                    lossA.backward()
                    gA = collect_flat_grads(model)
                    model.zero_grad(set_to_none=True)

                    # candidate grads with alpha+delta
                    mask_ctrl.apply_delta(d)
                    outB = model(**btr); logitsB = outB.logits
                    SB, _ = compute_S(logitsB, btr["labels"], args.score_mode, args.beta_conf)
                    wB, stB = mask_ctrl.weights(SB)
                    lossB = (F.cross_entropy(logitsB, btr["labels"], reduction='none') * wB).mean()
                    lossB.backward()
                    gB = collect_flat_grads(model)
                    model.zero_grad(set_to_none=True)

                    # 奖励：对齐 + （可选）谱惩罚 + 预算惩罚
                    cosA = cosine_similarity_flat(gA, g_val)
                    cosB = cosine_similarity_flat(gB, g_val)
                    lspecA = lspec.lspec(gA, g_val); lspecB = lspec.lspec(gB, g_val)
                    R_A = cosA - args.lambda_spec*lspecA - args.lambda_budget*abs(wA.mean().item() - args.target_keep)
                    R_B = cosB - args.lambda_spec*lspecB - args.lambda_budget*abs(wB.mean().item() - args.target_keep)

                    # 只把“更好的那个候选”存入窗口（去重/去弱）
                    better = {"alpha": mask_ctrl.clone_params(), "r_fast": R_B, "keep": float(wB.mean().item())}
                    worse  = {"alpha": base_alpha.clone(),          "r_fast": R_A, "keep": float(wA.mean().item())}
                    chosen = better if R_B > R_A else worse
                    cand_bank.append(chosen)

                    # 恢复 alpha
                    mask_ctrl.set_params(base_alpha)

                # 限制窗口大小
                if len(cand_bank) > args.tourney_max:
                    cand_bank = sorted(cand_bank, key=lambda x: x["r_fast"], reverse=True)[:args.tourney_max]

            # ---------- QV2: 慢环锦标赛（虚拟一步 + Val2 评估） ----------
            if (step % args.qv2_every) == 0 and len(cand_bank) > 0:
                # 存下当前 alpha 作为 baseline 参与比赛
                cand_bank.append({"alpha": mask_ctrl.clone_params(), "r_fast": -1e9, "keep": None})

                # 取一个小的 train 批（复用当前）与一个小的 Val2 批
                try:
                    vb2 = next(_va2_iter)
                except Exception:
                    _va2_iter = iter(va2)
                    vb2 = next(_va2_iter)

                scores = []
                for c in cand_bank:
                    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    # 应用候选 alpha & 做一次虚拟训练步
                    mask_ctrl.set_params(c["alpha"])
                    snap = virtual_train_step(model, batch, device, mask_ctrl, s_mode=args.score_mode, beta_conf=args.beta_conf, grad_scale=args.virt_lr)
                    # 在 Val2 上评估
                    _, acc2 = eval_loader(model, [vb2], device)
                    scores.append((acc2, c))
                    # 回滚到原模型 & 恢复模型状态
                    restore_snapshot(model, snapshot)

                # 选 Val2 得分最高的候选为新基线 alpha
                scores.sort(key=lambda x: x[0], reverse=True)
                best_acc2, best_c = scores[0]
                mask_ctrl.set_params(best_c["alpha"])
                cand_bank = []  # 清空窗口

                # 记录一次 Val2 的真实评估（更稳定）
                vloss, vacc = eval_loader(model, va2, device)
                csv.log({"step":step,"epoch":epoch,"split":"val2","loss":vloss,"acc":vacc,
                         "keep_rate":best_c["keep"] if best_c["keep"] is not None else None,
                         "alpha":mask_ctrl.alpha.detach().cpu().tolist()})
                last_qv2_acc = vacc

        # epoch end: test
        tloss, tacc = eval_loader(model, te, device)
        csv.log({"step":step,"epoch":epoch,"split":"test","loss":tloss,"acc":tacc,
                 "keep_rate":None,"alpha":mask_ctrl.alpha.detach().cpu().tolist()})
        print(f"[Epoch {epoch}] TEST acc={tacc:.4f} loss={tloss:.4f} | alpha={mask_ctrl.alpha.detach().cpu().numpy()}")

    csv.close()

if __name__ == "__main__":
    main()

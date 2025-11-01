#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_metagraph.py (text path supports CP/Betti; vision path unchanged)
- 文本(SST-2)现在也能在不改模型文件的前提下启用 CP/Betti：
  * DistilBERT: 用 backbone last_hidden_state 做句向量，再走 head
  * SmallBERT: 用 tok+pos -> blocks -> ln 做句向量，再走 textcls_head
  * 当传 --mg_use_cp / --mg_use_betti 且触发器满足时，对句向量进行软掩/缩放
- 视觉(C10/C100/MNIST)路径保持原状；SwanLab & CSV 日志保持一致
"""
import os, time, csv, random, argparse, math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# swanlab (optional)
try:
    import swanlab
    _SWAN_OK = True
except Exception:
    _SWAN_OK = False

from utils.registry import MODEL_REGISTRY
# ensure registry
from models.small_cnn import SmallCNN          # noqa
from models.small_transformer import SmallTransformer  # noqa
from models.small_bert import SmallBERT        # noqa
from models.distillbert_hf import DistilBertHF # noqa
from models.distillbert_hf_gate import DistillBertHFGate  # noqa
from models.distillbert_metagraph import DistilBertMetaGraph  # noqa

# MetaGraph（视觉用）
from modules.metagraph import MetaGraph, _BettiSketchLite, _ChangePointLite

# vision deps
import torchvision
import torchvision.transforms as T

# text deps
try:
    from datasets import load_dataset
    _HF_OK = True
except Exception:
    _HF_OK = False

def set_seed(seed: int):
    random.seed(seed); os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# --------------------
# Datasets
# --------------------
def get_vision_dataset(name: str, train: bool, root: str="./data"):
    name=name.lower()
    tf_train=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    tf_test=T.Compose([T.ToTensor()])
    if name=="c10":
        ds=torchvision.datasets.CIFAR10(root=root, train=train, download=True,
                                        transform=tf_train if train else tf_test); num=10; ch=3
    elif name=="c100":
        ds=torchvision.datasets.CIFAR100(root=root, train=train, download=True,
                                         transform=tf_train if train else tf_test); num=100; ch=3
    elif name=="mnist":
        ds=torchvision.datasets.MNIST(root=root, train=train, download=True,
                                      transform=tf_test if not train else T.Compose([T.RandomCrop(28,padding=2),T.ToTensor()])); num=10; ch=1
    else:
        raise ValueError(f"Unknown vision dataset: {name}")
    return ds, num, ch

def get_sst2(split: str="train"):
    assert _HF_OK, "pip install datasets"
    return load_dataset("glue","sst2", split=split)

def collate_vision(batch):
    xs, ys = zip(*batch); return torch.stack(xs,0), torch.tensor(ys, dtype=torch.long)

class SST2Pipe:
    def __init__(self, model_name="distilbert-base-uncased", max_len=128):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
    def encode_batch(self, batch):
        texts=[ex["sentence"] for ex in batch]
        lab=torch.tensor([int(ex["label"]) for ex in batch], dtype=torch.long)
        enc=self.tok(texts, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"], lab, enc["attention_mask"]

# --------------------
# Text feature utils (不改模型文件，直接用内部结构得到句向量)
# --------------------
def text_features_and_head(model: nn.Module, input_ids: torch.Tensor, attn: torch.Tensor
                           ) -> Tuple[torch.Tensor, nn.Module]:
    """
    返回： (句向量 [B,H], 分类头 callable)
    支持：DistilBertHF / DistilBertHFGate / DistilBertMetaGraph / SmallBERT
    """
    # DistilBERT 家族
    if hasattr(model, "bert") and hasattr(model, "head"):
        out = model.bert(input_ids=input_ids, attention_mask=attn)
        last = out.last_hidden_state              # [B,T,H]
        mask = attn.unsqueeze(-1).float()         # [B,T,1]
        feat = (last * mask).sum(1) / mask.sum(1).clamp_min(1.0)  # [B,H]
        head = model.head                         # nn.Linear(H, C)
        return feat, head

    # SmallBERT 文本路径（见你的 small_bert 实现）
    if isinstance(model, SmallBERT) and (getattr(model, "text_task", None) == "cls"):
        ids = input_ids
        B, T = ids.shape
        pos_ids = torch.arange(T, device=ids.device).unsqueeze(0).expand(B,T)
        tok = model.tok(ids) + model.text_pos(pos_ids)     # [B,T,D]
        # blocks（每层都是 1D BERTBlock）
        h = tok
        for b in model.blocks:
            h = b(h, attn_mask=(ids==getattr(model,'pad_id',0)))
        h = model.ln(h)
        # masked mean
        padmask = (ids == getattr(model,'pad_id',0))
        if padmask.any():
            lens = (~padmask).sum(1).clamp_min(1).unsqueeze(-1)
            h = h.masked_fill(padmask.unsqueeze(-1), 0.0)
            feat = h.sum(1)/lens
        else:
            feat = h.mean(1)
        head = model.textcls_head
        if head is None:
            raise RuntimeError("SmallBERT(text) missing textcls_head")
        return feat, head

    # 回退：直接走模型前向获取 logits，再反推 head 不安全；此时只返回 None
    raise RuntimeError("Unsupported text model for CP/Betti without modifying model.")

# --------------------
# CP/Betti (文本通道) —— 与之前你用的思路一致：对句向量做软掩/缩放
# --------------------
class CPChannelMaskText:
    def __init__(self, dim: int, k: float = 4.0, eps: float = 1e-6):
        self.mu = None
        self.var = None
        self.k = k
        self.eps = eps
        self.dim = dim
    def __call__(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B,H]
        bmean = feat.mean(0).detach()  # [H]
        if self.mu is None:
            self.mu = bmean.clone()
            self.var = torch.ones_like(bmean)
        else:
            self.mu = 0.9*self.mu + 0.1*bmean
            self.var = 0.9*self.var + 0.1*(bmean - self.mu).pow(2)
        noise = (bmean - self.mu).abs() / (self.var.sqrt()+self.eps)  # [H]
        ns_mean = noise.mean()
        mask_c = torch.sigmoid(self.k * (ns_mean - noise))            # [H]
        return feat * mask_c.unsqueeze(0)

class BettiScalerText:
    def __init__(self, in_dim: int, gamma: float = 0.05, max_drop: float = 0.2, device="cpu"):
        self.gamma = gamma
        self.max_drop = max_drop
        self._ema_b1: Optional[float] = None
        self.betti = _BettiSketchLite(in_dim=in_dim).to(device)
    def __call__(self, feat: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        # feat: [B,H]  -> betti on normalized/projected
        with torch.no_grad():
            b = self.betti(feat)  # [2] -> (b0,b1)
            b0, b1 = float(b[0].item()), float(b[1].item())
            if self._ema_b1 is None:
                self._ema_b1 = b1
            else:
                self._ema_b1 = 0.9*self._ema_b1 + 0.1*b1
            gap = max(0.0, b1 - self._ema_b1)
            scale = 1.0 - min(self.max_drop, self.gamma * math.tanh(gap))
        return feat * scale, b0, b1

# --------------------
# Metrics
# --------------------
def compute_acc(logits: torch.Tensor, y: torch.Tensor)->float:
    return (logits.argmax(-1)==y).float().mean().item()

def normalize_logits(out):
    if isinstance(out, dict) and "logits" in out: return out["logits"]
    if hasattr(out, "logits"): return out.logits
    return out

# --------------------
# Train/Eval epochs
# --------------------
def run_epoch_vision(model, mg: MetaGraph, loader, device, optimizer=None,
                     trigger_on: bool=True):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    tot_loss=tot_acc=0.0; n=0; b0s=b1s=0.0; nb=0
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        adv_x = mg.maybe_adv(model, x, y) if trigger_on else None
        loss, logits, extras = mg.forward_loss(model, x, y, adv_x=adv_x)
        if is_train:
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            mg.on_batch_end(model, x, y, logits, loss)
        with torch.no_grad():
            bs = x.size(0)
            tot_loss += float(loss.item())*bs; tot_acc += compute_acc(logits,y)*bs; n+=bs
            if "betti" in extras and isinstance(extras["betti"], torch.Tensor):
                b0s += float(extras["betti"][0].item()); b1s += float(extras["betti"][1].item()); nb+=1
    return {"loss":tot_loss/max(1,n), "acc":tot_acc/max(1,n), "b0":b0s/max(1,nb), "b1":b1s/max(1,nb)}

def run_epoch_text(model, loader, device, optimizer=None,
                   use_cp: bool=False, use_betti: bool=False,
                   trigger_on: bool=False,
                   cp_state: Optional[CPChannelMaskText]=None,
                   betti_state: Optional[BettiScalerText]=None):
    """
    文本专用：当 use_cp/use_betti 且 trigger_on 时，对句向量做软掩/缩放，再送入 head。
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    tot_loss=tot_acc=0.0; n=0; b0s=b1s=0.0; nb=0

    for input_ids, y, attn in loader:
        input_ids = input_ids.to(device); y = y.to(device); attn = attn.to(device)

        # 仅当开启且触发时，走“特征→降噪→head”路径；否则直接 model 前向
        if trigger_on and (use_cp or use_betti):
            feat, head = text_features_and_head(model, input_ids, attn)   # [B,H] + head
            # 初始化 CP/Betti 状态
            if use_cp and (cp_state is None or cp_state.dim != feat.size(1) if hasattr(cp_state,'dim') else False):
                cp_state = CPChannelMaskText(dim=feat.size(1)); cp_state.dim=feat.size(1)
            if use_betti and (betti_state is None or betti_state.betti.proj[0].weight.size(1) != feat.size(1)):
                betti_state = BettiScalerText(in_dim=feat.size(1), device=feat.device)

            # 先 CP 后 Betti（顺序和你之前一致的思路）
            if use_cp and cp_state is not None:
                feat = cp_state(feat)
            if use_betti and betti_state is not None:
                feat, b0, b1 = betti_state(feat)
                b0s += b0; b1s += b1; nb += 1

            logits = head(feat)
        else:
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = normalize_logits(out)

        loss = F.cross_entropy(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

        with torch.no_grad():
            bs = input_ids.size(0)
            tot_loss += float(loss.item())*bs
            tot_acc  += compute_acc(logits,y)*bs
            n += bs

    return {"loss":tot_loss/max(1,n), "acc":tot_acc/max(1,n), "b0":b0s/max(1,nb), "b1":b1s/max(1,nb),
            "cp_state": cp_state, "betti_state": betti_state}

# --------------------
# Main
# --------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", choices=["vision","textcls"], default="vision")
    ap.add_argument("--dataset", choices=["c10","c100","mnist","sst2"], default="c10")
    ap.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="small_cnn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight_lr", type=float, default=3e-4)
    ap.add_argument("--adam_eps", type=float, default=1e-8)
    ap.add_argument("--adam_wd", type=float, default=0.01)

    # MetaGraph/触发（沿用你已有参数，不新增 mask_mode）
    ap.add_argument("--mg_use_betti", action="store_true")
    ap.add_argument("--mg_use_cp", action="store_true")
    ap.add_argument("--mg_use_coupling", action="store_true")
    ap.add_argument("--mg_use_curr", action="store_true")
    ap.add_argument("--mg_adv_eps", type=float, default=0.1)
    ap.add_argument("--trigger_acc", type=float, default=0.90)
    ap.add_argument("--trigger_warmup_epochs", type=int, default=1)

    # Logging
    ap.add_argument("--log_dir", type=str, default="optimization/log")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--no_swanlab", action="store_true")

    args=ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    run_name = args.run_name or f"{args.dataset}_{args.model}_{int(time.time())}"
    csv_path = os.path.join(args.log_dir, f"{run_name}.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        print(f"[CUDA] torch={torch.__version__} cudnn={torch.backends.cudnn.version()} count={torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p=torch.cuda.get_device_properties(i)
            print(f"[CUDA] #{i}: {p.name} cc={p.major}.{p.minor} mem={p.total_memory/(1024**3):.1f}GB")
        cur=torch.cuda.current_device()
        print(f"[CUDA] current_device=#{cur} -> {torch.cuda.get_device_name(cur)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("[CPU] torch running on CPU")

    # Build loaders & models
    ModelCls = MODEL_REGISTRY.get(args.model)
    if ModelCls is None:
        raise ValueError(f"Unknown model: {args.model} ; available: {list(MODEL_REGISTRY.keys())}")

    if args.task=="vision":
        ds_tr, num_cls, in_ch = get_vision_dataset(args.dataset, True)
        ds_va, _, _ = get_vision_dataset(args.dataset, False)
        train_loader=DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_vision)
        val_loader  =DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_vision)
        if args.model=="small_cnn":
            model = ModelCls(in_ch=in_ch, num_classes=num_cls)
        elif args.model=="small_transformer":
            model = ModelCls(in_ch=in_ch, num_classes=num_cls, use_gate=False, pca_enable=False)
        elif args.model=="small_bert":
            # small_bert 的视觉路径保持与你原来一致
            model = ModelCls(text_task=None, in_ch=in_ch, num_classes=num_cls, pca_enable=False, use_gate=False)
        else:
            model = SmallCNN(in_ch=in_ch, num_classes=num_cls)
    else:
        assert args.dataset=="sst2", "textcls only supports sst2 here"
        ds_tr = get_sst2("train"); ds_va = get_sst2("validation")
        pipe = SST2Pipe(max_len=128)
        def collate_sst2(batch): return pipe.encode_batch(batch)
        train_loader=DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate_sst2)
        val_loader  =DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_sst2)
        # 文本模型
        if args.model=="distillbert_metagraph":
            model = DistilBertMetaGraph(text_task="cls", text_num_classes=2)
        elif args.model=="distillbert_hf":
            model = DistilBertHF(text_task="cls", text_num_classes=2)
        elif args.model=="distillbert_hf_gate":
            model = DistillBertHFGate(text_task="cls", text_num_classes=2, use_gate=False, pca_enable=False)
        elif args.model=="small_bert":
            model = SmallBERT(text_task="cls", text_num_classes=2, use_gate=False, pca_enable=False)
        else:
            model = DistilBertHF(text_task="cls", text_num_classes=2)

    model = model.to(device)

    # LR hint for text
    lr = args.weight_lr
    if args.task=="textcls" and lr >= 1e-4:
        print(f"[Hint] textcls: lowering lr {lr} -> 5e-5"); lr = 5e-5

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=args.adam_eps, weight_decay=args.adam_wd,
                                  fused=torch.cuda.is_available())

    # MetaGraph for vision
    mg = None
    if args.task=="vision":
        mg = MetaGraph(use_betti=args.mg_use_betti, use_cp=args.mg_use_cp,
                       use_coupling=args.mg_use_coupling, use_curr=args.mg_use_curr,
                       adv_eps=args.mg_adv_eps)

    # SwanLab
    use_swan = _SWAN_OK and (not args.no_swanlab)
    if use_swan:
        swanlab.init(project="metagraph", experiment_name=run_name, config=vars(args))

    # CSV header
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","phase","loss","acc","b0","b1","trigger_on"])

    best_val = -1.0
    trigger_on = False
    cp_state_txt = None
    betti_state_txt = None

    for ep in range(1, args.epochs+1):
        if ep <= args.trigger_warmup_epochs:
            trigger_on = False

        if args.task=="vision":
            tr = run_epoch_vision(model, mg, train_loader, device, optimizer=optimizer, trigger_on=trigger_on)
        else:
            tr = run_epoch_text(model, train_loader, device, optimizer=optimizer,
                                use_cp=args.mg_use_cp, use_betti=args.mg_use_betti,
                                trigger_on=trigger_on,
                                cp_state=cp_state_txt, betti_state=betti_state_txt)
            cp_state_txt = tr.pop("cp_state", cp_state_txt)
            betti_state_txt = tr.pop("betti_state", betti_state_txt)

        # 激活开关：上一轮 train acc 达阈值后启用
        if tr["acc"] >= args.trigger_acc:
            trigger_on = True

        if args.task=="vision":
            va = run_epoch_vision(model, mg, val_loader, device, optimizer=None, trigger_on=False)
        else:
            va = run_epoch_text(model, val_loader, device, optimizer=None,
                                use_cp=False, use_betti=False,  # eval 不做干预，只统计
                                trigger_on=False,
                                cp_state=cp_state_txt, betti_state=betti_state_txt)
            _ = va.pop("cp_state", None); _ = va.pop("betti_state", None)

        print(f"[Ep {ep:02d}] train: loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
              f"val: loss={va['loss']:.4f} acc={va['acc']:.4f} | trigger_on={trigger_on}")

        if use_swan:
            swanlab.log({
                "train/loss":tr["loss"], "train/acc":tr["acc"], "train/b0":tr["b0"], "train/b1":tr["b1"],
                "val/loss":va["loss"],   "val/acc":va["acc"],   "val/b0":va["b0"],   "val/b1":va["b1"],
                "trigger_on": float(trigger_on)
            }, step=ep)

        with open(csv_path, "a", newline="") as f:
            w=csv.writer(f)
            w.writerow([ep,"train",tr["loss"],tr["acc"],tr["b0"],tr["b1"],int(trigger_on)])
            w.writerow([ep,"val",  va["loss"],va["acc"],va["b0"],va["b1"],int(trigger_on)])

        if va["acc"]>best_val: best_val = va["acc"]

    print(f"Best val acc: {best_val:.4f}")
    print(f"CSV saved to: {csv_path}")

if __name__=="__main__":
    main()

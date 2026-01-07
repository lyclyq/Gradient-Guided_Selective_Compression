# train_eval.py
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


# DCA Derivative（导数版权重）
try:
    from modules.dca_derivative import DCADerivative, DCADrvCfg
except Exception:
    try:
        from dca_derivative import DCADerivative, DCADrvCfg
    except Exception:
        DCADerivative, DCADrvCfg = None, None

# from torchvision import datasets, transforms

# =========================
# 可选模块（DCA / Valor / RL / 解冻）
# =========================
try:
    from modules.dca import DCA, DCACfg
except Exception:
    DCA = DCACfg = None

try:
    from modules.valor import Valor, ValorCfg
except Exception:
    Valor = ValorCfg = None

try:
    from modules.valor_rl import ValorRL, ValorRLCfg
except Exception:
    ValorRL = ValorRLCfg = None

try:
    from modules.unfreezer import Unfreezer, UnfreezeCfg
except Exception:
    Unfreezer = UnfreezeCfg = None

try:
    from modules.gate_unfreezer import GateUnfreezer, GateUnfreezeCfg
except Exception:
    GateUnfreezer = GateUnfreezeCfg = None

# =========================
# 工具模块
# =========================
from utils.seed import set_seed
from utils.metrics import ece_score, EMAMeter
from utils.registry import MODEL_REGISTRY, META_REGISTRY
from utils.exp_logger import CSVLogger

import platform
# =========================
# 注册模型（图像 & 文本）
# =========================
import models.small_cnn
import models.small_transformer
import models.small_bert
import modules.metagraph

# 可选：timm 基座（ViT-Tiny / MLP-Mixer），如未安装会自动跳过注册
try:
    import models.vit_tiny_timm   # 若你添加了该文件
except Exception:
    pass
try:
    import models.mixer_tiny_timm # 若你添加了该文件
except Exception:
    pass

# 可选：HF DistilBERT 基座（若你添加了 models/distillbert_hf.py）
try:
    import models.distillbert_hf_gate  # 注册为 "distillbert_hf"
except Exception:
    pass


import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


# =========================
# 视觉数据集加载器（保持原状）
# =========================


# ========= Windows 多进程友好：顶层可 picklable 的 collate =========


def collate_textcls(batch):
    xs = []
    ys = []
    for b in batch:
        x = b["x"] if isinstance(b, dict) else getattr(b, "x")
        y = b["y"] if isinstance(b, dict) else getattr(b, "y")
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs).long()
    y = torch.stack(ys).long()
    return x, y


def collate_tokcls(batch):
    xs = []
    ys = []
    for b in batch:
        x = b["x"] if isinstance(b, dict) else getattr(b, "x")
        y = b["y"] if isinstance(b, dict) else getattr(b, "y")
        xs.append(x)
        y = torch.as_tensor(y)
        ys.append(y)
    x = torch.stack(xs).long()        # [B,T]
    y = torch.stack(ys).long()        # [B,T]
    return x, y
# ====================================================================

def get_fashion_mnist_loaders(batch=128, subset=20000, seed=42):

    from torchvision import datasets, transforms

    tfm = transforms.Compose([transforms.ToTensor()])
    dtrain = datasets.FashionMNIST("./data", train=True, download=True, transform=tfm)
    dtest = datasets.FashionMNIST("./data", train=False, download=True, transform=tfm)

    subset = min(subset, len(dtrain))
    part, _ = random_split(
        dtrain, [subset, len(dtrain) - subset],
        generator=torch.Generator().manual_seed(seed),
    )
    val_n = max(1000, int(0.1 * subset))
    tr_n = subset - val_n
    tr, val = random_split(
        part, [tr_n, val_n],
        generator=torch.Generator().manual_seed(seed + 1),
    )

    return (
        DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2),
        DataLoader(val, batch_size=256, shuffle=False, num_workers=2),
        DataLoader(dtest, batch_size=256, shuffle=False, num_workers=2),
    )


def get_cifar100_rgb_loaders(batch=64, subset=20000, seed=42):
    from torchvision import datasets, transforms

    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    dtrain = datasets.CIFAR100("./data", train=True, download=True, transform=tfm_train)
    dtest = datasets.CIFAR100("./data", train=False, download=True, transform=tfm_test)

    subset = min(subset, len(dtrain))
    part, _ = random_split(
        dtrain, [subset, len(dtrain) - subset],
        generator=torch.Generator().manual_seed(seed),
    )
    val_n = max(1000, int(0.1 * subset))
    tr_n = subset - val_n
    tr, val = random_split(
        part, [tr_n, val_n],
        generator=torch.Generator().manual_seed(seed + 1),
    )

    return (
        DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=False),
        DataLoader(val, batch_size=256, shuffle=False, num_workers=2, pin_memory=False),
        DataLoader(dtest, batch_size=256, shuffle=False, num_workers=2, pin_memory=False),
    )


def get_svhn_loaders(batch=64, subset=73257, seed=42):
    from torchvision import datasets, transforms

    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728),
                             (0.1980, 0.2010, 0.1970)),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728),
                             (0.1980, 0.2010, 0.1970)),
    ])

    dtrain = datasets.SVHN("./data", split="train", download=True, transform=tfm_train)
    dtest = datasets.SVHN("./data", split="test", download=True, transform=tfm_test)

    subset = min(subset, len(dtrain))
    idx = torch.randperm(len(dtrain), generator=torch.Generator().manual_seed(seed))[:subset]
    part = torch.utils.data.Subset(dtrain, idx)

    val_n = max(1000, int(0.1 * subset))
    tr_n = subset - val_n
    tr, val = random_split(
        part, [tr_n, val_n],
        generator=torch.Generator().manual_seed(seed + 1),
    )

    return (
        DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=False),
        DataLoader(val, batch_size=256, shuffle=False, num_workers=2, pin_memory=False),
        DataLoader(dtest, batch_size=256, shuffle=False, num_workers=2, pin_memory=False),
    )


def get_cifar10_gray_loaders(batch=64, subset=20000, seed=42):
    from torchvision import datasets, transforms

    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    tfm_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dtrain = datasets.CIFAR10("./data", train=True, download=True, transform=tfm_train)
    dtest = datasets.CIFAR10("./data", train=False, download=True, transform=tfm_test)

    subset = min(subset, len(dtrain))
    part, _ = random_split(
        dtrain, [subset, len(dtrain) - subset],
        generator=torch.Generator().manual_seed(seed),
    )
    val_n = max(1000, int(0.1 * subset))
    tr_n = subset - val_n
    tr, val = random_split(
        part, [tr_n, val_n],
        generator=torch.Generator().manual_seed(seed + 1),
    )

    return (
        DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=False),
        DataLoader(val, batch_size=256, shuffle=False, num_workers=2, pin_memory=False),
        DataLoader(dtest, batch_size=256, shuffle=False, num_workers=2, pin_memory=False),
    )


def get_cifar10_rgb_loaders(batch=64, subset=20000, seed=42):
    from torchvision import datasets, transforms

    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    dtrain = datasets.CIFAR10("./data", train=True, download=True, transform=tfm_train)
    dtest = datasets.CIFAR10("./data", train=False, download=True, transform=tfm_test)

    subset = min(subset, len(dtrain))
    part, _ = random_split(
        dtrain, [subset, len(dtrain) - subset],
        generator=torch.Generator().manual_seed(seed),
    )
    val_n = max(1000, int(0.1 * subset))
    tr_n = subset - val_n
    tr, val = random_split(
        part, [tr_n, val_n],
        generator=torch.Generator().manual_seed(seed + 1),
    )

    return (
        DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=False),
        DataLoader(val, batch_size=256, shuffle=False, num_workers=2, pin_memory=False),
        DataLoader(dtest, batch_size=256, shuffle=False, num_workers=2, pin_memory=False),
    )


def get_mnist_loaders(batch=128, subset=20000, seed=42):
    from torchvision import datasets, transforms

    tfm = transforms.Compose([transforms.ToTensor()])
    dtrain = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    dtest = datasets.MNIST("./data", train=False, download=True, transform=tfm)

    subset = min(subset, len(dtrain))
    part, _ = random_split(
        dtrain, [subset, len(dtrain) - subset],
        generator=torch.Generator().manual_seed(seed),
    )
    val_n = max(1000, int(0.1 * subset))
    tr_n = subset - val_n
    tr, val = random_split(
        part, [tr_n, val_n],
        generator=torch.Generator().manual_seed(seed + 1),
    )

    return (
        DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2),
        DataLoader(val, batch_size=256, shuffle=False, num_workers=2),
        DataLoader(dtest, batch_size=256, shuffle=False, num_workers=2),
    )


# =========================
# 文本数据集加载器
# =========================
def _require_hf():
    try:
        import datasets  # noqa
        import transformers  # noqa
    except Exception:
        raise RuntimeError("需要安装 `datasets` 与 `transformers` 才能使用文本任务：\n"
                           "pip install datasets transformers")


def get_text_classification_loaders(
    name="sst2", max_len=128, batch=64, subset=50000, seed=42, vocab="bert-base-uncased"
):
    _require_hf()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(vocab, use_fast=True)
    if name == "sst2":
        ds = load_dataset("glue", "sst2")
        text_key, label_key = "sentence", "label"
        num_classes = 2
    elif name == "ag_news":
        ds = load_dataset("ag_news")
        text_key, label_key = "text", "label"
        num_classes = 4
    elif name == "trec6":
        ds = load_dataset("trec", "coarse")
        text_key, label_key = "text", "label-coarse"
        num_classes = 6
    else:
        raise ValueError("unknown text dataset")

    def _proc(batch):
        enc = tok(batch[text_key], padding="max_length", truncation=True, max_length=max_len)
        return {"x": enc["input_ids"], "y": batch[label_key]}
    ds = ds.map(_proc, batched=True, remove_columns=ds["train"].column_names)
    ds.set_format(type="torch", columns=["x", "y"])

    # 验证划分
    if name == "sst2":
        tr, va, te = ds["train"], ds["validation"], ds["test"]
    else:
        tot = len(ds["train"])
        val_n = max(2000, int(0.1 * tot))
        tr = ds["train"].select(range(tot - val_n))
        va = ds["train"].select(range(tot - val_n, tot))
        te = ds["test"]

    g = torch.Generator().manual_seed(seed)
    if subset and len(tr) > subset:
        idx = torch.randperm(len(tr), generator=g)[:subset]
        tr = tr.select(idx.tolist())

    # Windows 友好 collate
    nw = 0 if platform.system() == "Windows" else 2
    return (
        DataLoader(tr, batch_size=batch, shuffle=True,  num_workers=nw, collate_fn=collate_textcls),
        DataLoader(va, batch_size=256,   shuffle=False, num_workers=nw, collate_fn=collate_textcls),
        DataLoader(te, batch_size=256,   shuffle=False, num_workers=nw, collate_fn=collate_textcls),
        num_classes,
    )


def get_token_classification_loaders(
    name="conll2003", max_len=128, batch=32, subset=20000, seed=42, vocab="bert-base-uncased"
):
    _require_hf()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("conll2003")  # 字段：tokens, ner_tags
    tok = AutoTokenizer.from_pretrained(vocab, use_fast=True)

    tag_names = ds["train"].features["ner_tags"].feature.names
    num_tags = len(tag_names)
    pad_tag_id = -100  # 供 CE 忽略

    # HuggingFace fast tokenizer 的 word_ids 对齐
    def _proc(batch):
        enc = tok(batch["tokens"], is_split_into_words=True,
                  padding="max_length", truncation=True, max_length=max_len)
        all_labels = []
        for i in range(len(batch["tokens"])):
            word_ids = enc.word_ids(batch_index=i)
            labels = batch["ner_tags"][i]
            lab_align, prev = [], None
            for wid in word_ids:
                if wid is None:
                    lab_align.append(pad_tag_id)
                elif wid != prev:
                    lab_align.append(labels[wid])
                else:
                    lab_align.append(labels[wid])  # 子词沿用同标签
                prev = wid
            all_labels.append(lab_align)
        return {"x": enc["input_ids"], "y": all_labels}

    ds = ds.map(_proc, batched=True, remove_columns=ds["train"].column_names)
    ds.set_format(type="torch", columns=["x", "y"])

    tr, va, te = ds["train"], ds["validation"], ds["test"]

    g = torch.Generator().manual_seed(seed)
    if subset and len(tr) > subset:
        idx = torch.randperm(len(tr), generator=g)[:subset]
        tr = tr.select(idx.tolist())

    def collate(batch):
        x = torch.stack([b["x"] for b in batch]).long()
        y = torch.stack([torch.tensor(b["y"]) for b in batch]).long()
        return x, y  # x:[B,T], y:[B,T]
    nw = 0 if platform.system() == "Windows" else 2
    return (
        DataLoader(tr, batch_size=batch, shuffle=True,  num_workers=nw, collate_fn=collate_tokcls),
        DataLoader(va, batch_size=128,   shuffle=False, num_workers=nw, collate_fn=collate_tokcls),
        DataLoader(te, batch_size=128,   shuffle=False, num_workers=nw, collate_fn=collate_tokcls),
        num_tags, pad_tag_id,
    )


# =========================
# 评估函数（兼容 vision / textcls / tokcls）
# =========================

def _forward_with_mask(model, x, task, pad_id):
    # DistilBERT 路径需要 attention_mask；SmallBERT/视觉按原状
    if task != "vision" and hasattr(model, "bert"):
        attn = (x != pad_id)
        return model(x, attention_mask=attn)
    else:
        return model(x)


@torch.no_grad()
def quick_eval(model, loader, device, task, pad_id_for_mask, max_batches=1):
    model.eval()
    n, corr, loss_sum = 0, 0, 0.0
    all_logits, all_y = [], []
    it = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = _forward_with_mask(model, x, task, pad_id_for_mask)

        if task in ("vision", "textcls"):
            loss = F.cross_entropy(logits, y)
            bs = x.size(0)
            loss_sum += float(loss.item()) * bs
            pred = logits.argmax(-1)
            corr += (pred == y).sum().item()
            n += bs
            all_logits.append(logits.detach().cpu()); all_y.append(y.detach().cpu())
        else:
            B, T, C = logits.shape
            logits_f = logits.reshape(B*T, C)
            y_f = y.reshape(B*T)
            loss = F.cross_entropy(logits_f, y_f, ignore_index=-100)
            loss_sum += float(loss.item()) * B
            valid = (y_f != -100)
            if valid.any():
                pred_f = logits_f.argmax(-1)
                corr += (pred_f[valid] == y_f[valid]).sum().item()
                n += valid.sum().item()
                all_logits.append(logits_f[valid].detach().cpu())
                all_y.append(y_f[valid].detach().cpu())

        it += 1
        if it >= max_batches:
            break

    ece = 0.0
    if len(all_logits) > 0:
        ece = ece_score(torch.cat(all_logits), torch.cat(all_y))
    acc = (corr / n) if n > 0 else 0.0
    denom = max(1, n if task != "tokcls" else it)  # tokcls 近似
    return loss_sum / denom, acc, ece


@torch.no_grad()
def evaluate(model, loader, device="cpu", ls=0.0, task="vision", pad_tag_id=-100, pad_id_for_mask=0):
    model.eval()
    n, corr, loss_sum = 0, 0, 0.0
    all_logits, all_y = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # —— 对文本任务优先尝试传 attention_mask（HF 模型需要），否则退回无 mask —— #
        logits = None
        if task in ("textcls", "tokcls"):
            try:
                attn = (x != pad_id_for_mask)
                logits = model(x, attention_mask=attn)
            except TypeError:
                logits = model(x)
        else:
            logits = model(x)

        if task in ("vision", "textcls"):
            # 过滤无效标签（例如 SST-2 test 的 y=-1）
            C = logits.shape[-1]
            yf = y.view(-1)
            valid = (yf >= 0) & (yf < C)
            if valid.any():
                lv = F.cross_entropy(logits[valid], yf[valid], label_smoothing=ls)
                loss_sum += float(lv.item()) * int(valid.sum().item())
                pred = logits.argmax(-1)
                corr += (pred[valid] == yf[valid]).sum().item()
                n += int(valid.sum().item())
                all_logits.append(logits[valid].detach().cpu())
                all_y.append(yf[valid].detach().cpu())

        elif task == "tokcls":
            # logits:[B,T,C], y:[B,T]，过滤 pad_tag_id
            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            y_flat = y.reshape(B * T)
            valid = (y_flat != pad_tag_id)
            if valid.any():
                lv = F.cross_entropy(logits_flat[valid], y_flat[valid])
                loss_sum += float(lv.item()) * B  # 以 batch 为单位累计
                pred_flat = logits_flat.argmax(-1)
                corr += (pred_flat[valid] == y_flat[valid]).sum().item()
                n += valid.sum().item()
                all_logits.append(logits_flat[valid].detach().cpu())
                all_y.append(y_flat[valid].detach().cpu())
        else:
            raise ValueError(f"unknown task={task}")

    ece = 0.0
    if len(all_logits) > 0:
        ece = ece_score(torch.cat(all_logits), torch.cat(all_y))
    acc = (corr / n) if n > 0 else 0.0
    # tokcls 用 batch 数近似；text/vision 用有效样本数
    denom = max(1, len(loader.dataset) if task != "tokcls" else len(loader))
    return loss_sum / denom, acc, ece

# =========================
# 主程序
# =========================
def main():
    ap = argparse.ArgumentParser()

    # 任务类型（新增）
    ap.add_argument("--task", choices=["vision", "textcls", "tokcls"], default="vision",
                    help="vision(图像)/textcls(句子分类)/tokcls(序列标注)")

    # 文本数据集（新增）
    ap.add_argument("--text_dataset", choices=["sst2", "ag_news", "trec6", "conll2003"], default="sst2")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--vocab_name", type=str, default="bert-base-uncased")
    ap.add_argument("--pad_id", type=int, default=0)

    # 视觉数据集
    ap.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist", "cifar10_gray", "cifar10_rgb", "cifar100_rgb", "svhn"],
        default="mnist",
    )

    # 模型 / 元模块
    ap.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="small_cnn")
    ap.add_argument("--meta", choices=["none", "valor", "valor_rl", "metagraph"], default="none")

    # 训练常规
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--subset", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_tag", type=str, default="exp1")
    ap.add_argument("--log_dir", type=str, default="./logs")

    # DARTS 一阶双优化器
    ap.add_argument("--weight_lr", type=float, default=3e-4)
    ap.add_argument("--arch_lr", type=float, default=1e-3)
    ap.add_argument("--weight_wd", type=float, default=1e-2)
    ap.add_argument("--gate_tau_start", type=float, default=1.5)
    ap.add_argument("--gate_tau_end", type=float, default=0.7)
    ap.add_argument("--arch_val_batches", type=int, default=1)

    # 逐层解冻（保留）
    ap.add_argument("--unfreeze_enable", action="store_true")
    ap.add_argument("--unfreeze_mode", choices=["epoch", "step", "metric"], default="epoch")
    ap.add_argument("--unfreeze_every", type=int, default=1)
    ap.add_argument("--unfreeze_min_layers", type=int, default=1)
    ap.add_argument("--unfreeze_llrd_decay", type=float, default=0.9)
    ap.add_argument("--unfreeze_target", choices=["weights", "darts"], default="weights")
    ap.add_argument("--unfreeze_alpha_lr_scale", type=float, default=1.0)

    # baseline / gate / resmix
    ap.add_argument("--gate_enable", action="store_true", help="启用 DARTS gate（baseline 设为 False）")
    ap.add_argument("--resmix_enable", action="store_true", help="启用软残差门控")
    ap.add_argument("--resmix_init", type=float, default=0.0, help="软门控初值（sigmoid 前的 beta）")

    # DCA
    ap.add_argument("--dca_enable", action="store_true")
    ap.add_argument("--dca_T", type=float, default=2.0)
    ap.add_argument("--dca_lambda_js", type=float, default=0.2)
    ap.add_argument("--dca_lambda_drift", type=float, default=0.2)
    ap.add_argument("--dca_beta_token_kl", type=float, default=0.05)
    ap.add_argument("--dca_entropy_w", type=float, default=1e-3)
    ap.add_argument("--dca_w_local", type=float, default=0.5, help="a: 层漂移贡献权重")
    ap.add_argument("--dca_w_global", type=float, default=0.5, help="b: 全局分布差对层能量的 proxy 权重")
    ap.add_argument("--resmix_from_dca_lr", type=float, default=0.0,
                    help=">0 启用：用 DCA 引导分数外部更新各层 resmix beta")
    ap.add_argument("--dca_verbose", action="store_true")
    ap.add_argument("--dca_w_lr", type=float, default=1e-3,
                help="DCA quickval 对控制核权重的学习率(>0 启用，仅作用于 DWConv 核)")

    ap.add_argument("--dca_mode", choices=["vanilla", "e_ctr", "e_ctr_resmix"], default="vanilla",
                    help="DCA 模式：vanilla(原逻辑) / e_ctr(只用 D 和 a_l) / e_ctr_resmix(外部引导 ResMix)")
    ap.add_argument("--dca_beta", type=float, default=1.0, help="E-CTR 的 β 耦合强度")




    # ------- DCA Derivative（导数门控）-------
    ap.add_argument("--dca_derivative_enable", action="store_true",
                    help="启用 DCA 导数版（趋势门控）")
    ap.add_argument("--dca_drv_mode", choices=["ema", "window"], default="ema",
                    help="导数估计方式：ema 或 window slope")
    ap.add_argument("--dca_drv_k_ctrl", type=int, default=5,
                    help="window slope 的 K（仅在 --dca_drv_mode=window 时使用）")
    ap.add_argument("--dca_drv_kappa", type=float, default=5.0,
                    help="sigmoid 的斜率 κ（越大门更“硬”）")
    ap.add_argument("--dca_drv_phi", choices=["sigmoid", "relu+"], default="sigmoid",
                    help="导数到权重的映射：sigmoid ∈ (0,1)；relu+ 仅放大上行趋势")
    ap.add_argument("--dca_drv_lambda", type=float, default=0.2,
                    help="导数分支的损失系数 λ_drv")
    ap.add_argument("--dca_drv_ema", type=float, default=0.8,
                    help="EMA 导数平滑系数 m（仅在 --dca_drv_mode=ema 时使用）")
    ap.add_argument("--dca_drv_only", action="store_true",
                    help="只用趋势门控：不叠加 E-CTR 的 loss 放缩（但仍可进行结构 nudging）")
    # ======== Filters / Schemes (1/2/3) ========
# ======== Filters / Schemes (1/2/3) ========
    ap.add_argument(
        "--filter_backend",
        choices=["darts", "lp_fixed", "none"],
        default="darts",
        help="门/滤波后端：darts(可学习) | lp_fixed(固定低通) | none(直通)",
    )

    ap.add_argument(
        "--ks_list",
        nargs="+",
        type=int,
        default=[3, 7, 15],
        help="darts 的多核 size 列表",
    )

    ap.add_argument(
        "--lp_k",
        type=int,
        default=7,
        help="固定低通核大小(奇数)",
    )

    # Scheme-3: Feature FFT Mask
    ap.add_argument(
        "--fftm_enable",
        action="store_true",
        help="启用特征FFT掩蔽（训练期更新EMA，验证期按重合度过滤）",
    )
    ap.add_argument(
        "--fftm_gamma",
        type=float,
        default=1.0,
        help="重合差异→频带衰减强度",
    )
    ap.add_argument(
        "--fftm_amin",
        type=float,
        default=0.8,
        help="频带最小保留比例 a_min",
    )
    ap.add_argument(
        "--fftm_apply_on",
        choices=["input", "output"],
        default="input",
        help="在门输入或输出上做频域衰减",
    )
    ap.add_argument(
        "--fftm_use_drv_gate",
        action="store_true",
        help="将 DCA 导数门 u 用作掩蔽强度增益",
    )

    # Scheme-2: Spectral regularization on kernels
    ap.add_argument(
        "--spec_reg_enable",
        action="store_true",
        help="对控制核做频域正则（抑制高频增益）",
    )
    ap.add_argument(
        "--spec_reg_lambda",
        type=float,
        default=1e-4,
        help="频域正则系数",
    )
    ap.add_argument(
        "--spec_reg_power",
        type=float,
        default=1.0,
        help="phi(f)=(lin)^power，增加高频惩罚曲率",
    )


    # ============ SwanLab & Model Saving ============
    ap.add_argument("--swan_enable", action="store_true", help="启用 SwanLab 日志")
    ap.add_argument("--swan_project", type=str, default="dca_exp", help="SwanLab 项目名")
    ap.add_argument("--swan_experiment", type=str, default=None, help="实验名（默认=run_tag）")
    ap.add_argument("--swan_dir", type=str, default="./swanlab", help="SwanLab 日志目录")

    ap.add_argument("--save_dir", type=str, default="./checkpoints", help="模型保存目录")
    ap.add_argument("--save_name", type=str, default=None, help="模型文件名（默认=run_tag.pt）")

    # —— 批间日志与快速验证 —— #
    ap.add_argument("--log_every", type=int, default=50,
                    help="每 log_every 个训练 batch 打印/记录一次训练窗口均值")
    ap.add_argument("--quickval_every", type=int, default=50,
                    help="每 quickval_every 个训练 batch 做一次快速验证与中途 α 更新")
    ap.add_argument("--quickval_batches", type=int, default=1,
                    help="快速验证最多取多少个 valid batch；同时用于中途 α 更新的 batch 数")

    # ==============================================

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.log_dir, exist_ok=True)

    # 尝试初始化 SwanLab（可选）
    swan = None
    if args.swan_enable:
        try:
            import swanlab
            exp_name = args.swan_experiment or args.run_tag
            swan = swanlab
            swan.init(
                project=args.swan_project,
                experiment_name=exp_name,
                logdir=args.swan_dir
            )
        except Exception as e:
            print(f"[SWANLAB] 初始化失败，自动跳过（{e}）")
            swan = None

    # =========================
    # 数据 & 通道/类别/任务配置
    # =========================
    task = args.task
    pad_tag_id = -100  # tokcls 用
    if task == "vision":
        if args.dataset == "mnist":
            tr, va, te = get_mnist_loaders(batch=args.batch, subset=args.subset, seed=args.seed)
            in_ch, num_classes = 1, 10
        elif args.dataset == "fashion_mnist":
            tr, va, te = get_fashion_mnist_loaders(batch=args.batch, subset=args.subset, seed=args.seed)
            in_ch, num_classes = 1, 10
        elif args.dataset == "cifar10_gray":
            tr, va, te = get_cifar10_gray_loaders(batch=args.batch, subset=args.subset, seed=args.seed)
            in_ch, num_classes = 1, 10
        elif args.dataset == "cifar10_rgb":
            tr, va, te = get_cifar10_rgb_loaders(batch=args.batch, subset=args.subset, seed=args.seed)
            in_ch, num_classes = 3, 10
        elif args.dataset == "cifar100_rgb":
            tr, va, te = get_cifar100_rgb_loaders(batch=args.batch, subset=args.subset, seed=args.seed)
            in_ch, num_classes = 3, 100
        elif args.dataset == "svhn":
            tr, va, te = get_svhn_loaders(batch=args.batch, subset=args.subset, seed=args.seed)
            in_ch, num_classes = 3, 10
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")
    elif task == "textcls":
        tr, va, te, num_classes = get_text_classification_loaders(
            name=args.text_dataset, max_len=args.max_len, batch=args.batch,
            subset=args.subset, seed=args.seed, vocab=args.vocab_name
        )
        in_ch = None
    elif task == "tokcls":
        tr, va, te, num_classes, pad_tag_id = get_token_classification_loaders(
            name="conll2003", max_len=args.max_len, batch=args.batch,
            subset=args.subset, seed=args.seed, vocab=args.vocab_name
        )
        in_ch = None
    else:
        raise ValueError(f"Unknown task {task}")

    # =========================
    # 模型
    # =========================
    Net = MODEL_REGISTRY[args.model]

    if task == "vision":
        if args.model == "small_transformer":
            net = Net(in_ch=in_ch, num_classes=num_classes,
                      gate_tau=args.gate_tau_start, cls_token=False,
                      use_gate=args.gate_enable,
                      use_resmix=args.resmix_enable, resmix_init=args.resmix_init).to(device)
        elif args.model == "small_cnn":
            net = Net(in_ch=in_ch, num_classes=num_classes).to(device)
        else:
            # 其它视觉基座（如 vit_tiny_timm / mixer_tiny_timm / small_bert 的图像路径等）
            try:
                net = Net(in_ch=in_ch, num_classes=num_classes,
                          dim=192, n_layers=4, heads=4, img_patch=4,
                          use_gate=args.gate_enable,
                          use_resmix=args.resmix_enable, resmix_init=args.resmix_init,
                          gate_tau=args.gate_tau_start).to(device)
            except TypeError:
                # 适配 timm 包装器等仅需 num_classes 的模型
                net = Net(num_classes=num_classes).to(device)

    elif task == "textcls":
        # SmallBERT / DistilBERT 均可
        if args.model == "small_bert":
            net = Net(text_task="cls", text_num_classes=num_classes,
                      vocab_size=30522, pad_id=args.pad_id, cls_token=False,
                      dim=192, n_layers=4, heads=4, img_patch=4,
                      use_gate=args.gate_enable, use_resmix=args.resmix_enable,
                      resmix_init=args.resmix_init, gate_tau=args.gate_tau_start,
                      filter_backend=args.filter_backend, ks_list=tuple(args.ks_list), lp_k=args.lp_k,
                      feature_mask=args.fftm_enable, ff_gamma=args.fftm_gamma, ff_amin=args.fftm_amin,
                      ff_apply_on=args.fftm_apply_on, ff_use_drv_gate=args.fftm_use_drv_gate).to(device)
        elif args.model == "distillbert_hf":
            net = Net(text_task="cls", text_num_classes=num_classes, pad_id=args.pad_id).to(device)

        elif args.model == "distillbert_hf_gate":
            net = Net(
                text_task="cls", text_num_classes=num_classes, pad_id=args.pad_id,
                use_gate=args.gate_enable, use_resmix=args.resmix_enable,
                resmix_init=args.resmix_init, gate_tau=args.gate_tau_start
            ).to(device)
        else:
            raise ValueError(f"模型 {args.model} 不支持文本分类（textcls）任务")

    elif task == "tokcls":
        if args.model == "small_bert":
            net = Net(text_task="tokcls", tok_num_classes=num_classes,
                      vocab_size=30522, pad_id=args.pad_id, cls_token=False,
                      dim=192, n_layers=4, heads=4,
                      use_gate=args.gate_enable, use_resmix=args.resmix_enable,
                      resmix_init=args.resmix_init, gate_tau=args.gate_tau_start).to(device)
        elif args.model == "distillbert_hf":
            net = Net(text_task="tokcls", tok_num_classes=num_classes, pad_id=args.pad_id).to(device)
        elif args.model == "distillbert_hf_gate":
            net = Net(
                text_task="tokcls", tok_num_classes=num_classes, pad_id=args.pad_id,
                use_gate=args.gate_enable, use_resmix=args.resmix_enable,
                resmix_init=args.resmix_init, gate_tau=args.gate_tau_start
            ).to(device)
        else:
            raise ValueError(f"模型 {args.model} 不支持序列标注（tokcls）任务")

    # =========================
    # 元模块（保留）
    # =========================
    meta = None
    if args.meta == "metagraph":
        meta = META_REGISTRY["metagraph"]()
        meta_name = "metagraph"
    elif args.meta == "valor":
        assert Valor is not None, "modules/valor.py 未找到；或移除 --meta valor"
        meta_valor = Valor(ValorCfg())
        knobs = {"ls": 0.0, "rdrop": 0.0, "dropout": 0.2, "smooth": 1}
        ema = EMAMeter(0.3)
        meta_valor.attach_validator(valid_loader=va, device=device, every_n_batches=0, max_batches=2)
        meta_name = "valor"
    elif args.meta == "valor_rl":
        assert ValorRL is not None, "modules/valor_rl.py 未找到；或移除 --meta valor_rl"
        meta_valor = ValorRL(ValorRLCfg())
        knobs = {"ls": 0.0, "rdrop": 0.0, "dropout": 0.2, "smooth": 1}
        ema = EMAMeter(0.3)
        meta_valor.attach_validator(valid_loader=va, device=device, every_n_batches=0, max_batches=2)
        meta_name = "valor_rl"
    else:
        meta_name = "none"
        knobs = {"ls": 0.0, "rdrop": 0.0, "dropout": 0.2, "smooth": 1}

    # =========================
    # 优化器：若有 arch/weight 分拆则双优化器
    # =========================
    has_arch = hasattr(net, "arch_parameters") and len(list(net.arch_parameters())) > 0
    if has_arch:
        weight_params = list(net.weight_parameters())
        arch_params = list(net.arch_parameters())
        opt_w = torch.optim.AdamW(weight_params, lr=args.weight_lr, weight_decay=args.weight_wd)
        opt_a = torch.optim.Adam(arch_params, lr=args.arch_lr)
        print(f"[OPT] DARTS enabled: |θ|={sum(p.numel() for p in weight_params)}, "
              f"|α|={sum(p.numel() for p in arch_params)}")
    else:
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    # =========================
    # 逐层解冻（保留）
    # =========================
    unfreezer = None
    if args.unfreeze_enable and has_arch:
        if args.unfreeze_target == "weights":
            assert Unfreezer is not None, "modules/unfreezer.py 未找到"
            ucfg = UnfreezeCfg(
                enable=True, mode=args.unfreeze_mode, every=args.unfreeze_every,
                min_layers=args.unfreeze_min_layers, llrd_decay=args.unfreeze_llrd_decay, verbose=True,
            )
            unfreezer = Unfreezer(net, opt_w, ucfg, path_resolver=None, base_lr=args.weight_lr)
        else:
            assert GateUnfreezer is not None, "modules/gate_unfreezer.py 未找到"
            gcfg = GateUnfreezeCfg(
                enable=True, mode=args.unfreeze_mode, every=args.unfreeze_every,
                min_layers=args.unfreeze_min_layers, lr_scale=args.unfreeze_alpha_lr_scale, verbose=True,
            )
            unfreezer = GateUnfreezer(net, opt_a, gcfg, base_lr=args.arch_lr)

    print(f"[SETUP] device={device} task={task} dataset={args.dataset if task=='vision' else args.text_dataset} "
          f"model={args.model} meta={args.meta} darts={has_arch} unfreeze={bool(unfreezer)}")

    # =========================
    # DCA
    # =========================
    dca = None
    if has_arch and args.dca_enable:
        assert DCA is not None, "modules/dca.py 未找到；关闭 --dca_enable 或添加该文件"
        dcfg = DCACfg(
            enable=True, T=args.dca_T, ema_beta=0.95,
            lambda_js=args.dca_lambda_js, lambda_drift=args.dca_lambda_drift,
            beta_token_kl=args.dca_beta_token_kl, entropy_w=args.dca_entropy_w,
            verbose=args.dca_verbose, w_local=args.dca_w_local, w_global=args.dca_w_global,
            beta_couple=args.dca_beta,
        )
        dca = DCA(dcfg, num_classes)
        dca.attach_hooks(net)

    # =========================
    # DCA Derivative（导数门控）
    # =========================
    dca_drv = None
    if (DCADerivative is not None) and args.dca_derivative_enable:
        dca_drv = DCADerivative(DCADrvCfg(
            enable=True,
            mode=args.dca_drv_mode,
            k_ctrl=args.dca_drv_k_ctrl,
            kappa=args.dca_drv_kappa,
            phi=args.dca_drv_phi,
            lambda_drv=args.dca_drv_lambda,
            ema_m=args.dca_drv_ema
        ))


    # =========================
    # CSV 日志器
    # =========================
    epoch_csv = os.path.join(args.log_dir, f"{args.run_tag}_epoch.csv")
    qv_csv    = os.path.join(args.log_dir, f"{args.run_tag}_quickval.csv")

    # 覆盖逻辑：先删除已有同名文件，避免本次 run 续写到上次文件尾部
    for p in (epoch_csv, qv_csv):
        try:
            if os.path.exists(p):
                os.remove(p)
                print(f"[LOG] overwrite: removed existing {p}")
        except Exception as e:
            print(f"[LOG] warn: failed to remove {p}: {e}")

    epoch_logger = CSVLogger(
        epoch_csv,
        fieldnames=[
            "run_tag", "epoch", "split", "loss", "acc", "ece", "meta",
            "ls", "rdrop", "dropout", "smooth", "val_every", "val_max_batches", "seed", "time",
        ],
    )
    qv_logger = CSVLogger(
        qv_csv,
        fieldnames=[
            "run_tag", "step", "epoch", "loss", "acc", "ece", "action",
            "ls", "rdrop", "dropout", "smooth", "meta", "time",
        ],
    )

    global_step = 0

    # =========================
    # 训练循环
    # =========================
    for ep in range(1, args.epochs + 1):
        # gate 温度退火
        if has_arch:
            tau0, tau1 = args.gate_tau_start, args.gate_tau_end
            tau = tau0 + (tau1 - tau0) * (ep - 1) / max(1, args.epochs - 1)
            for m in net.modules():
                if hasattr(m, "set_tau"):
                    m.set_tau(tau)

        net.train()
        n, loss_sum, acc_sum = 0, 0.0, 0.0

        # 窗口统计（每 log_every 打印/记录一次）
        win_loss, win_acc, win_cnt = 0.0, 0.0, 0

        for x, y in tr:
            x, y = x.to(device), y.to(device)

            if has_arch:
                # —— 更新 θ —— #
                opt_w.zero_grad()
                out_logits = _forward_with_mask(net, x, task, args.pad_id)

                if task in ("vision", "textcls"):
                    loss = F.cross_entropy(out_logits, y)
                else:  # tokcls
                    B, T, C = out_logits.shape
                    loss = F.cross_entropy(out_logits.reshape(B*T, C), y.reshape(B*T), ignore_index=pad_tag_id)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                opt_w.step()

                # 训练侧：更新 train 分布 EMA（DCA）
                if dca is not None:
                    with torch.no_grad():
                        if task in ("vision", "textcls"):
                            dca.update_train_distribution_ema(out_logits.detach(), y.detach())
                        else:
                            B, T, C = out_logits.shape
                            logits_f = out_logits.detach().reshape(B*T, C)
                            y_f = y.detach().reshape(B*T)
                            valid = (y_f != pad_tag_id)
                            if valid.any():
                                dca.update_train_distribution_ema(logits_f[valid], y_f[valid])

            else:
                if "opt" not in locals():
                    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
                opt.zero_grad()
                out_logits = _forward_with_mask(net, x, task, args.pad_id)
                if task in ("vision", "textcls"):
                    loss = F.cross_entropy(out_logits, y)
                else:
                    B, T, C = out_logits.shape
                    loss = F.cross_entropy(out_logits.reshape(B*T, C), y.reshape(B*T), ignore_index=pad_tag_id)
                loss.backward()
                opt.step()

            # —— 统计 —— #
            if task in ("vision", "textcls"):
                bs = x.size(0)
                batch_acc = (out_logits.argmax(-1) == y).float().mean().item()
                loss_sum += float(loss.item()) * bs
                acc_sum += (out_logits.argmax(-1) == y).float().sum().item()
                n += bs

                # 窗口
                win_loss += float(loss.item())
                win_acc  += batch_acc
                win_cnt  += 1
            else:
                # tokcls：按有效 token 计总 acc；窗口统计用 batch 均值
                B, T, C = out_logits.shape
                logits_f = out_logits.reshape(B*T, C)
                y_f = y.reshape(B*T)
                valid = (y_f != pad_tag_id)
                if valid.any():
                    pred_f = logits_f.argmax(-1)
                    acc_sum += (pred_f[valid] == y_f[valid]).float().sum().item()
                    n += valid.sum().item()
                    batch_acc = (pred_f[valid] == y_f[valid]).float().mean().item()
                else:
                    batch_acc = 0.0
                loss_sum += float(loss.item()) * B

                # 窗口
                win_loss += float(loss.item())
                win_acc  += batch_acc
                win_cnt  += 1

            # —— 每 log_every 批：打印/CSV/Swan —— #
            if args.log_every > 0 and (global_step + 1) % args.log_every == 0 and win_cnt > 0:
                avg_loss = win_loss / win_cnt
                avg_acc  = win_acc  / win_cnt

                # 控制台
                print(f"[S{global_step+1}] train(win@{args.log_every}) loss={avg_loss:.4f} acc={avg_acc:.4f}")

                # CSV quickval 里记一条“train_step”
                qv_logger.log({
                    "run_tag": args.run_tag, "step": global_step + 1, "epoch": ep,
                    "loss": f"{avg_loss:.6f}", "acc": f"{avg_acc:.6f}", "ece": "",
                    "action": "train_step",
                    "ls": f"{0.0:.4f}", "rdrop": f"{0.0:.4f}",
                    "dropout": f"{0.0:.4f}", "smooth": f"{1}",
                    "meta": args.meta, "time": f"{time.time():.0f}",
                })

                # SwanLab
                if swan is not None:
                    try:
                        swan.log({"train/step_loss": avg_loss, "train/step_acc": avg_acc}, step=global_step+1)
                    except Exception as e:
                        print(f"[SWANLAB] step-log 失败：{e}")

                # 清窗口
                win_loss, win_acc, win_cnt = 0.0, 0.0, 0

            # —— 每 quickval_every 批：做一次快速验证 + 中途 α 更新 —— #
            if has_arch and args.quickval_every > 0 and (global_step + 1) % args.quickval_every == 0:
                # 先 quick eval（无梯度）—— 注意这里传的是 pad_id（文本正确 mask）
                q_loss, q_acc, q_ece = quick_eval(net, va, device, task, args.pad_id, max_batches=args.quickval_batches)
                print(f"[S{global_step+1}] quickval(k={args.quickval_batches}) loss={q_loss:.4f} acc={q_acc:.4f} ece={q_ece:.4f}")

                # 记录 CSV/Swan
                qv_logger.log({
                    "run_tag": args.run_tag, "step": global_step + 1, "epoch": ep,
                    "loss": f"{q_loss:.6f}", "acc": f"{q_acc:.6f}", "ece": f"{q_ece:.6f}",
                    "action": "quickval",
                    "ls": f"{0.0:.4f}", "rdrop": f"{0.0:.4f}",
                    "dropout": f"{0.0:.4f}", "smooth": f"{1}",
                    "meta": args.meta, "time": f"{time.time():.0f}",
                })
                if swan is not None:
                    try:
                        swan.log({"quickval/loss": q_loss, "quickval/acc": q_acc, "quickval/ece": q_ece},
                                 step=global_step+1)
                    except Exception as e:
                        print(f"[SWANLAB] quickval-log 失败：{e}")

                # 再来中途 α 更新（带梯度，k 批）
                net.eval()
                with torch.enable_grad():
                    it_alpha = 0
                    for xv, yv in va:
                        if it_alpha >= max(0, args.quickval_batches):
                            break
                        xv, yv = xv.to(device), yv.to(device)

                        opt_a.zero_grad()
                        val_logits = _forward_with_mask(net, xv, task, args.pad_id)

                        if dca is not None:
                            controller_ws = []
                            if hasattr(net, "controller_kernel_weights"):
                                try:
                                    controller_ws = net.controller_kernel_weights()
                                except Exception:
                                    controller_ws = []

                            if task == "tokcls":
                                B, T, C = val_logits.shape
                                logits_f = val_logits.reshape(B*T, C)
                                y_f = yv.reshape(B*T)
                                valid = (y_f != pad_tag_id)
                                if valid.any():
                                    L_alpha, aux = dca.compute_arch_loss(
                                        logits_f[valid], y_f[valid], controller_kernel_weights=controller_ws
                                    )
                                else:
                                    L_alpha, aux = (val_logits.sum()*0.0), {}
                            else:
                                if args.dca_mode == "vanilla":
                                    L_alpha, aux = dca.compute_arch_loss(
                                        val_logits, yv, controller_kernel_weights=controller_ws
                                    )
                                elif args.dca_mode == "e_ctr":
                                    L_alpha, _ = dca.compute_e_ctr_loss(val_logits, yv, return_scores=False)
                                    aux = {}
                                elif args.dca_mode == "e_ctr_resmix":
                                    L_alpha, rs = dca.compute_e_ctr_loss(val_logits, yv, return_scores=True)
                                    aux = {"resmix_scores": rs}
                                else:
                                    L_alpha, aux = dca.compute_arch_loss(
                                        val_logits, yv, controller_kernel_weights=controller_ws
                                    )
                        else:
                            if task == "tokcls":
                                B, T, C = val_logits.shape
                                L_alpha = F.cross_entropy(val_logits.reshape(B*T, C), yv.reshape(B*T),
                                                          ignore_index=pad_tag_id)
                            else:
                                L_alpha = F.cross_entropy(val_logits, yv)
                            aux = {}

                        # ====== 兜底 + α-step + 控制核小步 ======
                        if (not isinstance(L_alpha, torch.Tensor)) or (not L_alpha.requires_grad):
                            print("[WARN] DCA arch loss has no grad; fallback to CE for α.")
                            if task == "tokcls":
                                B, T, C = val_logits.shape
                                L_alpha = F.cross_entropy(val_logits.reshape(B*T, C), yv.reshape(B*T),
                                                          ignore_index=pad_tag_id)
                            else:
                                L_alpha = F.cross_entropy(val_logits, yv)


                        # ====== DCA Derivative：用 L_alpha 作为 drift proxy 做趋势加权 ======
                        if dca_drv is not None:
                            # 只有当 dca_mode ∈ {e_ctr, e_ctr_resmix} 时，L_alpha 更贴近 D_t * a_l 的和；
                            # 若是 vanilla 分支也能用，只是 proxy 的“物理含义”不那么强。
                            try:
                                proxy = float(L_alpha.detach().item())
                            except Exception:
                                proxy = 0.0

                            weight, u_gate, d_est = dca_drv.update_and_weight(proxy)
                            # Propagate u_gate to FeatureFFTMask wrappers (scheme-3)
                            if args.fftm_use_drv_gate:
                                try:
                                    for m in net.modules():
                                        if hasattr(m, "set_gate"):
                                            m.set_gate(float(u_gate))
                                except Exception:
                                    pass
                            # Add spectral regularization on controller kernels (scheme-2)
                            if args.spec_reg_enable and hasattr(net, "controller_kernel_weights"):
                                try:
                                    from modules.spec_reg import spectral_penalty_depthwise
                                except Exception:
                                    from spec_reg import spectral_penalty_depthwise
                                cw = net.controller_kernel_weights()
                                # scale by (0.5 + u_gate) so drift↑ -> stronger penalty
                                L_spec = (0.5 + float(u_gate)) * spectral_penalty_depthwise(
                                    cw, lam=args.spec_reg_lambda, power=args.spec_reg_power
                                )
                                L_alpha = L_alpha + L_spec

                            # dca_drv_only: 只用导数分支；否则在原 L_alpha 上乘 (1 + λ_drv * weight)
                            if args.dca_drv_only:
                                L_alpha = (args.dca_drv_lambda * weight) * L_alpha
                            else:
                                L_alpha = (1.0 + args.dca_drv_lambda * weight) * L_alpha

                            # 可选：打印/记录（不影响训练）
                            if args.dca_verbose:
                                print(f"[DCA-DRV] proxy={proxy:.6f} d={d_est:.6f} gate(u)={u_gate:.4f} w={weight:.4f}")

                        # =======================================


                        L_alpha.backward()
                        opt_a.step()


                        # ====== 可选：用导数门控的 u_gate 轻推 resmix.beta ======
                        if (dca_drv is not None) and args.resmix_enable and hasattr(net, "blocks"):
                            try:
                                _proxy = float(L_alpha.detach().item())
                                _w, u_gate2, _d = dca_drv.update_and_weight(_proxy)
                                u_push = (2.0 * u_gate2 - 1.0)   # map (0,1)->(-1,+1)
                                step = getattr(args, "resmix_from_dca_lr", 0.0) * u_push
                                if step != 0.0:
                                    for b in net.blocks:
                                        if getattr(b, "use_resmix", False) and hasattr(b, "resmix"):
                                            with torch.no_grad():
                                                b.resmix.beta.add_(step)
                            except Exception:
                                pass

                        if args.dca_w_lr > 0.0 and dca is not None and controller_ws:
                            with torch.no_grad():
                                upd = 0
                                for p in controller_ws:
                                    if p.grad is not None:
                                        p.add_(-args.dca_w_lr * p.grad)
                                        p.grad.zero_()
                                        upd += 1
                            if upd > 0:
                                print(f"[DCA] applied weight-step to {upd} controller kernels (lr={args.dca_w_lr})")
                        # =====================================

                        # 可选：根据 DCA 分数微调 resmix beta
                        if args.resmix_enable and args.resmix_from_dca_lr > 0.0 and dca is not None:
                            rs = aux.get("resmix_scores", None)
                            if rs is not None and hasattr(net, "blocks"):
                                layer_idxs, scores = rs  # [L]
                                si = 0
                                for b in net.blocks:
                                    if getattr(b, "use_resmix", False) and hasattr(b, "resmix"):
                                        if si < len(scores):
                                            s = float(scores[si].item())
                                            step = (s - 0.5) * 2.0  # [-1, +1]
                                            with torch.no_grad():
                                                b.resmix.beta.add_(args.resmix_from_dca_lr * step)
                                        si += 1

                        if dca is not None:
                            dca.clear_val_cache()
                        it_alpha += 1

                net.train()

            # —— 逐层解冻（按 step）—— #
            if unfreezer is not None and args.unfreeze_mode == "step":
                unfreezer.on_step(global_step)

            global_step += 1

        # 验证
        va_loss, va_acc, va_ece = evaluate(net, va, device, task=task, pad_tag_id=pad_tag_id, pad_id_for_mask=args.pad_id)

        # —— 更新 α（小批验证上；支持 DCA）—— #
        if has_arch:
            net.eval()
            with torch.enable_grad():
                it = 0
                for xv, yv in va:
                    if it >= max(0, args.arch_val_batches):
                        break
                    xv, yv = xv.to(device), yv.to(device)

                    opt_a.zero_grad()
                    val_logits = _forward_with_mask(net, xv, task, args.pad_id)
                    if dca is not None:
                        controller_ws = []
                        if hasattr(net, "controller_kernel_weights"):
                            try:
                                controller_ws = net.controller_kernel_weights()
                            except Exception:
                                controller_ws = []

                        # tokcls：flatten 有效 token
                        if task == "tokcls":
                            B, T, C = val_logits.shape
                            logits_f = val_logits.reshape(B*T, C)
                            y_f = yv.reshape(B*T)
                            valid = (y_f != pad_tag_id)
                            if valid.any():
                                if args.dca_mode == "e_ctr":
                                    L_alpha, _ = dca.compute_e_ctr_loss(logits_f[valid], y_f[valid], return_scores=False)
                                    aux = {}
                                elif args.dca_mode == "e_ctr_resmix":
                                    L_alpha, rs = dca.compute_e_ctr_loss(logits_f[valid], y_f[valid], return_scores=True)
                                    aux = {"resmix_scores": rs}
                                else:
                                    L_alpha, aux = dca.compute_arch_loss(
                                        logits_f[valid], y_f[valid], controller_kernel_weights=controller_ws
                                    )
                            else:
                                L_alpha, aux = (val_logits.sum()*0.0), {}
                        else:
                            if args.dca_mode == "e_ctr":
                                L_alpha, _ = dca.compute_e_ctr_loss(val_logits, yv, return_scores=False)
                                aux = {}
                            elif args.dca_mode == "e_ctr_resmix":
                                L_alpha, rs = dca.compute_e_ctr_loss(val_logits, yv, return_scores=True)
                                aux = {"resmix_scores": rs}
                            else:
                                L_alpha, aux = dca.compute_arch_loss(
                                    val_logits, yv, controller_kernel_weights=controller_ws
                                )
                    else:
                        if task == "tokcls":
                            B, T, C = val_logits.shape
                            L_alpha = F.cross_entropy(val_logits.reshape(B*T, C), yv.reshape(B*T),
                                                      ignore_index=pad_tag_id)
                        else:
                            L_alpha = F.cross_entropy(val_logits, yv)
                        aux = {}

                    # ====== 兜底 + α-step + 控制核小步 ======
                    if (not isinstance(L_alpha, torch.Tensor)) or (not L_alpha.requires_grad):
                        print("[WARN] DCA arch loss has no grad; fallback to CE for α.")
                        if task == "tokcls":
                            B, T, C = val_logits.shape
                            L_alpha = F.cross_entropy(val_logits.reshape(B*T, C), yv.reshape(B*T),
                                                      ignore_index=pad_tag_id)
                        else:
                            L_alpha = F.cross_entropy(val_logits, yv)

                    L_alpha.backward()
                    opt_a.step()

                    if args.dca_w_lr > 0.0 and dca is not None and controller_ws:
                        with torch.no_grad():
                            upd = 0
                            for p in controller_ws:
                                if p.grad is not None:
                                    p.add_(-args.dca_w_lr * p.grad)
                                    p.grad.zero_()
                                    upd += 1
                        if upd > 0:
                            print(f"[DCA] applied weight-step to {upd} controller kernels (lr={args.dca_w_lr})")
                    # =====================================

                    # 可选：用 DCA 引导值微调 resmix beta（默认关闭）
                    if args.resmix_enable and args.resmix_from_dca_lr > 0.0 and dca is not None:
                        rs = aux.get("resmix_scores", None)
                        if rs is not None and hasattr(net, "blocks"):
                            layer_idxs, scores = rs  # [L]
                            si = 0
                            for b in net.blocks:
                                if getattr(b, "use_resmix", False) and hasattr(b, "resmix"):
                                    if si < len(scores):
                                        s = float(scores[si].item())
                                        step = (s - 0.5) * 2.0  # [-1, +1]
                                        with torch.no_grad():
                                            b.resmix.beta.add_(args.resmix_from_dca_lr * step)
                                    si += 1

                    if dca is not None:
                        dca.clear_val_cache()

                    it += 1

        # —— 记录 —— #
        epoch_logger.log({
            "run_tag": args.run_tag, "epoch": ep, "split": "train",
            "loss": f"{loss_sum / (n if task!='tokcls' else max(1, len(tr))):.6f}",
            "acc": f"{(acc_sum / max(1, n)):.6f}", "ece": "",
            "meta": args.meta, "ls": f"{0.0:.4f}", "rdrop": f"{0.0:.4f}",
            "dropout": f"{0.0:.4f}", "smooth": f"{1}",
            "val_every": 0, "val_max_batches": 2,
            "seed": args.seed, "time": f"{time.time():.0f}",
        })
        epoch_logger.log({
            "run_tag": args.run_tag, "epoch": ep, "split": "valid",
            "loss": f"{va_loss:.6f}", "acc": f"{va_acc:.6f}", "ece": f"{va_ece:.6f}",
            "meta": args.meta, "ls": f"{0.0:.4f}", "rdrop": f"{0.0:.4f}",
            "dropout": f"{0.0:.4f}", "smooth": f"{1}",
            "val_every": 0, "val_max_batches": 2,
            "seed": args.seed, "time": f"{time.time():.0f}",
        })

        # —— SwanLab 同步（按 epoch）—— #
        if swan is not None:
            try:
                swan.log({
                    "train/loss": loss_sum / (n if task != "tokcls" else max(1, len(tr))),
                    "train/acc": acc_sum / max(1, n),
                    "valid/loss": va_loss,
                    "valid/acc": va_acc,
                    "valid/ece": va_ece
                }, step=ep)
            except Exception as e:
                print(f"[SWANLAB] log 失败：{e}")

        if unfreezer is not None:
            if args.unfreeze_mode == "epoch":
                unfreezer.on_epoch_end(ep - 1)
            elif args.unfreeze_mode == "metric":
                unfreezer.on_metric(va_loss)

        print(
            f"[E{ep:02d}] train_loss={loss_sum / (n if task!='tokcls' else max(1, len(tr))):.4f} "
            f"train_acc={acc_sum / max(1, n):.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} ece={va_ece:.4f}"
        )

    # =========================
    # 测试
    # =========================
    te_loss, te_acc, te_ece = evaluate(net, te, device, task=task, pad_tag_id=pad_tag_id, pad_id_for_mask=args.pad_id)
    print(f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f} ece={te_ece:.4f}")

    # —— 保存最终权重（仅最终一次）—— #
    os.makedirs(args.save_dir, exist_ok=True)
    save_name = args.save_name or f"{args.run_tag}.pt"
    save_path = os.path.join(args.save_dir, save_name)
    torch.save(net.state_dict(), save_path)
    print(f"[SAVE] Final model saved to {save_path}")

    # —— SwanLab 记录测试并收尾 —— #
    if swan is not None:
        try:
            swan.log({"test/loss": te_loss, "test/acc": te_acc, "test/ece": te_ece})
            swan.finish()
        except Exception as e:
            print(f"[SWANLAB] finish 失败：{e}")

    epoch_logger.log({
        "run_tag": args.run_tag, "epoch": args.epochs, "split": "test",
        "loss": f"{te_loss:.6f}", "acc": f"{te_acc:.6f}", "ece": f"{te_ece:.6f}",
        "meta": args.meta, "ls": f"{0.0:.4f}", "rdrop": f"{0.0:.4f}",
        "dropout": f"{0.0:.4f}", "smooth": f"{1}",
        "val_every": 0, "val_max_batches": 2,
        "seed": args.seed, "time": f"{time.time():.0f}",
    })

    epoch_logger.close()
    qv_logger.close()


if __name__ == "__main__":
    main()

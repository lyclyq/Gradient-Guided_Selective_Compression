import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Tuple

def _make_generators(seed: int):
    """Create numpy RNG and torch.Generator for reproducibility."""
    rng = np.random.RandomState(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return rng, g

def _split_indices(n: int, ratios: Tuple[float, ...], rng: np.random.RandomState):
    """
    Return disjoint index splits according to ratios (sum<=1.0), with a shuffled, deterministic order.
    Returns (*splits, rest).
    """
    idx = np.arange(n)
    rng.shuffle(idx)
    cuts = []
    start = 0
    for r in ratios:
        k = int(round(n * r))
        cuts.append(idx[start:start + k])
        start += k
    rest = idx[start:]
    return (*cuts, rest)

def _dl(dataset, batch_size, shuffle, num_workers, gen=None, pin=True, collate_fn=None,
        prefetch_factor=4):
    """
    通用 DataLoader 构造：
    - pin_memory=True（默认）
    - persistent_workers：num_workers>0 时开启，避免重复 spawn
    - prefetch_factor：加速 worker 预取，默认 4（CPU/SSD 下通常更稳）
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        generator=gen,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

# ====================== Vision: MNIST ======================
def get_mnist(batch_size=128, val_ratio=0.1, num_workers=2, val2_ratio=None, seed=42):
    """
    Return 4 loaders: train, QV1, QV2, test with deterministic splits.

    Rules for MNIST:
    - 官方 test 存在：test = 官方 test；
    - 官方 val 不提供：QV1/QV2 均从官方 train 切分；
      总占比 = (val_ratio + (val2_ratio or val_ratio/2))，
      默认把 val_ratio 对半拆给 QV1/QV2。
    """
    import torchvision
    from torchvision import transforms
    rng, gen = _make_generators(seed)

    tfm = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    testset  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    n = len(trainset)
    if val2_ratio is None:
        vr1 = float(val_ratio) / 2.0
        vr2 = float(val_ratio) - vr1
    else:
        vr1 = float(val_ratio)
        vr2 = float(val2_ratio)

    val1_idx, val2_idx, tr_idx = _split_indices(n, (vr1, vr2), rng)

    tr_sub  = Subset(trainset, tr_idx)
    va1_sub = Subset(trainset, val1_idx)
    va2_sub = Subset(trainset, val2_idx)

    tr  = _dl(tr_sub,  batch_size, True,  num_workers, gen)
    va1 = _dl(va1_sub, batch_size, False, num_workers)
    va2 = _dl(va2_sub, batch_size, False, num_workers)
    te  = _dl(testset,  batch_size, False, num_workers)
    return tr, va1, va2, te

# ====================== NLP: SST-2 (GLUE) ======================
def get_sst2(batch_size=32, val_ratio=0.1, num_workers=2, model_name="distilbert-base-uncased", seed=42):
    """
    SST-2 没有公开可用标签的官方 test（GLUE 的 test 隐藏标签）：
    - 把“官方 validation”当 test；
    - QV1/QV2 从官方 train 切：总 0.1，各 0.05（val_ratio 控制总量）。
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    rng, gen = _make_generators(seed)

    ds = load_dataset("glue", "sst2")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def enc(ex):
        out = tok(ex["sentence"], truncation=True, padding="max_length", max_length=128)
        out["labels"] = ex["label"]
        return out

    # 预编码并去掉原始列
    ds_enc = ds.map(enc, batched=True, remove_columns=ds["train"].column_names)
    ds_enc = ds_enc.rename_column("labels", "label")

    # ✅ 关键：直接让 datasets 产出 torch.Tensor，避免手写 collate_fn 的 Python 循环开销
    ds_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_ds = ds_enc["train"]
    test_ds  = ds_enc["validation"]  # 官方 val 作为 test

    n = len(train_ds)
    if val_ratio <= 0:
        vr1 = 0.05; vr2 = 0.05
    else:
        vr1 = float(val_ratio) / 2.0
        vr2 = float(val_ratio) - vr1

    # deterministic split on train
    idx = np.arange(n); rng.shuffle(idx)
    v1 = int(round(n * vr1))
    v2 = int(round(n * vr2))
    val1_idx = idx[:v1]
    val2_idx = idx[v1:v1+v2]
    tr_idx   = idx[v1+v2:]

    tr_ds  = train_ds.select(tr_idx.tolist())
    va1_ds = train_ds.select(val1_idx.tolist())
    va2_ds = train_ds.select(val2_idx.tolist())

    # 直接用默认的 torch collate（已经是张量），不再需要自定义 collate_fn
    tr  = _dl(tr_ds,  batch_size, True,  num_workers, gen)
    va1 = _dl(va1_ds, batch_size, False, num_workers)
    va2 = _dl(va2_ds, batch_size, False, num_workers)
    te  = _dl(test_ds,  batch_size, False, num_workers)
    return tr, va1, va2, te, tok

# ====================== Generic (无官方 test/val 的情况) ======================
def split_general_dataset(dataset, batch_size=128, num_workers=2, seed=42,
                          val_total_ratio=0.1, qv1_ratio=0.05, qv2_ratio=0.05):
    """
    当既没有官方 test，也没有官方 val 时使用：
    - 总体切 0.8 / 0.2 -> test；
    - 0.8 中再切 0.9 / 0.1 -> train / val；
    - val 再对半给 QV1 / QV2（默认各 0.05，对应总 0.1）。
    返回 (train_loader, qv1_loader, qv2_loader, test_loader)
    """
    rng, gen = _make_generators(seed)
    n = len(dataset)
    # 先切 test
    test_ratio = 0.2
    test_idx, rest_idx = _split_indices(n, (test_ratio,), rng)
    rest = Subset(dataset, rest_idx)
    test = Subset(dataset, test_idx)

    # 再把 rest 切 train/val（0.9/0.1）
    nr = len(rest_idx)
    val_ratio = val_total_ratio  # 0.1
    tr_ratio  = 1.0 - val_ratio  # 0.9
    val_idx_r, tr_idx_r = _split_indices(nr, (val_ratio,), rng)  # 返回 (val, rest->train)
    val = Subset(rest, val_idx_r)
    train = Subset(rest, tr_idx_r)

    # 把 val 对半给 QV1/QV2
    nv = len(val)
    nv1 = int(round(nv * (qv1_ratio / val_total_ratio)))
    idx_v = np.arange(nv); rng.shuffle(idx_v)
    qv1_sel = idx_v[:nv1]
    qv2_sel = idx_v[nv1:]
    qv1 = Subset(val, qv1_sel)
    qv2 = Subset(val, qv2_sel)

    tr_loader  = _dl(train, batch_size, True,  num_workers, gen)
    qv1_loader = _dl(qv1,   batch_size, False, num_workers)
    qv2_loader = _dl(qv2,   batch_size, False, num_workers)
    te_loader  = _dl(test,  batch_size, False, num_workers)
    return tr_loader, qv1_loader, qv2_loader, te_loader

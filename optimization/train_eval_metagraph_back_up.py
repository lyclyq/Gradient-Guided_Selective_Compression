
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_metagraph.py
- Plug-in MetaGraph runner with optional feature-noise masking + accuracy trigger.
- Keeps dataset/model choices and logging style close to train_eval_final.py, but focuses on MetaGraph behaviors.

Key features:
* Datasets: CIFAR-10, CIFAR-100, MNIST (vision) and SST-2 (text classification).
* Models: small_cnn, small_transformer, small_bert, distillbert_hf, distillbert_hf_gate (as available in your repo).
* MetaGraph integration:
    - Two optional noise-mask modules:
        (A) CP-mask: channel-wise mask from drift/variance (ChangePointLite-inspired)
        (B) Betti-scale: global soft scaling from Betti b1
    - Accuracy trigger: only activate MetaGraph corrections after train_acc >= threshold (e.g., 0.90).
* Logging:
    - Console epoch summaries
    - CSV logging (train/val metrics + meta stats)
    - Optional SwanLab logging if installed (auto-skip if not)
"""

import os, sys, math, time, json, csv, random, argparse
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Optional dependencies
try:
    import swanlab
    _SWAN_OK = True
except Exception:
    _SWAN_OK = False

# ===== Project utils / registry =====
from utils.registry import MODEL_REGISTRY

# ===== Models (ensure import for registry population) =====
# The user provided these files in the project.
from models.small_cnn import SmallCNN       # noqa: F401
from models.small_transformer import SmallTransformer  # noqa: F401
from models.small_bert import SmallBERT     # noqa: F401
from models.distillbert_hf import DistilBertHF        # noqa: F401
from models.distillbert_hf_gate import DistillBertHFGate  # noqa: F401

# ===== MetaGraph (user's module, as given) =====
from modules.metagraph import MetaGraph

# ===== Vision/Text datasets =====
import torchvision
import torchvision.transforms as T

try:
    from datasets import load_dataset
    _HF_OK = True
except Exception:
    _HF_OK = False


# ------------------------------
# Utility: seeding
# ------------------------------
def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ------------------------------
# Feature hooks (optional masking)
# ------------------------------
class FeatureTap:
    """
    Optional feature interception for models exposing a 'features' Sequential (e.g., SmallCNN).
    Provides a simple way to read/modify the activation before the classifier.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self._handle = None
        self._last_feat = None
        self._mask_fn = None  # callable(tensor)->tensor

    def install(self):
        feats = getattr(self.model, "features", None)
        if feats is None or not isinstance(feats, nn.Sequential):
            return False

        def _hook(module, inp, out):
            x = out
            if self._mask_fn is not None and isinstance(x, torch.Tensor):
                try:
                    x = self._mask_fn(x)
                except Exception:
                    pass
            self._last_feat = x.detach() if isinstance(x, torch.Tensor) else None
            return x

        # register on the last module of features
        last = feats[-1]
        self._handle = last.register_forward_hook(_hook)
        return True

    def set_mask(self, fn):
        self._mask_fn = fn

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def last(self):
        return self._last_feat


# ------------------------------
# CP-based channel mask (Scheme A)
# ------------------------------
class CPChannelMask:
    """
    Maintain EMA mean/var for per-channel average and return a soft channel mask.
    - Compute channel stats on pooled feature [B,C,H,W] -> [C]
    - noise_score = |bmean - mu| / (sqrt(var)+eps)
    - mask = sigmoid(k*(mean(noise_score) - noise_score))  ∈ (0,1)
    """
    def __init__(self, channels: int, win: int = 50, alpha: float = 0.05, k: float = 4.0, eps: float = 1e-6):
        self.C = channels
        self.mu = None
        self.var = None
        self.k = k
        self.eps = eps

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() == 4:
            # [B,C,H,W] -> [C]
            bmean = feats.mean(dim=(0, 2, 3)).detach()
        elif feats.dim() == 2:
            bmean = feats.mean(dim=0).detach()
        else:
            return feats

        if self.mu is None:
            self.mu = bmean.clone()
            self.var = torch.ones_like(bmean)
        else:
            # EMA
            self.mu = 0.9 * self.mu + 0.1 * bmean
            self.var = 0.9 * self.var + 0.1 * (bmean - self.mu).pow(2)

        noise_score = (bmean - self.mu).abs() / (self.var.sqrt() + self.eps)  # [C]
        ns_mean = noise_score.mean()
        # higher noise_score → lower mask
        mask_c = torch.sigmoid(self.k * (ns_mean - noise_score))  # [C] in (0,1)
        if feats.dim() == 4:
            return feats * mask_c.view(1, -1, 1, 1)
        else:
            return feats * mask_c.view(1, -1)


# ------------------------------
# Betti-based global scaling (Scheme B)
# ------------------------------
class BettiScaler:
    """
    Use MetaGraph._BettiSketchLite output b1 to scale features globally.
    - Track EMA of b1; if b1 is above EMA, downscale features softly.
    - scale = 1 - clamp(gamma * tanh((b1 - ema)/norm), 0, max_drop)
    """
    def __init__(self, metagraph: MetaGraph, gamma: float = 0.05, max_drop: float = 0.2):
        self.mg = metagraph
        self.gamma = gamma
        self.max_drop = max_drop
        self._ema_b1: Optional[float] = None

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                b = self.mg._betti(feats) if self.mg._betti is not None else None
            except Exception:
                b = None
            if b is None:
                return feats
            b1 = float(b[1].item()) if isinstance(b, torch.Tensor) else float(b[1])
            if self._ema_b1 is None:
                self._ema_b1 = b1
            else:
                self._ema_b1 = 0.9 * self._ema_b1 + 0.1 * b1
            gap = max(0.0, b1 - self._ema_b1)
            scale = 1.0 - min(self.max_drop, self.gamma * math.tanh(gap))
        return feats * scale


# ------------------------------
# Datasets
# ------------------------------
def get_vision_dataset(name: str, train: bool, root: str = "./data"):
    name = name.lower()
    tf_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    tf_test = T.Compose([T.ToTensor()])

    if name == "c10":
        ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True,
                                          transform=tf_train if train else tf_test)
        num_classes = 10; in_ch = 3
    elif name == "c100":
        ds = torchvision.datasets.CIFAR100(root=root, train=train, download=True,
                                           transform=tf_train if train else tf_test)
        num_classes = 100; in_ch = 3
    elif name == "mnist":
        ds = torchvision.datasets.MNIST(root=root, train=train, download=True,
                                        transform=tf_test if not train else T.Compose([
                                            T.RandomCrop(28, padding=2),
                                            T.ToTensor()]))
        num_classes = 10; in_ch = 1
    else:
        raise ValueError(f"Unknown vision dataset: {name}")
    return ds, num_classes, in_ch


def get_sst2(split: str = "train"):
    assert _HF_OK, "datasets[HF] is required for SST-2"
    ds = load_dataset("glue", "sst2", split=split)
    # Simple tokenizer: rely on DistilBertHF/DistillBertHFGate internal tokenization? No – those expect input_ids.
    # For simplicity, we will use a basic HF tokenizer pipeline if the user chooses distillbert models.
    return ds


# ------------------------------
# Collates
# ------------------------------
def collate_vision(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


# ------------------------------
# Simple Text pipeline (SST-2) for DistilBERT family
# ------------------------------
class SST2Pipe:
    def __init__(self, model_name="distilbert-base-uncased", max_len=128, pad_id=0):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.pad_id = pad_id

    def encode_batch(self, batch):
        texts = [ex["sentence"] for ex in batch]
        lab = torch.tensor([int(ex["label"]) for ex in batch], dtype=torch.long)
        enc = self.tok(texts, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        input_ids = enc["input_ids"]
        return input_ids, lab


# ------------------------------
# Accuracy & loss
# ------------------------------
def compute_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    if logits.dim() == 3:
        # token classification → mask not handled here; use simple argmax mean
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
    else:
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
    return acc


# ------------------------------
# Train/Eval loops
# ------------------------------
def run_epoch(model, mg: MetaGraph, loader, device, task: str, optimizer=None,
              feature_tap: Optional[FeatureTap] = None,
              mask_mode: str = "none",
              betti_scaler: Optional[BettiScaler] = None,
              cp_masker: Optional[CPChannelMask] = None,
              trigger_on: bool = True):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n_samples = 0
    b0_sum = 0.0
    b1_sum = 0.0
    n_betti = 0

    for batch in loader:
        if task == "vision":
            x, y = (t.to(device) for t in batch)
            adv_x = mg.maybe_adv(model, x, y) if trigger_on else None
            loss, logits, extras = mg.forward_loss(model, x, y, adv_x=adv_x)
        else:  # textcls (SST-2)
            input_ids, y = (t.to(device) for t in batch)
            # MetaGraph requires model(x) signature returning logits; distillbert returns directly.
            logits = model(input_ids) if hasattr(model, "__call__") else model.forward(input_ids)
            loss = F.cross_entropy(logits, y)
            extras = {}

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # curriculum / knobs
            if task == "vision":
                mg.on_batch_end(model, x, y, logits, loss)

        with torch.no_grad():
            acc = compute_acc(logits, y)
            total_loss += float(loss.item()) * x.size(0) if task == "vision" else float(loss.item()) * input_ids.size(0)
            total_acc  += acc * (x.size(0) if task == "vision" else input_ids.size(0))
            n_samples  += (x.size(0) if task == "vision" else input_ids.size(0))
            if "betti" in extras and isinstance(extras["betti"], torch.Tensor):
                b0_sum += float(extras["betti"][0].item())
                b1_sum += float(extras["betti"][1].item())
                n_betti += 1

    mean_loss = total_loss / max(1, n_samples)
    mean_acc  = total_acc  / max(1, n_samples)
    mean_b0   = b0_sum / max(1, n_betti)
    mean_b1   = b1_sum / max(1, n_betti)
    return {"loss": mean_loss, "acc": mean_acc, "b0": mean_b0, "b1": mean_b1}


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["vision", "textcls"], default="vision")
    ap.add_argument("--dataset", choices=["c10", "c100", "mnist", "sst2"], default="c10")
    ap.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="small_cnn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight_lr", type=float, default=3e-4)
    ap.add_argument("--adam_eps", type=float, default=1e-8)
    ap.add_argument("--adam_wd", type=float, default=0.01)

    # MetaGraph options
    ap.add_argument("--mg_use_betti", action="store_true")
    ap.add_argument("--mg_use_cp", action="store_true")
    ap.add_argument("--mg_use_coupling", action="store_true")
    ap.add_argument("--mg_use_curr", action="store_true")
    ap.add_argument("--mg_adv_eps", type=float, default=0.1)

    # Trigger
    ap.add_argument("--trigger_acc", type=float, default=0.90, help="Activate MetaGraph & masks after this train acc is reached")
    ap.add_argument("--trigger_warmup_epochs", type=int, default=1, help="Always off for the first N epochs")

    # Noise mask modules
    ap.add_argument("--mask_mode", choices=["none", "cp", "betti", "both"], default="none")
    ap.add_argument("--cp_k", type=float, default=4.0)
    ap.add_argument("--betti_gamma", type=float, default=0.05)
    ap.add_argument("--betti_max_drop", type=float, default=0.2)

    # Logging
    ap.add_argument("--log_dir", type=str, default="optimization/log")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--no_swanlab", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    run_name = args.run_name or f"{args.dataset}_{args.model}_{int(time.time())}"
    csv_path = os.path.join(args.log_dir, f"{run_name}.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- CUDA startup info ----
    if device.type == "cuda":
        print(f"[CUDA] torch={torch.__version__}  cudnn={torch.backends.cudnn.version()}  "
            f"device_count={torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            total_gb = p.total_memory / (1024**3)
            print(f"[CUDA] #{i}: {p.name}  cc={p.major}.{p.minor}  total_mem={total_gb:.1f}GB")
        cur = torch.cuda.current_device()
        print(f"[CUDA] current_device=#{cur} -> {torch.cuda.get_device_name(cur)}")
    else:
        print("[CPU] torch running on CPU")


    # Build model
    ModelCls = MODEL_REGISTRY.get(args.model)
    if ModelCls is None:
        raise ValueError(f"Unknown model: {args.model} ; available: {list(MODEL_REGISTRY.keys())}")

    # Dataset & loaders
    if args.task == "vision":
        ds_train, num_classes, in_ch = get_vision_dataset(args.dataset, train=True)
        ds_val, _, _ = get_vision_dataset(args.dataset, train=False)
        collate_fn = collate_vision
        if args.model == "small_cnn":
            model = ModelCls(in_ch=in_ch, num_classes=num_classes)
        elif args.model == "small_transformer":
            model = ModelCls(in_ch=in_ch, num_classes=num_classes, use_gate=False, pca_enable=False)
        elif args.model == "small_bert":
            # Use image path of SmallBERT
            model = ModelCls(text_task=None, in_ch=in_ch, num_classes=num_classes, pca_enable=False, use_gate=False)
        else:
            # fall back to small_cnn-like head if vision
            model = SmallCNN(in_ch=in_ch, num_classes=num_classes)
        train_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
        val_loader   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    else:
        assert args.dataset == "sst2", "Only SST-2 supported for textcls here"
        assert _HF_OK, "Please install datasets for SST-2"
        ds_tr = get_sst2("train")
        ds_va = get_sst2("validation")
        pipe = SST2Pipe()
        def collate_sst2(batch):
            return pipe.encode_batch(batch)
        train_loader = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate_sst2)
        val_loader   = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_sst2)
        # Text models
        if args.model == "distillbert_hf":
            model = DistilBertHF(text_task="cls", text_num_classes=2)
        elif args.model == "distillbert_hf_gate":
            model = DistillBertHFGate(text_task="cls", text_num_classes=2, use_gate=False, pca_enable=False)
        else:
            # fallback to DistilBertHF
            model = DistilBertHF(text_task="cls", text_num_classes=2)

    model = model.to(device)

    # MetaGraph instance
    mg = MetaGraph(
        use_betti=args.mg_use_betti, use_cp=args.mg_use_cp,
        use_coupling=args.mg_use_coupling, use_curr=args.mg_use_curr,
        adv_eps=args.mg_adv_eps
    )

    # Optional feature tap + masks (only effective if model has .features Sequential, e.g., SmallCNN)
    tap = FeatureTap(model)
    tap_ok = tap.install()
    cp_masker = None
    betti_scaler = None
    if tap_ok:
        if args.mask_mode in ("cp", "both"):
            # infer channels from a dummy forward
            with torch.no_grad():
                if args.task == "vision":
                    sample = next(iter(val_loader))[0][:1].to(device)
                    feat = model.features(sample)
                    C = feat.shape[1]
                else:
                    C = None
            if C is not None:
                cp_masker = CPChannelMask(channels=C, k=args.cp_k)
        if args.mask_mode in ("betti", "both"):
            betti_scaler = BettiScaler(metagraph=mg, gamma=args.betti_gamma, max_drop=args.betti_max_drop)

        def mask_fn(x):
            y = x
            if cp_masker is not None:
                y = cp_masker(y)
            if betti_scaler is not None:
                y = betti_scaler(y)
            return y
        tap.set_mask(mask_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.weight_lr, eps=args.adam_eps, weight_decay=args.adam_wd)

    # SwanLab
    use_swan = _SWAN_OK and (not args.no_swanlab)
    if use_swan:
        swanlab.init(project="metagraph", experiment_name=run_name, config=vars(args))

    # CSV writer
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "loss", "acc", "b0", "b1", "trigger_on"])

    best_val = -1.0
    trigger_on = False

    for ep in range(1, args.epochs + 1):
        # Trigger rule: warmup epochs off; after that, enable if previous epoch train acc >= threshold
        if ep <= args.trigger_warmup_epochs:
            trigger_on = False
        else:
            # keep previous value unless last train acc crosses threshold
            pass

        # ---- Train ----
        tr = run_epoch(model, mg, train_loader, device, args.task, optimizer=optimizer,
                       feature_tap=tap if tap_ok else None,
                       mask_mode=args.mask_mode, betti_scaler=betti_scaler, cp_masker=cp_masker,
                       trigger_on=trigger_on)
        # Update trigger by last epoch's train acc
        if tr["acc"] >= args.trigger_acc:
            trigger_on = True

        # ---- Val ----
        va = run_epoch(model, mg, val_loader, device, args.task, optimizer=None,
                       feature_tap=tap if tap_ok else None,
                       mask_mode="none", betti_scaler=None, cp_masker=None,
                       trigger_on=False)  # no trigger for eval

        # Logging
        msg = f"[Ep {ep:02d}] train: loss={tr['loss']:.4f} acc={tr['acc']:.4f} | val: loss={va['loss']:.4f} acc={va['acc']:.4f} | trigger_on={trigger_on}"
        print(msg)

        if use_swan:
            swanlab.log({
                "train/loss": tr["loss"], "train/acc": tr["acc"], "train/b0": tr["b0"], "train/b1": tr["b1"],
                "val/loss": va["loss"],   "val/acc": va["acc"],   "val/b0": va["b0"],   "val/b1": va["b1"],
                "trigger_on": float(trigger_on)
            }, step=ep)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, "train", tr["loss"], tr["acc"], tr["b0"], tr["b1"], int(trigger_on)])
            writer.writerow([ep, "val", va["loss"], va["acc"], va["b0"], va["b1"], int(trigger_on)])

        if va["acc"] > best_val:
            best_val = va["acc"]

    print(f"Best val acc: {best_val:.4f}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

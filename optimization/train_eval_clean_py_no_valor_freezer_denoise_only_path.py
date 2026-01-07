#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored train_eval.py (step 1):
- REMOVE: valor / valor_rl / freeze related flags & branches
- KEEP:   meta graph (DCA), residual gating (SoftResMix), fixed low-pass, FFT feature mask, FFT weight spectral reg,
          DCA derivative (EMA) for overfit-direction sensing
- ADD:    insert_layers selector (e.g., "last", "last2", "0,1,4", "all")
- POLICY: strictly **denoise-only** — NO train→valid compensation. Any valid-driven signals are used only for
          *gating/regularization strength*, not for shifting/centering features.

NOTE
----
This file is designed to be a drop-in replacement for the existing train_eval.py.
It assumes the following modules exist in the project tree (already present in your repo uploads):
 - dca.py, dca_e_ctr.py, dca_e_ctr_resmix.py, dca_derivative.py
 - soft_resmix.py, filter_lp_fixed.py, feature_fft_mask.py, spec_reg.py
 - small_transformer.py, small_bert.py, distillbert_hf.py, distillbert_hf_gate.py
 - filter_factory.py, gate_unfreezer.py (optional, still compatible)

If a given model/backend does not expose the exact helper attributes, the code branches defensively (hasattr checks)
so it will not crash and will simply skip the optional hook.

Step 2/3 (PCA/SVD denoising) will be added later in a separate patch.
"""

import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# === Optional, only if you use HF datasets for textcls (SST-2) ===
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

# === Project modules (present in repo) ===
from dca import DCA
from dca_derivative import DCADerivative
from dca_e_ctr import build_e_ctr
from dca_e_ctr_resmix import build_e_ctr_resmix

from filter_lp_fixed import make_lowpass_fixed
from feature_fft_mask import FeatureFFTMask1D
from spec_reg import spectral_penalty_depthwise
from soft_resmix import SoftResMixNudger

# Models
from distillbert_hf_gate import build_model as build_distillbert_hf_gate
from small_bert import build_model as build_small_bert
from small_transformer import build_model as build_small_transformer

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_insert_layers(arg: str, n_layers: int) -> List[int]:
    if arg.lower() == "last":
        return [n_layers - 1]
    if arg.lower() in ("last2", "last_2", "last-2"):
        return [max(0, n_layers - 2), n_layers - 1]
    if arg.lower() == "all":
        return list(range(n_layers))
    # comma separated indices
    try:
        idxs = [int(x.strip()) for x in arg.split(",") if x.strip() != ""]
        idxs = [i for i in idxs if 0 <= i < n_layers]
        return sorted(set(idxs))
    except Exception:
        return [n_layers - 1]


@dataclass
class TrainState:
    global_step: int = 0
    best_val_acc: float = -1.0


# -----------------------------------------------------------------------------
# Datasets: SST-2 quick helper (text classification)
# -----------------------------------------------------------------------------

def build_sst2_loaders(batch_size: int, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    assert load_dataset is not None, "datasets library not available; install `datasets` or plug in your own loader."
    ds = load_dataset("glue", "sst2")
    # Minimal tokenizer pipeline lives inside model (HF DistilBERT)
    train = list(zip(ds['train']['sentence'], ds['train']['label']))
    val   = list(zip(ds['validation']['sentence'], ds['validation']['label']))

    # Simple on-the-fly batch tokenize in collate_fn handled by model wrapper
    return train, val


# -----------------------------------------------------------------------------
# Model factory & wrappers
# -----------------------------------------------------------------------------

class TextClsWrapper(nn.Module):
    """A minimal interface expected by this trainer.
    The underlying `build_*` functions should return a module exposing:
    - forward(batch) -> dict(logits=..., loss=optional)
    - maybe attributes for gates/resmix/filter hooks (duck-typed)
    - a `collate_fn(list_of_(text,label))` for on-the-fly tokenization (only for HF models)
    """
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, batch):
        return self.core(batch)

    def collate_fn(self, samples):
        if hasattr(self.core, 'collate_fn'):
            return self.core.collate_fn(samples)
        # fallback: samples are already tensors
        xs, ys = zip(*samples)
        return xs, torch.tensor(ys)

    # Duck-typed query points used by DCA/ResMix nudger
    @property
    def n_layers(self) -> int:
        return getattr(self.core, 'n_layers', 1)

    @property
    def layer_modules(self) -> List[nn.Module]:
        return getattr(self.core, 'layer_modules', [])

    @property
    def resmix_modules(self) -> List[nn.Module]:
        return getattr(self.core, 'resmix_modules', [])


def build_model_by_name(name: str, num_labels: int, args) -> TextClsWrapper:
    name = name.lower()
    if name == 'distillbert_hf_gate':
        m = build_distillbert_hf_gate(num_labels=num_labels, args=args)
    elif name == 'small_bert':
        m = build_small_bert(num_labels=num_labels, args=args)
    elif name == 'small_transformer':
        m = build_small_transformer(num_labels=num_labels, args=args)
    else:
        raise ValueError(f"Unknown model: {name}")
    return TextClsWrapper(m)


# -----------------------------------------------------------------------------
# Trainer core
# -----------------------------------------------------------------------------

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(-1)
    return (preds == labels).float().mean().item()


def run_epoch(model: TextClsWrapper,
              loader: DataLoader,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              state: TrainState,
              args,
              dca: Optional[DCA] = None,
              dca_drv: Optional[DCADerivative] = None,
              train: bool = True) -> Dict[str, float]:

    model.train(train)
    tot_loss, tot_acc, n = 0.0, 0.0, 0

    for step, batch in enumerate(loader):
        if callable(getattr(model, 'collate_fn', None)):
            batch = model.collate_fn(batch)
        # move to device (HF wrapper returns dict tensors already on CPU)
        if isinstance(batch, dict):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        else:
            x, y = batch
            if torch.is_tensor(x): x = x.to(device)
            y = y.to(device)
            batch = (x, y)

        if train:
            optimizer.zero_grad(set_to_none=True)

        out = model(batch)
        logits = out['logits']
        labels = out.get('labels')
        loss = out.get('loss')
        if loss is None:
            assert labels is not None, "Either model provides loss, or labels must be present in batch."
            loss = F.cross_entropy(logits, labels)

        acc = accuracy_from_logits(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        tot_loss += loss.item() * bs
        tot_acc += acc * bs
        n += bs

        state.global_step += int(train)

        if train and args.log_every > 0 and state.global_step % args.log_every == 0:
            print(f"[S{state.global_step}] train loss={tot_loss/n:.4f} acc={tot_acc/n:.4f}")

        # quickval & architecture step lives in outer loop (not here)

    return {"loss": tot_loss / max(1, n), "acc": tot_acc / max(1, n)}


@torch.no_grad()
def evaluate(model: TextClsWrapper, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for batch in loader:
        if callable(getattr(model, 'collate_fn', None)):
            batch = model.collate_fn(batch)
        if isinstance(batch, dict):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        else:
            x, y = batch
            if torch.is_tensor(x): x = x.to(device)
            y = y.to(device)
            batch = (x, y)
        out = model(batch)
        logits = out['logits']
        labels = out.get('labels')
        loss = out.get('loss')
        if loss is None:
            assert labels is not None
            loss = F.cross_entropy(logits, labels)
        acc = accuracy_from_logits(logits, labels)
        bs = labels.size(0)
        tot_loss += loss.item() * bs
        tot_acc += acc * bs
        n += bs
    return {"loss": tot_loss / max(1, n), "acc": tot_acc / max(1, n)}


# -----------------------------------------------------------------------------
# Quickval-driven architecture step (DCA + Derivative + optional ResMix nudging)
# -----------------------------------------------------------------------------

def quickval_arch_step(model: TextClsWrapper,
                       val_loader: DataLoader,
                       device: torch.device,
                       args,
                       dca: Optional[DCA],
                       dca_drv: Optional[DCADerivative],
                       resmix_nudger: Optional[SoftResMixNudger]):
    """Run a small validation slice, compute DCA/E-CTR losses, derivative gates,
    and (denoise-only) *scalings* for spectral regularization / ResMix nudging.

    IMPORTANT: This step **must not** perform train→valid compensation. Only
    denoising/regularization is allowed.
    """
    model.eval()

    # 1) Collect a small batch (arch_val_batches) for quickval
    batches = []
    it = iter(val_loader)
    for _ in range(max(1, args.arch_val_batches)):
        try:
            b = next(it)
        except StopIteration:
            break
        if callable(getattr(model, 'collate_fn', None)):
            b = model.collate_fn(b)
        if isinstance(b, dict):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        else:
            x, y = b
            if torch.is_tensor(x): x = x.to(device)
            y = y.to(device)
            b = (x, y)
        batches.append(b)
    if not batches:
        return

    # 2) Forward & compute validation CE as proxy
    ce_vals = []
    for b in batches:
        out = model(b)
        logits = out['logits']
        labels = out.get('labels')
        if labels is None:
            raise RuntimeError("Validation batch must include labels for proxy CE.")
        ce_vals.append(F.cross_entropy(logits, labels, reduction='mean'))
    proxy_ce = torch.stack(ce_vals).mean()

    # 3) DCA alpha-loss (e-ctr or e-ctr-resmix) — acts as *regularizer*, not compensation
    L_alpha = torch.tensor(0.0, device=device)
    if args.dca_enable:
        if args.dca_mode == 'e_ctr':
            L_alpha = build_e_ctr(dca).loss()
        elif args.dca_mode == 'e_ctr_resmix':
            L_alpha = build_e_ctr_resmix(dca).loss()
        else:
            pass

    # 4) Derivative gate (EMA of proxy CE); produces a weight in [0,1]
    drv_weight = 1.0
    if args.dca_derivative_enable and dca_drv is not None:
        drv_weight = dca_drv.update_and_weight(proxy_ce, mode=args.dca_drv_mode,
                                              lam=args.dca_drv_lambda,
                                              kappa=args.dca_drv_kappa)

    # 5) Combine: scaled alpha-loss (denoise-only)
    arch_loss = (1.0 + args.dca_beta * float(drv_weight)) * L_alpha

    # 6) Optional spectral regularization (on controller kernels)
    if args.spec_reg_enable:
        arch_loss = arch_loss + spectral_penalty_depthwise(model, lam=args.spec_reg_lambda,
                                                           power=args.spec_reg_power)

    # 7) Backprop into architecture/gates only if they are learnable (no compensation)
    if arch_loss.requires_grad and float(arch_loss) != 0.0:
        arch_loss.backward()
        # NOTE: caller should have zero_grad on the specific optimizers

    # 8) Optional ResMix nudging (denoise-only): use drv_weight as *magnitude gate*
    if resmix_nudger is not None and args.resmix_from_dca_lr > 0:
        resmix_nudger.nudge(step_size=args.resmix_from_dca_lr * float(drv_weight))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()

    # Task / data
    p.add_argument('--task', type=str, default='textcls')
    p.add_argument('--text_dataset', type=str, default='sst2')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)

    # Model
    p.add_argument('--model', type=str, default='distillbert_hf_gate')
    p.add_argument('--num_labels', type=int, default=2)

    # Insert-layer policy (applied by model builders)
    p.add_argument('--insert_layers', type=str, default='last',
                  help='Which layers to attach gates/filters: last|last2|all|comma list (e.g., 0,2,4)')

    # Optim
    p.add_argument('--weight_lr', type=float, default=3e-4)
    p.add_argument('--adam_eps', type=float, default=1e-8)
    p.add_argument('--adam_wd', type=float, default=0.0)

    # Gating / ResMix
    p.add_argument('--gate_enable', action='store_true')
    p.add_argument('--resmix_enable', action='store_true')
    p.add_argument('--resmix_from_dca_lr', type=float, default=0.0,
                  help='External nudging step for SoftResMix beta (denoise-only).')

    # Filters (denoise-only)
    p.add_argument('--filter_backend', type=str, default='lp_fixed', choices=['lp_fixed','feature_mask','none'])
    p.add_argument('--lp_k', type=int, default=5)
    p.add_argument('--spec_reg_enable', action='store_true')
    p.add_argument('--spec_reg_lambda', type=float, default=1e-4)
    p.add_argument('--spec_reg_power', type=float, default=1.0)

    # DCA (meta-graph)
    p.add_argument('--dca_enable', action='store_true')
    p.add_argument('--dca_mode', type=str, default='e_ctr', choices=['e_ctr','e_ctr_resmix'])
    p.add_argument('--dca_beta', type=float, default=1.0)
    p.add_argument('--dca_w_lr', type=float, default=0.0)

    # DCA Derivative (EMA)
    p.add_argument('--dca_derivative_enable', action='store_true')
    p.add_argument('--dca_drv_mode', type=str, default='ema', choices=['ema'])
    p.add_argument('--dca_drv_lambda', type=float, default=0.2)
    p.add_argument('--dca_drv_kappa', type=float, default=5.0)

    # Logging / eval cadence
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--quickval_every', type=int, default=400)
    p.add_argument('--arch_val_batches', type=int, default=2)

    # IO
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--run_tag', type=str, default='run')
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--save_name', type=str, default='model.pt')

    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    if args.task == 'textcls' and args.text_dataset.lower() == 'sst2':
        train_raw, val_raw = build_sst2_loaders(args.batch)
        # Build loaders after model, because we may need collate_fn from model wrapper
    else:
        raise ValueError('Only textcls/sst2 quick helper provided in this refactor. Plug your own loaders otherwise.')

    # Model
    model = build_model_by_name(args.model, num_labels=args.num_labels, args=args).to(device)

    # Resolve insert_layers indices if model exposes n_layers
    n_layers = getattr(model, 'n_layers', 1)
    args._insert_ids = parse_insert_layers(args.insert_layers, n_layers)

    # Optimizer (single weights opt here; architecture/gates should own their own params internally if needed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.weight_lr, eps=args.adam_eps, weight_decay=args.adam_wd)

    # DCA / Derivative
    dca = DCA(model=model.core) if args.dca_enable else None
    dca_drv = DCADerivative() if args.dca_derivative_enable else None

    # ResMix external nudger (denoise-only)
    resmix_nudger = SoftResMixNudger(model.core) if args.resmix_enable and args.resmix_from_dca_lr > 0 else None

    # Dataloaders (build now to get collate_fn)
    collate = getattr(model, 'collate_fn', None)
    train_loader = DataLoader(train_raw, batch_size=args.batch, shuffle=True, num_workers=2,
                              collate_fn=collate if callable(collate) else None, drop_last=True)
    val_loader = DataLoader(val_raw, batch_size=args.batch, shuffle=False, num_workers=2,
                            collate_fn=collate if callable(collate) else None)

    state = TrainState()

    for epoch in range(1, args.epochs + 1):
        # Train
        m = run_epoch(model, train_loader, optimizer, device, state, args, dca=dca, dca_drv=dca_drv, train=True)
        print(f"[E{epoch:02d}] train loss={m['loss']:.4f} acc={m['acc']:.4f}")

        # Quickval-driven arch step (denoise-only)
        if args.quickval_every > 0 and state.global_step % args.quickval_every == 0:
            optimizer.zero_grad(set_to_none=True)
            quickval_arch_step(model, val_loader, device, args, dca=dca, dca_drv=dca_drv, resmix_nudger=resmix_nudger)
            # NOTE: If some gates have separate optimizers, step them here.
            for p in model.parameters():
                if p.grad is not None and not p.requires_grad:
                    p.grad = None

        # Full eval
        vm = evaluate(model, val_loader, device)
        print(f"[E{epoch:02d}] valid loss={vm['loss']:.4f} acc={vm['acc']:.4f}")

        # Save best
        os.makedirs(args.save_dir, exist_ok=True)
        if vm['acc'] > state.best_val_acc:
            state.best_val_acc = vm['acc']
            torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': epoch},
                       os.path.join(args.save_dir, args.save_name))
            print(f"[E{epoch:02d}] saved best to {os.path.join(args.save_dir, args.save_name)}")


if __name__ == '__main__':
    main()

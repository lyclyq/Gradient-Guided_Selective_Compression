#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: pca_overlap.py
----------------------
Maintains train-side EMA PCA bases (U_tr), and computes train-vs-valid overlap scores.
This module does **not** perform compensation — only overlap/denoise analysis.

Usage:
  pca_mgr = PCAOverlapManager(k=32, ema_decay=0.99)
  pca_mgr.update_train(layer_id, X_btC)   # update with train activations [B,T,C]
  scores = pca_mgr.overlap_scores(layer_id, Xval_btC)  # dict with O_i[], O_res

Notes:
- Train PCA is maintained as EMA covariance sketch (cov = (1-α)cov + αXᵀX)
- At quickval, valid batch is projected to train basis to ensure alignment
- Returns per-component overlap scores O_i (0..1, higher=more overlap)
- Residual overlap O_res summarises the tail components (C-k)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class PCAOverlapManager:
    def __init__(self, k: int = 32, ema_decay: float = 0.99, device: str = 'cuda'):
        self.k = k
        self.decay = ema_decay
        self.device = device
        # state: per-layer EMA mean & cov sketch
        self.layer_state: Dict[int, Dict[str, torch.Tensor]] = {}

    def update_train(self, layer_id: int, X: torch.Tensor):
        """Update EMA PCA stats from train activations [B,T,C]."""
        if X.dim() == 3:
            B,T,C = X.shape
            Xf = X.reshape(B*T, C)
        else:
            Xf = X
        Xf = Xf.to(self.device)

        mu = Xf.mean(0, keepdim=True) # [1,C]
        Xc = Xf - mu
        cov = Xc.t() @ Xc / max(1, Xc.size(0))

        st = self.layer_state.get(layer_id, None)
        if st is None:
            self.layer_state[layer_id] = {
                'mu': mu.detach(),
                'cov': cov.detach()
            }
        else:
            st['mu'] = (1-self.decay)*mu.detach() + self.decay*st['mu']
            st['cov'] = (1-self.decay)*cov.detach() + self.decay*st['cov']

    def _eigvecs(self, cov: torch.Tensor, k: int) -> torch.Tensor:
        # top-k eigenvectors of covariance (symmetric)
        evals, evecs = torch.linalg.eigh(cov)
        idx = torch.argsort(evals, descending=True)
        return evecs[:, idx[:k]]

    @torch.no_grad()
    def overlap_scores(self, layer_id: int, Xval: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute overlap scores O_i and O_res using stored train PCA basis.
        Xval: [B,T,C]
        """
        if Xval.dim() == 3:
            B,T,C = Xval.shape
            Xv = Xval.reshape(B*T, C).to(self.device)
        else:
            Xv = Xval.to(self.device)

        st = self.layer_state.get(layer_id, None)
        if st is None:
            return {"O_i": torch.ones(self.k, device=self.device), "O_res": torch.tensor(1.0, device=self.device)}

        mu_tr, cov_tr = st['mu'], st['cov']
        U_tr = self._eigvecs(cov_tr, self.k)  # [C,k]

        # project train and valid into train basis
        # (train stats approximated by EMA, so use mu_tr only)
        Xv_c = Xv - mu_tr
        Z_val = Xv_c @ U_tr   # [N,k]
        # train stats in this basis (approx)
        # to avoid storing raw train data, approximate as cov diag
        evals, _ = torch.linalg.eigh(cov_tr)
        idx = torch.argsort(evals, descending=True)
        var_tr = evals[idx[:self.k]].clamp(min=1e-6)

        # compute valid variances and means in this basis
        mu_val = Z_val.mean(0)
        var_val = Z_val.var(0, unbiased=False) + 1e-6

        # per-comp score: penalize mean/var mismatch
        diff_mu = (mu_val.abs() / (var_tr.sqrt()+1e-6)).clamp(max=10.0)
        diff_var = (torch.log(var_val) - torch.log(var_tr)).abs()
        noise_score = diff_mu + diff_var
        O_i = torch.exp(-noise_score)

        # residual overlap: fraction of energy captured by top-k
        total_energy = evals.sum().clamp(min=1e-6)
        head_energy = var_tr.sum()
        frac = head_energy / total_energy
        O_res = frac  # if frac~1.0 -> residual small (good)

        return {"O_i": O_i, "O_res": O_res}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: overfit_guard.py
------------------------
A small utility to detect the "overfitting direction" and gate actions with patience.
Signals (streamed): train proxy (e.g., CE) and valid proxy (CE) per quickval step.

Arming rule (default):
  - compute EMA derivatives d_train, d_val (w.r.t. quickval steps)
  - consider overfitting when (d_val > +eps) and (d_train <= 0)
  - require this to hold for `patience` consecutive ticks to ARM

Returns:
  armed: bool        # whether to enable denoise this round
  weight: float      # a soft gate in [0,1], increasing with how strong the condition is

Notes:
- This guard does not implement any compensation; only gating.
- You can use `weight` to scale L_alpha or ResMix nudging (step size).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math

class EMA:
    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.decay * self.value + (1 - self.decay) * float(x)
        return self.value

@dataclass
class OverfitGuardConfig:
    ema_decay: float = 0.9      # EMA for loss
    ema_deriv_decay: float = 0.9  # EMA for finite diff derivatives
    eps: float = 1e-5           # positive threshold for d_val
    patience: int = 2           # consecutive counts to arm
    kappa: float = 5.0          # sharpness for sigmoid soft gate
    tau: float = 0.0            # center for sigmoid on d_val (usually 0)
    clip: float = 1.0           # cap the soft weight

class OverfitGuard:
    def __init__(self, cfg: OverfitGuardConfig = OverfitGuardConfig()):
        self.cfg = cfg
        self.train_ema = EMA(cfg.ema_decay)
        self.val_ema = EMA(cfg.ema_decay)
        self.d_train_ema = EMA(cfg.ema_deriv_decay)
        self.d_val_ema = EMA(cfg.ema_deriv_decay)
        self.prev_train: Optional[float] = None
        self.prev_val: Optional[float] = None
        self._streak: int = 0

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def tick(self, train_proxy: float, val_proxy: float) -> Tuple[bool, float, dict]:
        """
        Push one observation pair (train CE, valid CE) and get arming decision.
        Returns (armed, weight, extras)
        """
        te = self.train_ema.update(train_proxy)
        ve = self.val_ema.update(val_proxy)

        # finite-difference derivatives (EMA-smoothed)
        if self.prev_train is None:
            dtr = 0.0
        else:
            dtr = te - self.prev_train
        if self.prev_val is None:
            dva = 0.0
        else:
            dva = ve - self.prev_val
        self.prev_train = te
        self.prev_val = ve

        dtr_e = self.d_train_ema.update(dtr)
        dva_e = self.d_val_ema.update(dva)

        # overfit condition
        cond = (dva_e > self.cfg.eps) and (dtr_e <= 0.0)
        if cond:
            self._streak += 1
        else:
            self._streak = 0

        armed = (self._streak >= self.cfg.patience)

        # soft weight ∈ [0, clip], depends on d_val magnitude; zeroed if not cond
        soft = self._sigmoid(self.cfg.kappa * (dva_e - self.cfg.tau))
        weight = min(self.cfg.clip, soft) if cond else 0.0

        extras = {
            'train_ema': te,
            'val_ema': ve,
            'd_train_ema': dtr_e,
            'd_val_ema': dva_e,
            'streak': self._streak,
        }
        return armed, weight, extras

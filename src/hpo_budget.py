# src/hpo_budget.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BudgetSpec:
    """
    Budget can be specified in either:
      - total_trials (int): total number of seed-runs allowed (each seed-run counts)
      - total_gpu_hours (float): optional; convert to trials via cost model if provided
    """
    total_trials: Optional[int] = None
    total_gpu_hours: Optional[float] = None


@dataclass(frozen=True)
class CostModel:
    """
    Convert gpu-hours to trials.
    trials_per_gpu_hour is a coarse factor; calibrate once and reuse.
    """
    trials_per_gpu_hour: float = 50.0


def _require_dict(x: Any, path: str) -> Dict[str, Any]:
    if not isinstance(x, dict):
        raise TypeError(f"[hpo_budget] '{path}' must be dict, got {type(x)}")
    return x


def _require_list(x: Any, path: str) -> List[Any]:
    if not isinstance(x, list):
        raise TypeError(f"[hpo_budget] '{path}' must be list, got {type(x)}")
    return x


def _logspace(min_lr: float, max_lr: float, points: int) -> List[float]:
    min_lr = float(min_lr)
    max_lr = float(max_lr)
    points = int(points)
    if points <= 1:
        return [min_lr]
    xs = np.logspace(np.log10(min_lr), np.log10(max_lr), num=points).astype(float).tolist()
    out: List[float] = []
    for x in xs:
        x = float(x)
        if x not in out:
            out.append(x)
    return out


def _extend_lr_range(
    min_lr: float,
    max_lr: float,
    extend_factor: float,
    clamp_min: float,
    clamp_max: float,
) -> Tuple[float, float]:
    extend_factor = float(max(1.0, extend_factor))
    mn = float(min_lr) / extend_factor
    mx = float(max_lr) * extend_factor
    mn = max(float(clamp_min), mn)
    mx = min(float(clamp_max), mx)
    if mn > mx:
        mn, mx = mx, mn
    return mn, mx


def derive_total_trials(cfg: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Read budget from cfg.hpo.budget:
      - total_trials OR total_gpu_hours + trials_per_gpu_hour
    Returns: total_trials (int), meta
    """
    hpo = _require_dict(cfg["hpo"], "hpo")
    bud = _require_dict(hpo["budget"], "hpo.budget")

    total_trials = bud.get("total_trials", None)
    total_gpu_hours = bud.get("total_gpu_hours", None)
    trials_per_gpu_hour = None

    meta = {"source": None, "trials_per_gpu_hour": None}

    if total_trials is not None:
        meta["source"] = "total_trials"
        cm = bud.get("cost_model", None)
        if isinstance(cm, dict) and "trials_per_gpu_hour" in cm:
            meta["trials_per_gpu_hour"] = float(cm["trials_per_gpu_hour"])
        return max(1, int(total_trials)), meta

    if total_gpu_hours is not None:
        cm = _require_dict(bud.get("cost_model", None), "hpo.budget.cost_model")
        trials_per_gpu_hour = float(cm["trials_per_gpu_hour"])
        meta["source"] = "total_gpu_hours"
        meta["trials_per_gpu_hour"] = trials_per_gpu_hour
        est = float(total_gpu_hours) * trials_per_gpu_hour
        return max(1, int(math.floor(est))), meta

    raise RuntimeError("[hpo_budget] missing hpo.budget.total_trials or total_gpu_hours")


def _alpha_probe_meta(cfg: Dict[str, Any], refine_seeds: List[int]) -> Dict[str, Any]:
    """
    alpha_probe meta (NOT counted in budget):
      default: 2 alpha × 2 lr × 2 seeds = 8 seed-runs
    """
    hpo = _require_dict(cfg["hpo"], "hpo")
    grid = _require_dict(hpo["grid"], "hpo.grid")
    ap = _require_dict(grid["alpha_probe"], "hpo.grid.alpha_probe")

    enabled = bool(ap["enabled"])
    num_alpha = int(ap["num_alpha"])
    num_lr = int(ap["num_lr"])
    seeds2 = [int(x) for x in _require_list(ap["seeds"], "hpo.grid.alpha_probe.seeds")]
    if len(seeds2) < 2:
        raise RuntimeError("[hpo_budget] hpo.grid.alpha_probe.seeds must have at least 2 seeds")

    n_seeds = int(len(seeds2))
    trials = (num_alpha * num_lr * n_seeds) if enabled else 0

    return {
        "enabled": enabled,
        "num_alpha": int(num_alpha),
        "num_lr": int(num_lr),
        "seeds": seeds2,
        "n_seeds": int(n_seeds),
        "alpha_probe_trials": int(trials),
        "counted_in_budget": False,
    }


def _baseline_tags(cfg: Dict[str, Any]) -> List[str]:
    """
    STRICT: cfg.hpo.baseline_variants is list of dicts with only {tag: baseline_r|baseline_R}
    """
    hpo = _require_dict(cfg["hpo"], "hpo")
    variants = _require_list(hpo["baseline_variants"], "hpo.baseline_variants")
    tags: List[str] = []
    for i, v in enumerate(variants):
        if not isinstance(v, dict) or "tag" not in v:
            raise ValueError(f"[hpo_budget] baseline_variants[{i}] must be dict with key 'tag'")
        tag = str(v["tag"])
        if tag not in {"baseline_r", "baseline_R"}:
            raise ValueError(f"[hpo_budget] baseline_variants[{i}].tag must be baseline_r/baseline_R, got {tag}")
        tags.append(tag)
    if not tags:
        raise ValueError("[hpo_budget] baseline_variants empty")
    # stable unique
    out: List[str] = []
    seen = set()
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def build_grid_plan(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an HPO plan that:
      - accounts baseline sweep + baseline refine as budget consumption (DEDUCTED)
      - alpha_probe is NOT counted (as you requested)
      - remaining budget split by ratios (sens/grid/bayes)
      - auto-derives grid density (configs per seed) from grid_trials
    """
    hpo = _require_dict(cfg["hpo"], "hpo")
    bandit = _require_dict(hpo["bandit"], "hpo.bandit")

    total_trials, meta = derive_total_trials(cfg)
    use_bayes = bool(hpo["use_bayes"])

    ratio_sens = float(hpo["ratio_sensitivity"])
    ratio_grid = float(hpo["ratio_grid"])
    ratio_bayes = float(hpo["ratio_bayes"])
    if not use_bayes:
        ratio_grid = ratio_grid + ratio_bayes
        ratio_bayes = 0.0

    refine_seeds = [int(x) for x in _require_list(bandit["refine_seeds"], "hpo.bandit.refine_seeds")]
    n_seeds = max(1, len(refine_seeds))

    # lr grid meta
    lr_grid = _require_dict(hpo["lr_grid"], "hpo.lr_grid")
    min_lr = float(lr_grid["min_lr"])
    max_lr = float(lr_grid["max_lr"])
    clamp_min = float(lr_grid["clamp_min_lr"])
    clamp_max = float(lr_grid["clamp_max_lr"])
    extend_factor = float(lr_grid["extend_factor"])
    min_lr_ext, max_lr_ext = _extend_lr_range(min_lr, max_lr, extend_factor, clamp_min, clamp_max)

    baseline_points = int(lr_grid["baseline_points"])
    neighbor_points = int(lr_grid["neighbor_points"])

    tags = _baseline_tags(cfg)

    # baseline cost model: sweep uses 1 fixed seed (seed=12) per lr per tag
    baseline_sweep_trials = int(len(tags) * baseline_points * 1)

    # baseline refine: neighbor_points per tag per refine_seed
    baseline_refine_trials = int(len(tags) * neighbor_points * n_seeds)

    alpha_probe_meta = _alpha_probe_meta(cfg, refine_seeds=refine_seeds)
    alpha_probe_trials = int(alpha_probe_meta.get("alpha_probe_trials", 0))  # NOT deducted

    # remaining for sens/grid/bayes
    remaining = int(max(1, total_trials - baseline_sweep_trials - baseline_refine_trials))

    sens_trials = max(1, int(math.floor(remaining * ratio_sens)))
    grid_trials = max(1, int(math.floor(remaining * ratio_grid)))
    bayes_trials = max(0, int(math.floor(remaining * ratio_bayes)))

    # per-config count in grid/bayes (each config evaluated on n_seeds)
    grid_configs = max(1, int(grid_trials // n_seeds))
    bayes_configs = max(0, int(bayes_trials // n_seeds))

    # for reporting and optional tuning
    plan = {
        "budget": {
            "total_trials": int(total_trials),
            "meta": meta,
            "use_bayes": bool(use_bayes),
            "deducted": {
                "baseline_sweep_trials": int(baseline_sweep_trials),
                "baseline_refine_trials": int(baseline_refine_trials),
            },
            "remaining_after_baseline": int(remaining),
            "alloc": {
                "sensitivity_trials": int(sens_trials),
                "grid_trials": int(grid_trials),
                "bayes_trials": int(bayes_trials),
                "n_seeds_refine": int(n_seeds),
                "grid_configs": int(grid_configs),
                "bayes_configs": int(bayes_configs),
            },
            "not_counted": {
                "alpha_probe": alpha_probe_meta,
            },
        },
        "lr": {
            "min_lr": float(min_lr),
            "max_lr": float(max_lr),
            "min_lr_ext": float(min_lr_ext),
            "max_lr_ext": float(max_lr_ext),
            "baseline_points": int(baseline_points),
            "neighbor_points": int(neighbor_points),
            "clamp_min_lr": float(clamp_min),
            "clamp_max_lr": float(clamp_max),
            "extend_factor": float(extend_factor),
        },
        "stages": [
            {"name": "baseline_sweep", "counted_in_budget": True, "budget_trials": int(baseline_sweep_trials)},
            {"name": "baseline_refine", "counted_in_budget": True, "budget_trials": int(baseline_refine_trials)},
            {"name": "sensitivity", "counted_in_budget": True, "budget_trials": int(sens_trials)},
            {"name": "alpha_probe", "counted_in_budget": False, "meta": alpha_probe_meta},
            {"name": "grid2", "counted_in_budget": True, "budget_trials": int(grid_trials), "configs": int(grid_configs)},
            {"name": "bayes_refine", "counted_in_budget": bool(use_bayes), "budget_trials": int(bayes_trials), "configs": int(bayes_configs)},
        ],
    }
    return plan


def make_lr_candidates_from_plan(plan: Dict[str, Any]) -> List[float]:
    lr = _require_dict(plan["lr"], "plan.lr")
    mn = float(lr["min_lr_ext"])
    mx = float(lr["max_lr_ext"])
    pts = int(lr["baseline_points"])
    return _logspace(mn, mx, pts)


def lr_neighborhood_from_plan(plan: Dict[str, Any], best_lr: float) -> List[float]:
    lr = _require_dict(plan["lr"], "plan.lr")
    mn = float(lr["min_lr_ext"])
    mx = float(lr["max_lr_ext"])
    pts = int(lr["baseline_points"])
    neighbor = int(lr["neighbor_points"])

    full = _logspace(mn, mx, pts)
    arr = np.array(full, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(best_lr))))

    radius = (neighbor - 1) // 2
    lo = max(0, idx - radius)
    hi = min(len(full), idx + radius + 1)
    cand = [float(x) for x in full[lo:hi]]

    while len(cand) < neighbor and lo > 0:
        lo -= 1
        cand.insert(0, float(full[lo]))
    while len(cand) < neighbor and hi < len(full):
        cand.append(float(full[hi]))
        hi += 1

    out: List[float] = []
    for x in cand:
        if x not in out:
            out.append(float(x))
    return out

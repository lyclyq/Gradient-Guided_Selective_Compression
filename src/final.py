# src/final.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .artifacts import prepare_run_dir, dump_json, make_run_name
from .runner import run_train


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_latest_best(io_root: str) -> Optional[Path]:
    root = Path(io_root)
    if not root.exists():
        return None
    cands: List[Path] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        p = d / "best_hparams.json"
        if p.exists():
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _extract_best_trial_cfg(best_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected format from hpo.py:
      {"best": { ..., "trial_cfg_json": "<json string>" }, "weights": {...}}
    """
    best = best_json.get("best", {})
    raw = best.get("trial_cfg_json", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def run_final(cfg: Dict[str, Any], best_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Final evaluation:
    - load best_hparams.json
    - apply best trial's method/ours + lr (but keep final schedule epochs)
    - run multiple seeds (cfg.seeds.final)
    - write final_summary.json (mean/std + per-seed)
    """
    io_root = str(cfg.get("io", {}).get("root", "runs"))

    best_p = Path(best_path) if best_path else _find_latest_best(io_root)
    if best_p is None or not best_p.exists():
        raise FileNotFoundError("best_hparams.json not found. Provide --best or run HPO first.")

    best_json = _read_json(best_p)
    best_trial_cfg = _extract_best_trial_cfg(best_json)
    if not best_trial_cfg:
        raise ValueError(f"Cannot parse best trial config from {best_p}")

    # apply best hyperparams onto current cfg
    # keep final epochs/warmup from schedule, but pull lr + method + ours knobs from best
    best_overlay: Dict[str, Any] = {}

    # method + ours
    if "method" in best_trial_cfg:
        best_overlay["method"] = best_trial_cfg["method"]

    # lr
    best_lr = best_trial_cfg.get("train", {}).get("lr", None)
    if best_lr is not None:
        best_overlay.setdefault("train", {})["lr"] = float(best_lr)

    final_cfg = json.loads(json.dumps(cfg))  # cheap deep copy
    _deep_merge(final_cfg, best_overlay)

    # seeds
    seeds_block = final_cfg.get("seeds", {}) if isinstance(final_cfg.get("seeds", {}), dict) else {}
    final_seeds = seeds_block.get("final", None)
    if not final_seeds:
        # fallback
        final_seeds = [2, 3, 5, 7, 11]
    final_seeds = [int(x) for x in final_seeds]

    # group dir under io.root
    group_name = f"final__from_{best_p.parent.name}__{time.strftime('%Y%m%d-%H%M%S')}"
    group_dir, _ = prepare_run_dir(root=io_root, run_name=group_name, overwrite=final_cfg.get("io", {}).get("overwrite", "ask"))

    # force children to write under group_dir
    final_cfg.setdefault("io", {})["root"] = str(group_dir)

    # persist what we actually run
    dump_json(group_dir / "config_final_resolved.json", final_cfg)
    dump_json(group_dir / "best_source.json", {"best_path": str(best_p), "best_row": best_json.get("best", {})})

    per_seed: List[Dict[str, Any]] = []
    vals_max, vals_final, vals_avg = [], [], []

    for s in final_seeds:
        # run_train uses train.seed
        seed_cfg = json.loads(json.dumps(final_cfg))
        seed_cfg.setdefault("train", {})["seed"] = int(s)

        summary = run_train(seed_cfg, trial_tag=f"final_seed{s}")
        per_seed.append({"seed": s, "summary": summary})

        vals_max.append(float(summary.get("val_max", summary.get("best_val_acc", -1.0))))
        vals_final.append(float(summary.get("val_final", vals_max[-1])))
        vals_avg.append(float(summary.get("val_avg", vals_max[-1])))

    out = {
        "best_path": str(best_p),
        "group_dir": str(group_dir),
        "seeds": final_seeds,
        "val_max_mean": float(np.mean(vals_max)) if vals_max else None,
        "val_max_std": float(np.std(vals_max)) if vals_max else None,
        "val_final_mean": float(np.mean(vals_final)) if vals_final else None,
        "val_final_std": float(np.std(vals_final)) if vals_final else None,
        "val_avg_mean": float(np.mean(vals_avg)) if vals_avg else None,
        "val_avg_std": float(np.std(vals_avg)) if vals_avg else None,
        "per_seed": per_seed,
    }
    dump_json(group_dir / "final_summary.json", out)
    print(f"[OK] final summary saved to {group_dir / 'final_summary.json'}")
    return out

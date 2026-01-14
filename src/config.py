# src/config.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _parse_scalar(val: str) -> Any:
    s = val.strip()
    lo = s.lower()
    if lo in {"true", "false"}:
        return lo == "true"
    if lo in {"none", "null"}:
        return None
    try:
        if s.startswith("0") and len(s) > 1 and s[1].isdigit():
            raise ValueError
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def apply_set_overrides(cfg: Dict[str, Any], set_args: Optional[List[str]]) -> Dict[str, Any]:
    if not set_args:
        return cfg
    for kv in set_args:
        if "=" not in kv:
            continue
        key_path, raw_val = kv.split("=", 1)
        val = _parse_scalar(raw_val)
        keys = key_path.split(".")
        cur = cfg
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = val
    return cfg


def resolve_config(
    base_path: str | Path,
    schedule_path: Optional[str | Path] = None,
    set_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cfg = load_yaml(base_path)
    if schedule_path:
        sch = load_yaml(schedule_path)
        cfg = deep_merge(cfg, sch)
    cfg = apply_set_overrides(cfg, set_args)
    validate_config(cfg)
    return cfg


def _validate_staged_hpo(cfg: Dict[str, Any]) -> None:
    hpo = cfg.get("hpo", {})
    if not isinstance(hpo, dict):
        return
    ours = hpo.get("ours", {})
    if not isinstance(ours, dict):
        return
    stages = ours.get("stages", {})
    if not stages:
        return
    if not isinstance(stages, dict):
        raise ValueError("hpo.ours.stages must be a dict")

    for key in ["s1", "s2", "s3"]:
        if key not in stages:
            continue
        s = stages[key]
        if not isinstance(s, dict):
            raise ValueError(f"hpo.ours.stages.{key} must be a dict")
        if "epochs" in s and int(s["epochs"]) <= 0:
            raise ValueError(f"hpo.ours.stages.{key}.epochs must be > 0")
        if "trials" in s and int(s["trials"]) <= 0:
            raise ValueError(f"hpo.ours.stages.{key}.trials must be > 0")

    # score weights sanity
    score = stages.get("s3", {}).get("score", {})
    if isinstance(score, dict):
        w = float(score.get("w_max", 0.5)) + float(score.get("w_final", 0.4)) + float(score.get("w_avg", 0.1))
        if abs(w - 1.0) > 1e-6:
            raise ValueError("hpo.ours.stages.s3.score weights must sum to 1.0")


def validate_config(cfg: Dict[str, Any]) -> None:
    method = cfg.get("method", {}).get("name")
    if method not in {"baseline", "ours"}:
        raise ValueError(f"method.name must be 'baseline' or 'ours', got: {method}")

    lora = cfg.get("method", {}).get("lora", {})
    r, R = int(lora.get("r", 0)), int(lora.get("R", 0))
    if r <= 0 or R <= 0 or R <= r:
        raise ValueError(f"LoRA ranks must satisfy 0 < r < R. Got r={r}, R={R}")

    compute = cfg.get("compute", {})
    if int(compute.get("gpus_per_trial", 1)) <= 0:
        raise ValueError("compute.gpus_per_trial must be >= 1")

    # warmup search is forbidden (must be fixed single value)
    hpo = cfg.get("hpo", {})
    if isinstance(hpo, dict) and "warmup_grid" in hpo:
        wg = hpo.get("warmup_grid")
        if isinstance(wg, list) and len(wg) > 1:
            raise ValueError("warmup_grid length > 1 is disabled (warmup fixed). Set hpo.warmup_grid=[0.06] only.")

    _validate_staged_hpo(cfg)


def dump_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

# /home/lyclyq/Optimization/grad-shake-align/src/config.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# -------------------------
# YAML load / merge
# -------------------------

def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """In-place deep merge: src overwrites dst."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML must be a dict at top-level: {p}")
    return obj


def _parse_scalar(s: str) -> Any:
    """Parse CLI scalar: true/false/null/int/float/json-string fallback."""
    ss = s.strip()
    low = ss.lower()

    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None

    # int
    try:
        if ss.startswith("0") and ss != "0" and not ss.startswith("0."):
            # keep "01" as string
            raise ValueError
        return int(ss)
    except Exception:
        pass

    # float
    try:
        return float(ss)
    except Exception:
        pass

    # json obj/list
    if (ss.startswith("{") and ss.endswith("}")) or (ss.startswith("[") and ss.endswith("]")):
        return json.loads(ss)

    return ss


def _set_by_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """
    Set cfg["a"]["b"]["c"] = value by dotted path "a.b.c"
    """
    keys = [k for k in dotted.split(".") if k]
    if not keys:
        return
    cur: Dict[str, Any] = cfg
    for k in keys[:-1]:
        if k not in cur:
            cur[k] = {}
        if not isinstance(cur[k], dict):
            raise TypeError(f"Cannot set '{dotted}': '{k}' is not a dict (got {type(cur[k])})")
        cur = cur[k]
    cur[keys[-1]] = value


def _apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    overrides: list of "a.b.c=xxx"
    """
    for s in overrides or []:
        if "=" not in s:
            raise ValueError(f"Bad --set override (missing '='): {s}")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Bad --set override (empty key): {s}")
        _set_by_dotted(cfg, k, _parse_scalar(v))
    return cfg


# -------------------------
# Strict getters / validators
# -------------------------

def _require_dict(x: Any, path: str) -> Dict[str, Any]:
    if not isinstance(x, dict):
        raise TypeError(f"Config path '{path}' must be a dict, got {type(x)}")
    return x


def _require_list(x: Any, path: str) -> List[Any]:
    if not isinstance(x, list):
        raise TypeError(f"Config path '{path}' must be a list, got {type(x)}")
    return x


def _get_path(cfg: Dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required config key: '{dotted}' (stuck at '{k}')")
        cur = cur[k]
    return cur


def _ensure_int(x: Any, path: str) -> int:
    if isinstance(x, bool):
        raise TypeError(f"Config '{path}' must be int, got bool")
    try:
        return int(x)
    except Exception as e:
        raise TypeError(f"Config '{path}' must be int-like, got {x} ({type(x)})") from e


def _ensure_float(x: Any, path: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"Config '{path}' must be float-like, got {x} ({type(x)})") from e


def _ensure_bool(x: Any, path: str) -> bool:
    if isinstance(x, bool):
        return x
    raise TypeError(f"Config '{path}' must be bool, got {x} ({type(x)})")


def _ensure_str(x: Any, path: str) -> str:
    if not isinstance(x, str):
        raise TypeError(f"Config '{path}' must be str, got {type(x)}")
    return x


def _validate_lora_block(cfg: Dict[str, Any], dotted: str, *, require_R: bool) -> None:
    """
    Validate a lora block at dotted path:
      - require r (int)
      - optionally require R (int) if require_R=True
      - require alpha (float)
      - require dropout (float)
    """
    _require_dict(_get_path(cfg, dotted), dotted)
    _ensure_int(_get_path(cfg, f"{dotted}.r"), f"{dotted}.r")
    if require_R:
        _ensure_int(_get_path(cfg, f"{dotted}.R"), f"{dotted}.R")
    _ensure_float(_get_path(cfg, f"{dotted}.alpha"), f"{dotted}.alpha")
    _ensure_float(_get_path(cfg, f"{dotted}.dropout"), f"{dotted}.dropout")


def _validate_ours_block(cfg: Dict[str, Any]) -> None:
    """
    Ours block is intentionally strict: if a knob is missing, crash NOW.
    """
    _require_dict(_get_path(cfg, "method.ours"), "method.ours")

    # lora
    _validate_lora_block(cfg, "method.ours.lora", require_R=True)

    # core knobs
    _ensure_int(_get_path(cfg, "method.ours.ema_H"), "method.ours.ema_H")
    _ensure_int(_get_path(cfg, "method.ours.votes"), "method.ours.votes")

    # ablate
    _require_dict(_get_path(cfg, "method.ours.ablate"), "method.ours.ablate")
    _ensure_bool(_get_path(cfg, "method.ours.ablate.interp"), "method.ours.ablate.interp")

    # gate0 noise
    _require_dict(_get_path(cfg, "method.ours.gate0_noise"), "method.ours.gate0_noise")
    _ensure_float(_get_path(cfg, "method.ours.gate0_noise.tau"), "method.ours.gate0_noise.tau")
    _ensure_float(_get_path(cfg, "method.ours.gate0_noise.kappa"), "method.ours.gate0_noise.kappa")

    # trigger
    _require_dict(_get_path(cfg, "method.ours.trigger_gate0"), "method.ours.trigger_gate0")
    _ensure_float(_get_path(cfg, "method.ours.trigger_gate0.tau_N"), "method.ours.trigger_gate0.tau_N")
    _ensure_float(_get_path(cfg, "method.ours.trigger_gate0.tau_D"), "method.ours.trigger_gate0.tau_D")

    # routing / pulling / voting
    _ensure_float(_get_path(cfg, "method.ours.routing_delta"), "method.ours.routing_delta")

    _require_dict(_get_path(cfg, "method.ours.pulling"), "method.ours.pulling")
    _ensure_float(_get_path(cfg, "method.ours.pulling.gamma_pull"), "method.ours.pulling.gamma_pull")
    _ensure_float(_get_path(cfg, "method.ours.pulling.k_pull"), "method.ours.pulling.k_pull")

    _require_dict(_get_path(cfg, "method.ours.voting"), "method.ours.voting")
    _ensure_int(_get_path(cfg, "method.ours.voting.samples_per_vote"), "method.ours.voting.samples_per_vote")
    _ensure_bool(_get_path(cfg, "method.ours.voting.allow_tail"), "method.ours.voting.allow_tail")
    _ensure_bool(_get_path(cfg, "method.ours.voting.keep_single_votes"), "method.ours.voting.keep_single_votes")

    # history
    _require_dict(_get_path(cfg, "method.ours.history"), "method.ours.history")
    _ensure_bool(_get_path(cfg, "method.ours.history.enabled"), "method.ours.history.enabled")
    _ensure_int(_get_path(cfg, "method.ours.history.window_steps"), "method.ours.history.window_steps")
    _ensure_str(_get_path(cfg, "method.ours.history.weighting"), "method.ours.history.weighting")
    _ensure_float(_get_path(cfg, "method.ours.history.exp_beta"), "method.ours.history.exp_beta")

    # compensation (required for shake_align correction)
    _require_dict(_get_path(cfg, "method.ours.compensation"), "method.ours.compensation")
    _ensure_bool(_get_path(cfg, "method.ours.compensation.enabled"), "method.ours.compensation.enabled")
    _ensure_float(_get_path(cfg, "method.ours.compensation.ridge_lambda"), "method.ours.compensation.ridge_lambda")

    # compression
    _require_dict(_get_path(cfg, "method.ours.compression"), "method.ours.compression")
    _ensure_bool(_get_path(cfg, "method.ours.compression.enabled"), "method.ours.compression.enabled")
    _ensure_int(_get_path(cfg, "method.ours.compression.every_steps"), "method.ours.compression.every_steps")
    # target_rank can be null => allow None, else int-like
    tr = _get_path(cfg, "method.ours.compression.target_rank")
    if tr is not None:
        _ensure_int(tr, "method.ours.compression.target_rank")

    _require_dict(_get_path(cfg, "method.ours.compression.qr"), "method.ours.compression.qr")
    _ensure_bool(_get_path(cfg, "method.ours.compression.qr.enabled"), "method.ours.compression.qr.enabled")
    _require_dict(_get_path(cfg, "method.ours.compression.svd"), "method.ours.compression.svd")
    _ensure_bool(_get_path(cfg, "method.ours.compression.svd.enabled"), "method.ours.compression.svd.enabled")


def validate_config(cfg: Dict[str, Any], cmd: str) -> None:
    """
    Strict validation: no defaults, no fallback.
    Raise immediately if any required fields are missing / wrong type.

    cmd: "train" | "resume" | "hpo" | "final" | "plot"
    """
    _require_dict(cfg, "<root>")

    # --- common required roots ---
    _require_dict(_get_path(cfg, "model"), "model")
    _ensure_str(_get_path(cfg, "model.name"), "model.name")

    _require_dict(_get_path(cfg, "task"), "task")
    _ensure_str(_get_path(cfg, "task.name"), "task.name")
    _ensure_int(_get_path(cfg, "task.max_len"), "task.max_len")

    _require_dict(_get_path(cfg, "train"), "train")
    _ensure_int(_get_path(cfg, "train.epochs"), "train.epochs")
    _ensure_int(_get_path(cfg, "train.batch_size"), "train.batch_size")
    _ensure_float(_get_path(cfg, "train.lr"), "train.lr")
    _ensure_float(_get_path(cfg, "train.warmup_ratio"), "train.warmup_ratio")
    _ensure_int(_get_path(cfg, "train.seed"), "train.seed")

    _require_dict(_get_path(cfg, "train.eval"), "train.eval")
    _ensure_str(_get_path(cfg, "train.eval.strategy"), "train.eval.strategy")
    _ensure_int(_get_path(cfg, "train.eval.max_batches"), "train.eval.max_batches")

    # optional-but-typed fields (if present)
    teval = _get_path(cfg, "train.eval")
    if "dense_early_per_epoch" in teval:
        _ensure_int(_get_path(cfg, "train.eval.dense_early_per_epoch"), "train.eval.dense_early_per_epoch")
    if "dense_early_epochs" in teval:
        _ensure_int(_get_path(cfg, "train.eval.dense_early_epochs"), "train.eval.dense_early_epochs")
    if "every_steps" in teval:
        _ensure_int(_get_path(cfg, "train.eval.every_steps"), "train.eval.every_steps")
    if "first_step" in teval:
        _ensure_bool(_get_path(cfg, "train.eval.first_step"), "train.eval.first_step")
    if "compute_train_acc" in teval:
        _ensure_bool(_get_path(cfg, "train.eval.compute_train_acc"), "train.eval.compute_train_acc")
    if "train_max_batches" in teval:
        _ensure_int(_get_path(cfg, "train.eval.train_max_batches"), "train.eval.train_max_batches")
    if "log_r_only" in teval:
        _ensure_bool(_get_path(cfg, "train.eval.log_r_only"), "train.eval.log_r_only")

    _require_dict(_get_path(cfg, "io"), "io")
    _ensure_str(_get_path(cfg, "io.root"), "io.root")
    _ensure_str(_get_path(cfg, "io.overwrite"), "io.overwrite")

    # --- method: STRICT 3-way only ---
    _require_dict(_get_path(cfg, "method"), "method")
    mode = _ensure_str(_get_path(cfg, "method.name"), "method.name")
    if mode not in {"baseline_r", "baseline_R", "ours"}:
        raise ValueError("method.name must be one of: baseline_r / baseline_R / ours")

    # baseline blocks must exist (so you can switch without hidden defaults)
    _require_dict(_get_path(cfg, "method.baseline_r"), "method.baseline_r")
    _validate_lora_block(cfg, "method.baseline_r.lora", require_R=False)

    _require_dict(_get_path(cfg, "method.baseline_R"), "method.baseline_R")
    _validate_lora_block(cfg, "method.baseline_R.lora", require_R=False)

    # ours block must exist + strict knobs
    _validate_ours_block(cfg)

    # --- command-specific ---
    if cmd == "hpo":
        _require_dict(_get_path(cfg, "hpo"), "hpo")
        _require_dict(_get_path(cfg, "hpo.budget"), "hpo.budget")
        _ensure_int(_get_path(cfg, "hpo.budget.total_trials"), "hpo.budget.total_trials")
        _ensure_float(_get_path(cfg, "hpo.ratio_sensitivity"), "hpo.ratio_sensitivity")
        _ensure_float(_get_path(cfg, "hpo.ratio_grid"), "hpo.ratio_grid")
        _ensure_float(_get_path(cfg, "hpo.ratio_bayes"), "hpo.ratio_bayes")
        _ensure_bool(_get_path(cfg, "hpo.use_bayes"), "hpo.use_bayes")

        _require_dict(_get_path(cfg, "hpo.lr_grid"), "hpo.lr_grid")
        _ensure_float(_get_path(cfg, "hpo.lr_grid.min_lr"), "hpo.lr_grid.min_lr")
        _ensure_float(_get_path(cfg, "hpo.lr_grid.max_lr"), "hpo.lr_grid.max_lr")
        _ensure_int(_get_path(cfg, "hpo.lr_grid.baseline_points"), "hpo.lr_grid.baseline_points")
        _ensure_int(_get_path(cfg, "hpo.lr_grid.neighbor_points"), "hpo.lr_grid.neighbor_points")
        _ensure_float(_get_path(cfg, "hpo.lr_grid.extend_factor"), "hpo.lr_grid.extend_factor")
        _ensure_float(_get_path(cfg, "hpo.lr_grid.clamp_min_lr"), "hpo.lr_grid.clamp_min_lr")
        _ensure_float(_get_path(cfg, "hpo.lr_grid.clamp_max_lr"), "hpo.lr_grid.clamp_max_lr")

        _require_dict(_get_path(cfg, "hpo.bandit"), "hpo.bandit")
        _ensure_float(_get_path(cfg, "hpo.bandit.fixed_warmup_ratio"), "hpo.bandit.fixed_warmup_ratio")
        refine_seeds = _require_list(_get_path(cfg, "hpo.bandit.refine_seeds"), "hpo.bandit.refine_seeds")
        if len(refine_seeds) < 2:
            raise ValueError("hpo.bandit.refine_seeds must contain at least 2 seeds")
        for i, s in enumerate(refine_seeds):
            _ensure_int(s, f"hpo.bandit.refine_seeds[{i}]")
        _require_dict(_get_path(cfg, "hpo.bandit.score"), "hpo.bandit.score")
        _ensure_float(_get_path(cfg, "hpo.bandit.score.w_max"), "hpo.bandit.score.w_max")
        _ensure_float(_get_path(cfg, "hpo.bandit.score.w_final"), "hpo.bandit.score.w_final")
        _ensure_float(_get_path(cfg, "hpo.bandit.score.w_avg"), "hpo.bandit.score.w_avg")

        _require_dict(_get_path(cfg, "hpo.grid"), "hpo.grid")
        knobs = _require_list(_get_path(cfg, "hpo.grid.knobs"), "hpo.grid.knobs")
        if not knobs:
            raise ValueError("hpo.grid.knobs must be non-empty")
        for i, k in enumerate(knobs):
            _ensure_str(k, f"hpo.grid.knobs[{i}]")
        _ensure_float(_get_path(cfg, "hpo.grid.drop_if_weight_lt"), "hpo.grid.drop_if_weight_lt")
        _ensure_int(_get_path(cfg, "hpo.grid.sensitivity_epochs"), "hpo.grid.sensitivity_epochs")
        _ensure_int(_get_path(cfg, "hpo.grid.grid_epochs"), "hpo.grid.grid_epochs")
        _ensure_int(_get_path(cfg, "hpo.grid.baseline_sweep_epochs"), "hpo.grid.baseline_sweep_epochs")
        _ensure_int(_get_path(cfg, "hpo.grid.baseline_refine_epochs"), "hpo.grid.baseline_refine_epochs")
        _ensure_int(_get_path(cfg, "hpo.grid.max_retries"), "hpo.grid.max_retries")
        _ensure_int(_get_path(cfg, "hpo.grid.sens_seed"), "hpo.grid.sens_seed")
        _ensure_int(_get_path(cfg, "hpo.grid.rng_seed"), "hpo.grid.rng_seed")
        _ensure_int(_get_path(cfg, "hpo.grid.max_m_float"), "hpo.grid.max_m_float")
        _ensure_int(_get_path(cfg, "hpo.grid.max_m_choice"), "hpo.grid.max_m_choice")
        _ensure_int(_get_path(cfg, "hpo.grid.bayes_rng_seed"), "hpo.grid.bayes_rng_seed")
        _ensure_int(_get_path(cfg, "hpo.grid.bayes_pool"), "hpo.grid.bayes_pool")
        _ensure_float(_get_path(cfg, "hpo.grid.bayes_gp_length"), "hpo.grid.bayes_gp_length")
        _ensure_float(_get_path(cfg, "hpo.grid.bayes_gp_var"), "hpo.grid.bayes_gp_var")
        _ensure_float(_get_path(cfg, "hpo.grid.bayes_gp_noise"), "hpo.grid.bayes_gp_noise")
        _ensure_float(_get_path(cfg, "hpo.grid.bayes_ei_xi"), "hpo.grid.bayes_ei_xi")

        ap = _require_dict(_get_path(cfg, "hpo.grid.alpha_probe"), "hpo.grid.alpha_probe")
        _ensure_bool(_get_path(cfg, "hpo.grid.alpha_probe.enabled"), "hpo.grid.alpha_probe.enabled")
        _ensure_int(_get_path(cfg, "hpo.grid.alpha_probe.num_alpha"), "hpo.grid.alpha_probe.num_alpha")
        _ensure_int(_get_path(cfg, "hpo.grid.alpha_probe.num_lr"), "hpo.grid.alpha_probe.num_lr")
        _ensure_int(_get_path(cfg, "hpo.grid.alpha_probe.epochs"), "hpo.grid.alpha_probe.epochs")
        ap_seeds = _require_list(_get_path(cfg, "hpo.grid.alpha_probe.seeds"), "hpo.grid.alpha_probe.seeds")
        if len(ap_seeds) < 2:
            raise ValueError("hpo.grid.alpha_probe.seeds must contain at least 2 seeds")
        for i, s in enumerate(ap_seeds):
            _ensure_int(s, f"hpo.grid.alpha_probe.seeds[{i}]")

        knob_specs = _require_dict(_get_path(cfg, "hpo.grid.knob_specs"), "hpo.grid.knob_specs")
        for kn in knobs:
            if kn not in knob_specs:
                raise KeyError(f"Missing knob spec for '{kn}' in hpo.grid.knob_specs")
            spec_path = f"hpo.grid.knob_specs.{kn}"
            spec = _require_dict(_get_path(cfg, spec_path), spec_path)
            kind = _ensure_str(spec.get("kind", None), f"{spec_path}.kind")
            if kind == "choice":
                choices = _require_list(spec.get("choices", None), f"{spec_path}.choices")
                if not choices:
                    raise ValueError(f"{spec_path}.choices must be non-empty")
                for i, v in enumerate(choices):
                    _ensure_float(v, f"{spec_path}.choices[{i}]")
            elif kind == "float":
                _ensure_float(spec.get("lo", None), f"{spec_path}.lo")
                _ensure_float(spec.get("hi", None), f"{spec_path}.hi")
                _ensure_float(spec.get("top_quantile", None), f"{spec_path}.top_quantile")
                _ensure_float(spec.get("padding_ratio", None), f"{spec_path}.padding_ratio")
                if "step" in spec and spec["step"] is not None:
                    _ensure_float(spec["step"], f"{spec_path}.step")
                if "clamp_lo" in spec and spec["clamp_lo"] is not None:
                    _ensure_float(spec["clamp_lo"], f"{spec_path}.clamp_lo")
                if "clamp_hi" in spec and spec["clamp_hi"] is not None:
                    _ensure_float(spec["clamp_hi"], f"{spec_path}.clamp_hi")
            else:
                raise ValueError(f"{spec_path}.kind must be 'float' or 'choice', got {kind!r}")

        variants = _require_list(_get_path(cfg, "hpo.baseline_variants"), "hpo.baseline_variants")
        if not variants:
            raise ValueError("hpo.baseline_variants must be a non-empty list")

        # STRICT: baseline_variants only carries tags (truth lives in method.baseline_{r,R}.lora)
        forbidden_keys = {"alpha", "r", "R", "dropout", "lora"}
        for i, v in enumerate(variants):
            if not isinstance(v, dict):
                raise TypeError(f"hpo.baseline_variants[{i}] must be dict, got {type(v)}")

            if "tag" not in v:
                raise KeyError(f"Missing required key in hpo.baseline_variants[{i}]: 'tag'")

            tag = str(v["tag"])
            if tag not in {"baseline_r", "baseline_R"}:
                raise ValueError(
                    f"hpo.baseline_variants[{i}].tag must be baseline_r or baseline_R, got {v['tag']}"
                )

            extra = set(v.keys()) - {"tag"}
            bad = extra & forbidden_keys
            if bad:
                raise ValueError(
                    "hpo.baseline_variants must NOT repeat baseline lora truth. "
                    f"Found forbidden keys in baseline_variants[{i}]: {sorted(bad)}. "
                    "Put these under method.baseline_r.lora / method.baseline_R.lora instead."
                )

    if cmd == "final":
        _require_dict(_get_path(cfg, "final"), "final")
        _ensure_int(_get_path(cfg, "final.epochs"), "final.epochs")
        seeds = _require_list(_get_path(cfg, "final.seeds"), "final.seeds")
        if not seeds:
            raise ValueError("final.seeds must be a non-empty list")
        for i, s in enumerate(seeds):
            _ensure_int(s, f"final.seeds[{i}]")

    # plot: no extra enforcement here (runner/plotting handles runs_dir etc.)


# -------------------------
# Public API
# -------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config (single file). STRICT: no defaults."""
    return _load_yaml(path)


def load_config_with_cli_overrides(
    config_path: str,
    schedule_path: Optional[str] = None,
    set_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load base config + optional schedule yaml + apply --set overrides.

    Priority (low -> high):
      base.yaml
      schedule.yaml (if provided)
      CLI overrides (--set)

    STRICT: no defaults, no fallback.
    """
    base = load_config(config_path)

    if schedule_path:
        sch = _load_yaml(schedule_path)
        _deep_update(base, sch)

    if set_args:
        _apply_overrides(base, set_args)

    # provenance for reproducibility (written to run_dir by runner)
    base["_provenance"] = {
        "base_config_path": str(config_path),
        "schedule_path": str(schedule_path) if schedule_path else None,
        "cli_set_args": list(set_args) if set_args else [],
    }
    return base


def build_argparser() -> argparse.ArgumentParser:
    """
    scripts/run.py has its own parser, but keeping this helps reuse in tests/tools.
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--schedule", type=str, default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--trial_tag", type=str, default=None)
    return ap

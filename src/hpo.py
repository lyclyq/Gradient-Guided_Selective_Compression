# src/hpo.py
from __future__ import annotations

import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .artifacts import make_run_name, prepare_run_dir, dump_json


# ----------------------------
# Helpers: grids, flattening
# ----------------------------

def _logspace(min_exp: int, max_exp: int, points: int) -> List[float]:
    return np.logspace(min_exp, max_exp, num=points).astype(float).tolist()


def _task_short_name(cfg: Dict[str, Any]) -> str:
    name = (cfg.get("task", {}).get("name") or "").lower()
    if "/" in name:
        return name.split("/")[-1]
    return name


def _is_small_sensitive_task(task: str) -> bool:
    return task in {"cola", "rte", "mrpc"}


def _fixed_warmup(cfg: Dict[str, Any]) -> float:
    return float(cfg.get("train", {}).get("warmup_ratio", 0.06))


def _flatten_sets(prefix: str, d: Dict[str, Any], out: List[str]) -> None:
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_sets(p, v, out)
        else:
            out.append(f"{p}={v}")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ----------------------------
# Candidate pools (S1)
# ----------------------------

def _default_param_candidates(param: str) -> List[float]:
    if param == "lambda_n":
        return [0.1, 0.3, 0.5, 0.7]
    if param == "lambda_o":
        return [0.1, 0.3, 0.5, 0.7]
    if param == "eta":
        return [0.05, 0.1, 0.2, 0.3]
    if param == "k":
        return [4.0, 8.0, 12.0, 16.0]
    if param == "gamma_hi":
        return [0.1, 0.3, 0.5, 0.7]
    if param == "gamma_r":
        return [0.1, 0.3, 0.5, 0.7]
    return []


# ----------------------------
# Run dir + trials.csv
# ----------------------------

def _ensure_hpo_run_dir(cfg: Dict[str, Any]) -> Path:
    io = cfg.get("io", {})
    root = io.get("root", "runs")
    overwrite = io.get("overwrite", "ask")
    run_name = make_run_name(cfg, extra={"kind": "hpo"})
    run_dir, _ = prepare_run_dir(root=root, run_name=run_name, overwrite=overwrite)
    return run_dir


def _read_all_rows(csv_path: Path) -> List[Dict[str, Any]]:
    import csv
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(x) for x in r]


def _best_row(rows: List[Dict[str, Any]], stage_prefix: Optional[str] = None) -> Optional[Dict[str, Any]]:
    best = None
    best_score = float("-inf")
    for r in rows:
        if stage_prefix and not str(r.get("stage", "")).startswith(stage_prefix):
            continue
        try:
            s = float(r.get("score", "-inf"))
        except Exception:
            continue
        if s > best_score:
            best = r
            best_score = s
    return best


# ----------------------------
# LR search helpers
# ----------------------------

def _baseline_lr_candidates(cfg: Dict[str, Any], points: int) -> List[float]:
    hpo = cfg.get("hpo", {})
    lr_cfg = hpo.get("lr_grid", {})
    full = _logspace(int(lr_cfg.get("min_exp", -6)), int(lr_cfg.get("max_exp", -3)), int(lr_cfg.get("points", 15)))
    if points >= len(full):
        return full
    idxs = np.linspace(0, len(full) - 1, num=points).round().astype(int).tolist()
    uniq: List[float] = []
    for i in idxs:
        v = float(full[i])
        if v not in uniq:
            uniq.append(v)
    return uniq


def _lr_neighborhood(cfg: Dict[str, Any], best_lr: float, neighbor_points: int) -> List[float]:
    hpo = cfg.get("hpo", {})
    lr_cfg = hpo.get("lr_grid", {})
    full = _logspace(int(lr_cfg.get("min_exp", -6)), int(lr_cfg.get("max_exp", -3)), int(lr_cfg.get("points", 15)))

    arr = np.array(full, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(best_lr))))

    radius = (neighbor_points - 1) // 2
    lo = max(0, idx - radius)
    hi = min(len(full), idx + radius + 1)
    cand = [float(x) for x in full[lo:hi]]

    while len(cand) < neighbor_points and lo > 0:
        lo -= 1
        cand.insert(0, float(full[lo]))
    while len(cand) < neighbor_points and hi < len(full):
        cand.append(float(full[hi]))
        hi += 1

    return cand


# ----------------------------
# Trial execution
# ----------------------------

def _run_one_trial(
    *,
    base_config_path: str,
    schedule_path: Optional[str],
    base_set_args: List[str],
    trial_override: Dict[str, Any],
    trial_tag: str,
    stage: str,
    hpo_run_dir: Path,
    score_weights: Tuple[float, float, float],
) -> None:
    cmd = ["python", "scripts/run.py", "train", "--config", base_config_path]
    if schedule_path:
        cmd += ["--schedule", schedule_path]

    sets: List[str] = []
    if base_set_args:
        sets.extend(base_set_args)

    trial_sets: List[str] = []
    _flatten_sets("", trial_override, trial_sets)
    sets.extend(trial_sets)

    for s in sets:
        cmd += ["--set", s]

    cmd += ["--trial_tag", trial_tag]

    env = os.environ.copy()
    env["GSA_HPO_RUN_DIR"] = str(hpo_run_dir)
    env["GSA_STAGE"] = stage
    w_max, w_final, w_avg = score_weights
    env["GSA_W_MAX"] = str(w_max)
    env["GSA_W_FINAL"] = str(w_final)
    env["GSA_W_AVG"] = str(w_avg)

    print(" ".join(cmd))
    subprocess.run(cmd, check=False, env=env)


# ----------------------------
# Main: staged HPO
# ----------------------------

def run_hpo(cfg: Dict[str, Any], base_config_path: str, schedule_path: Optional[str], set_args: List[str]) -> None:
    """
    Successive-Halving style HPO:
      - baseline: LR search only (warmup fixed)
      - ours S1: 1-epoch one-factor screening
      - ours S2: 2-epoch small combinational search (top-K params)
      - ours S3: 4-epoch refinement under budget

    Training trials append rows into <hpo_run_dir>/trials.csv via runner.py (env GSA_HPO_RUN_DIR).
    """
    hpo_run_dir = _ensure_hpo_run_dir(cfg)
    state_path = hpo_run_dir / "state.json"
    trials_csv = hpo_run_dir / "trials.csv"
    best_path = hpo_run_dir / "best_hparams.json"

    task = _task_short_name(cfg)
    small_task = _is_small_sensitive_task(task)

    hpo_cfg = cfg.get("hpo", {})
    ours_cfg = hpo_cfg.get("ours", {}) if isinstance(hpo_cfg.get("ours", {}), dict) else {}
    stages = ours_cfg.get("stages", {}) if isinstance(ours_cfg.get("stages", {}), dict) else {}

    s1 = stages.get("s1", {})
    s2 = stages.get("s2", {})
    s3 = stages.get("s3", {})

    score_cfg = (s3.get("score") or {}) if isinstance(s3.get("score"), dict) else {}
    w_max = float(score_cfg.get("w_max", 0.5))
    w_final = float(score_cfg.get("w_final", 0.4))
    w_avg = float(score_cfg.get("w_avg", 0.1))
    score_weights = (w_max, w_final, w_avg)

    neighbor_points = int(
        ours_cfg.get("lr_neighbor_points_small", 7) if small_task else ours_cfg.get("lr_neighbor_points_large", 5)
    )

    fixed_wu = _fixed_warmup(cfg)

    # resume state
    state = _read_json(state_path)
    done_tags = set(state.get("done_tags", []))
    stage = state.get("stage", "baseline")
    s1_top = state.get("s1_top", {})
    s2_top_params = state.get("s2_top_params", [])
    s2_top_combos = state.get("s2_top_combos", [])
    best_baseline_lr = state.get("best_baseline_lr", None)

    def _save_state():
        dump_json(state_path, {
            "stage": stage,
            "done_tags": sorted(done_tags),
            "s1_top": s1_top,
            "s2_top_params": s2_top_params,
            "s2_top_combos": s2_top_combos,
            "best_baseline_lr": best_baseline_lr,
        })

    # ---------------- baseline ----------------
    if stage == "baseline":
        baseline_lr_points = int(hpo_cfg.get("baseline_lr_points", 5))
        base_lrs = _baseline_lr_candidates(cfg, points=baseline_lr_points)

        for i, lr in enumerate(base_lrs):
            trial_tag = f"baseline_lr_{i}"
            if trial_tag in done_tags:
                continue

            trial_override = {
                "method": {"name": "baseline"},
                "train": {"lr": float(lr), "warmup_ratio": float(fixed_wu)},
            }
            _run_one_trial(
                base_config_path=base_config_path,
                schedule_path=schedule_path,
                base_set_args=set_args,
                trial_override=trial_override,
                trial_tag=trial_tag,
                stage="baseline",
                hpo_run_dir=hpo_run_dir,
                score_weights=score_weights,
            )
            done_tags.add(trial_tag)
            _save_state()

        rows = _read_all_rows(trials_csv)
        best_base = _best_row(rows, stage_prefix="baseline")
        if not best_base:
            print("[WARN] No baseline rows found in trials.csv yet. Did runner append trials.csv?")
            return

        try:
            cfg_json = json.loads(best_base.get("trial_cfg_json", "{}"))
            best_baseline_lr = float(cfg_json.get("train", {}).get("lr", cfg.get("train", {}).get("lr", 2e-5)))
        except Exception:
            best_baseline_lr = float(cfg.get("train", {}).get("lr", 2e-5))

        stage = "s1"
        _save_state()

    if best_baseline_lr is None:
        best_baseline_lr = float(cfg.get("train", {}).get("lr", 2e-5))

    lr_neighbors = _lr_neighborhood(cfg, best_lr=float(best_baseline_lr), neighbor_points=neighbor_points)

    # ---------------- S1: one-factor screening ----------------
    if stage == "s1":
        s1_trials_budget = int(s1.get("trials", 100))
        s1_epochs = int(s1.get("epochs", 1))
        param_pool = s1.get("param_pool", ["lambda_n", "lambda_o", "eta", "k", "gamma_hi"])
        candidates_per_param = int(s1.get("candidates_per_param", 4))
        keep_top = int(s1.get("keep_top", 2))

        pool: List[Tuple[str, List[float]]] = []
        for p in param_pool:
            cand = _default_param_candidates(str(p))
            if not cand:
                continue
            pool.append((str(p), cand[:candidates_per_param]))

        planned: List[Tuple[str, float, float]] = []
        for param, cand_list in pool:
            for v in cand_list:
                for lr in lr_neighbors:
                    planned.append((param, float(v), float(lr)))

        planned = planned[:s1_trials_budget]

        for j, (param, v, lr) in enumerate(planned):
            trial_tag = f"ours_s1_{param}_{j}"
            if trial_tag in done_tags:
                continue

            trial_override = {
                "method": {"name": "ours", "ours": {param: float(v)}},
                "train": {"lr": float(lr), "warmup_ratio": float(fixed_wu), "epochs": int(s1_epochs)},
            }
            _run_one_trial(
                base_config_path=base_config_path,
                schedule_path=schedule_path,
                base_set_args=set_args,
                trial_override=trial_override,
                trial_tag=trial_tag,
                stage="ours_s1",
                hpo_run_dir=hpo_run_dir,
                score_weights=score_weights,
            )
            done_tags.add(trial_tag)
            _save_state()

        # summarize S1 from trials.csv
        rows = _read_all_rows(trials_csv)
        per_param_value: Dict[str, Dict[float, float]] = {}
        for r in rows:
            if str(r.get("stage", "")).startswith("ours_s1") is False:
                continue
            try:
                cfg_json = json.loads(r.get("trial_cfg_json", "{}"))
                ours = cfg_json.get("method", {}).get("ours", {})
                if not isinstance(ours, dict) or len(ours) != 1:
                    continue
                (p, val) = next(iter(ours.items()))
                score = float(r.get("score", "-inf"))
                per_param_value.setdefault(str(p), {})
                prev = per_param_value[str(p)].get(float(val), float("-inf"))
                per_param_value[str(p)][float(val)] = max(prev, score)
            except Exception:
                continue

        s1_top = {}
        sensitivity: List[Tuple[str, float]] = []
        for p, m in per_param_value.items():
            ranked = sorted(m.items(), key=lambda kv: kv[1], reverse=True)
            kept = [float(kv[0]) for kv in ranked[:keep_top]]
            if kept:
                s1_top[p] = kept
            if ranked:
                scores = [kv[1] for kv in ranked]
                sens = ranked[0][1] - float(np.median(scores))
                sensitivity.append((p, sens))

        combine_topk = int(s2.get("combine_topk_params", 3))
        sensitivity.sort(key=lambda x: x[1], reverse=True)
        s2_top_params = [p for (p, _) in sensitivity[:combine_topk] if p in s1_top]

        stage = "s2"
        _save_state()

    # ---------------- S2: small combinational search ----------------
    if stage == "s2":
        s2_trials_budget = int(s2.get("trials", 40))
        s2_epochs = int(s2.get("epochs", 2))

        top_params = list(s2_top_params) if s2_top_params else []
        if not top_params:
            for p in ["lambda_n", "lambda_o", "eta"]:
                if p in s1_top:
                    top_params.append(p)
            top_params = top_params[:3]

        cand_lists: List[Tuple[str, List[float]]] = []
        for p in top_params:
            cand = s1_top.get(p, [])
            if len(cand) < 2:
                cand = cand + cand
            cand_lists.append((p, [float(x) for x in cand[:2]]))

        combos: List[Dict[str, float]] = []
        for vals in itertools.product(*[cl for _, cl in cand_lists]):
            d: Dict[str, float] = {}
            for (p, _), v in zip(cand_lists, vals):
                d[p] = float(v)
            combos.append(d)

        planned: List[Tuple[float, Dict[str, float]]] = []
        for lr in lr_neighbors:
            for c in combos:
                planned.append((float(lr), c))
        planned = planned[:s2_trials_budget]

        for j, (lr, c) in enumerate(planned):
            trial_tag = f"ours_s2_{j}"
            if trial_tag in done_tags:
                continue

            trial_override = {
                "method": {"name": "ours", "ours": c},
                "train": {"lr": float(lr), "warmup_ratio": float(fixed_wu), "epochs": int(s2_epochs)},
            }
            _run_one_trial(
                base_config_path=base_config_path,
                schedule_path=schedule_path,
                base_set_args=set_args,
                trial_override=trial_override,
                trial_tag=trial_tag,
                stage="ours_s2",
                hpo_run_dir=hpo_run_dir,
                score_weights=score_weights,
            )
            done_tags.add(trial_tag)
            _save_state()

        # pick top-M combos to carry into S3
        rows = _read_all_rows(trials_csv)
        s2_rows = [r for r in rows if str(r.get("stage", "")).startswith("ours_s2")]
        s2_rows_sorted = sorted(s2_rows, key=lambda r: float(r.get("score", "-inf")), reverse=True)

        top_m = min(10, len(s2_rows_sorted))
        s2_top_combos = []
        for r in s2_rows_sorted[:top_m]:
            try:
                cfg_json = json.loads(r.get("trial_cfg_json", "{}"))
                ours = cfg_json.get("method", {}).get("ours", {})
                if isinstance(ours, dict) and ours:
                    s2_top_combos.append(ours)
            except Exception:
                pass

        stage = "s3"
        _save_state()

    # ---------------- S3: 4-epoch refinement ----------------
    if stage == "s3":
        s3_epochs = int(s3.get("epochs", 4))
        s3_trials_budget = int(s3.get("trials_small", 120) if small_task else s3.get("trials_large", 90))

        combos = list(s2_top_combos) if s2_top_combos else [{"lambda_n": 0.4, "lambda_o": 0.4, "eta": 0.2}]
        planned: List[Tuple[float, Dict[str, Any]]] = []
        for lr in lr_neighbors:
            for c in combos:
                planned.append((float(lr), c))
        planned = planned[:s3_trials_budget]

        for j, (lr, c) in enumerate(planned):
            trial_tag = f"ours_s3_{j}"
            if trial_tag in done_tags:
                continue

            trial_override = {
                "method": {"name": "ours", "ours": c},
                "train": {"lr": float(lr), "warmup_ratio": float(fixed_wu), "epochs": int(s3_epochs)},
            }
            _run_one_trial(
                base_config_path=base_config_path,
                schedule_path=schedule_path,
                base_set_args=set_args,
                trial_override=trial_override,
                trial_tag=trial_tag,
                stage="ours_s3",
                hpo_run_dir=hpo_run_dir,
                score_weights=score_weights,
            )
            done_tags.add(trial_tag)
            _save_state()

        rows = _read_all_rows(trials_csv)
        best = _best_row(rows, stage_prefix=None)
        if best:
            dump_json(best_path, {"best": best, "weights": {"w_max": w_max, "w_final": w_final, "w_avg": w_avg}})
            print(f"[OK] best saved to {best_path}")
        else:
            print("[WARN] No rows found in trials.csv; cannot choose best.")

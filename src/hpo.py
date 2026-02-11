# src/hpo.py
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .artifacts import make_run_name, prepare_run_dir, dump_json
from .hpo_budget import build_grid_plan, make_lr_candidates_from_plan, lr_neighborhood_from_plan


TRIALS_HEADER = [
    "trial_id",
    "trial_tag",
    "stage",
    "score",
    "val_max",
    "val_final",
    "val_avg",
    "trial_cfg_json",
    "seeds",
    "run_dir",
    "metrics_csv",
]


# -----------------------------
# Small helpers
# -----------------------------
def _stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _read_all_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(x) for x in r]


def _append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIALS_HEADER)
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in TRIALS_HEADER})


def _row_by_trial_tag(rows: List[Dict[str, Any]], trial_tag: str) -> Optional[Dict[str, Any]]:
    for r in rows[::-1]:
        if str(r.get("trial_tag", "")) == str(trial_tag):
            return r
    return None


def _best_row(rows: List[Dict[str, Any]], *, stage_prefix: Optional[str] = None, only_aggregate: bool = False) -> Optional[Dict[str, Any]]:
    best = None
    best_score = float("-inf")
    for r in rows:
        if stage_prefix and not str(r.get("stage", "")).startswith(stage_prefix):
            continue
        if only_aggregate:
            seeds = str(r.get("seeds", "")).strip()
            if "," not in seeds:
                continue
        try:
            s = float(r.get("score", "-inf"))
        except Exception:
            continue
        if s > best_score:
            best, best_score = r, s
    return best


def _flatten_sets(prefix: str, d: Dict[str, Any], out: List[str]) -> None:
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_sets(p, v, out)
        else:
            out.append(f"{p}={v}")


def _copy_metrics_csv(row: Dict[str, Any], dest: Path) -> bool:
    src = str(row.get("metrics_csv", "")).strip()
    if not src:
        return False
    sp = Path(src)
    if not sp.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sp, dest)
    return True


# -----------------------------
# Run dir + strict invariants
# -----------------------------
def _ensure_hpo_run_dir(cfg: Dict[str, Any]) -> Path:
    """
    Priority:
      1) cfg.io.run_dir + overwrite semantics
      2) cfg.io.root + timestamped make_run_name(kind=hpo)
    """
    io = cfg["io"]
    overwrite = str(io["overwrite"])

    run_dir_cfg = io.get("run_dir", None)
    if run_dir_cfg:
        p = Path(str(run_dir_cfg))
        if not p.is_absolute():
            p = Path(os.getcwd()) / p
        p.parent.mkdir(parents=True, exist_ok=True)

        if p.exists():
            if overwrite == "resume":
                return p
            if overwrite == "force":
                shutil.rmtree(p)
                p.mkdir(parents=True, exist_ok=True)
                return p
            ans = input(f"[artifacts] {p} exists. Overwrite? [y/N] ").strip().lower()
            if ans in {"y", "yes"}:
                shutil.rmtree(p)
                p.mkdir(parents=True, exist_ok=True)
                return p
            return p

        p.mkdir(parents=True, exist_ok=True)
        return p

    root = io["root"]
    run_name = make_run_name(cfg, extra={"kind": "hpo"})
    run_dir, _ = prepare_run_dir(root=root, run_name=run_name, overwrite=overwrite)
    return Path(run_dir)


def _assert_capacity_unified(cfg: Dict[str, Any]) -> None:
    """
    Hard command:
      baseline_r.rank == ours.r
      baseline_R.rank == ours.R
    """
    br = int(cfg["method"]["baseline_r"]["lora"]["r"])
    bR = int(cfg["method"]["baseline_R"]["lora"]["r"])
    orr = int(cfg["method"]["ours"]["lora"]["r"])
    oRR = int(cfg["method"]["ours"]["lora"]["R"])
    if br != orr:
        raise RuntimeError(f"[HPO][CapacityMismatch] baseline_r.r={br} != ours.r={orr}")
    if bR != oRR:
        raise RuntimeError(f"[HPO][CapacityMismatch] baseline_R.r={bR} != ours.R={oRR}")


def _baseline_tags(cfg: Dict[str, Any]) -> List[str]:
    hpo = cfg["hpo"]
    variants = hpo["baseline_variants"]
    if not isinstance(variants, list) or not variants:
        raise RuntimeError("[HPO] missing hpo.baseline_variants (must be list of {tag: ...})")
    tags: List[str] = []
    for i, v in enumerate(variants):
        if not isinstance(v, dict) or "tag" not in v:
            raise RuntimeError(f"[HPO] baseline_variants[{i}] must be dict with tag")
        tag = str(v["tag"])
        if tag not in {"baseline_r", "baseline_R"}:
            raise RuntimeError(f"[HPO] baseline_variants[{i}].tag must be baseline_r/baseline_R, got {tag}")
        tags.append(tag)
    out: List[str] = []
    seen = set()
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _baseline_lora_from_method(cfg: Dict[str, Any], tag: str) -> Dict[str, Any]:
    """
    STRICT: baseline truth only from method.baseline_{r,R}.lora
    """
    if tag not in {"baseline_r", "baseline_R"}:
        raise ValueError(tag)
    blk = cfg["method"][tag]["lora"]
    r = int(blk["r"])
    alpha = float(blk["alpha"])
    dropout = float(blk["dropout"])
    return {"r": r, "alpha": alpha, "dropout": dropout}


# -----------------------------
# plan_status.json (explicit)
# -----------------------------
@dataclass
class PlanItem:
    idx: int
    stage: str
    trial_tag: str
    seed: Optional[int]  # None for aggregate rows
    run_dir: str
    override: Dict[str, Any]
    status: str = "pending"  # pending | running | done | failed
    tries: int = 0
    last_rc: Optional[int] = None
    last_update: int = 0


def _load_plan_status(path: Path) -> Dict[str, Any]:
    obj = _read_json(path, default=None)
    if not isinstance(obj, dict) or "items" not in obj:
        return {"items": [], "created_at": int(time.time()), "updated_at": int(time.time())}
    if not isinstance(obj.get("items", None), list):
        obj["items"] = []
    return obj


def _save_plan_status(path: Path, st: Dict[str, Any]) -> None:
    st2 = dict(st)
    st2["updated_at"] = int(time.time())
    _atomic_write_json(path, st2)


def _plan_index(st: Dict[str, Any]) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    items = st.get("items", [])
    if not isinstance(items, list):
        return idx
    for i, it in enumerate(items):
        tt = str(it.get("trial_tag", ""))
        sd = it.get("seed", None)
        key = f"{tt}__seed_{sd}" if sd is not None else f"{tt}__agg"
        idx[key] = i
    return idx


def _update_item(st: Dict[str, Any], item: PlanItem) -> None:
    items = st.get("items", [])
    if not isinstance(items, list):
        items = []
        st["items"] = items

    key = f"{item.trial_tag}__seed_{item.seed}" if item.seed is not None else f"{item.trial_tag}__agg"
    index = _plan_index(st)
    payload = {
        "idx": int(item.idx),
        "stage": str(item.stage),
        "trial_tag": str(item.trial_tag),
        "seed": item.seed,
        "run_dir": str(item.run_dir),
        "override": item.override,
        "status": str(item.status),
        "tries": int(item.tries),
        "last_rc": item.last_rc,
        "last_update": int(item.last_update),
    }
    if key in index:
        items[index[key]] = payload
    else:
        items.append(payload)


def _is_done_in_trials_csv(trials_csv: Path, trial_tag: str) -> bool:
    rows = _read_all_rows(trials_csv)
    return _row_by_trial_tag(rows, trial_tag) is not None


def _next_pending_item(st: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    items = st.get("items", [])
    if not isinstance(items, list):
        return None
    cand = [it for it in items if str(it.get("status", "")) in {"pending", "failed"}]
    if not cand:
        return None
    cand.sort(key=lambda it: int(it.get("idx", 10**9)))
    return cand[0]


# -----------------------------
# runner invocation
# -----------------------------
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
    run_dir: Optional[Path] = None,
    overwrite_mode: str = "resume",
) -> int:
    """
    Returns subprocess returncode.
    NOTE: trials.csv is appended by runner ONLY if training finishes.
    """
    cmd = ["python", "scripts/run.py", "train", "--config", base_config_path]
    if schedule_path:
        cmd += ["--schedule", schedule_path]

    sets: List[str] = []
    if base_set_args:
        sets.extend(base_set_args)

    if run_dir is not None:
        sets.append(f"io.run_dir={str(run_dir)}")
        sets.append(f"io.overwrite={overwrite_mode}")

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
    p = subprocess.run(cmd, check=False, env=env)
    return int(p.returncode)


# -----------------------------
# seed aggregation
# -----------------------------
def _aggregate_seed_trial(
    *,
    trials_csv: Path,
    base_trial_tag: str,
    stage: str,
    seeds: List[int],
    trial_cfg_json: str,
) -> Optional[Dict[str, Any]]:
    rows = _read_all_rows(trials_csv)
    seed_rows: List[Dict[str, Any]] = []
    for sd in seeds:
        tag = f"{base_trial_tag}__s{sd}"
        r = _row_by_trial_tag(rows, tag)
        if r is not None:
            seed_rows.append(r)
    if len(seed_rows) != len(seeds):
        return None

    def _f(x, default=-1e18):
        try:
            return float(x)
        except Exception:
            return float(default)

    score_mean = float(np.mean([_f(r.get("score")) for r in seed_rows]))
    val_max_mean = float(np.mean([_f(r.get("val_max")) for r in seed_rows]))
    val_final_mean = float(np.mean([_f(r.get("val_final")) for r in seed_rows]))
    val_avg_mean = float(np.mean([_f(r.get("val_avg")) for r in seed_rows]))

    best_seed_row = max(seed_rows, key=lambda r: _f(r.get("score")))
    agg_row = {
        "trial_id": "",
        "trial_tag": base_trial_tag,
        "stage": stage,
        "score": score_mean,
        "val_max": val_max_mean,
        "val_final": val_final_mean,
        "val_avg": val_avg_mean,
        "trial_cfg_json": trial_cfg_json,
        "seeds": ",".join([str(x) for x in seeds]),
        "run_dir": str(best_seed_row.get("run_dir", "")),
        "metrics_csv": str(best_seed_row.get("metrics_csv", "")),
    }

    if _row_by_trial_tag(rows, base_trial_tag) is None:
        _append_row(trials_csv, agg_row)

    return agg_row


# -----------------------------
# Grid solve (drop + density)
# -----------------------------
@dataclass
class KnobRange:
    name: str
    kind: str  # float | choice
    lo: Optional[float] = None
    hi: Optional[float] = None
    choices: Optional[List[float]] = None
    weight: float = 0.0
    m: int = 0


def _solve_grid_points(*, L: int, B: int, knobs: List[KnobRange], max_m_float: int = 7, max_m_choice: int = 0) -> Tuple[List[KnobRange], Dict[str, Any]]:
    knobs = [k for k in knobs]
    knobs = sorted(knobs, key=lambda k: float(k.weight), reverse=True)

    def m_min(k: KnobRange) -> int:
        if k.kind == "choice":
            if not k.choices:
                return 0
            return min(2, len(k.choices))
        return 2

    def m_max(k: KnobRange) -> int:
        if k.kind == "choice":
            if not k.choices:
                return 0
            if max_m_choice and max_m_choice > 0:
                return min(int(max_m_choice), len(k.choices))
            return len(k.choices)
        return int(max(2, max_m_float))

    def total_points(ks: List[KnobRange]) -> int:
        prod = 1
        for kk in ks:
            prod *= max(1, int(kk.m))
        return int(L * prod)

    for k in knobs:
        k.m = m_min(k)

    dropped: List[str] = []
    while knobs and total_points(knobs) > B:
        tail = knobs.pop(-1)
        dropped.append(tail.name)

    if not knobs:
        meta = {"dropped_knobs": dropped, "reason": "budget_too_small_even_after_drop"}
        return [], meta

    def score_gain(k: KnobRange) -> float:
        mm = max(1, int(k.m))
        return float(k.weight) / float((mm + 1) / mm)

    changed = True
    while changed:
        changed = False
        cand = sorted(knobs, key=lambda k: score_gain(k), reverse=True)
        for k in cand:
            if k.m >= m_max(k):
                continue
            old_m = int(k.m)
            k.m = old_m + 1
            if total_points(knobs) <= B:
                changed = True
                break
            k.m = old_m

    meta = {
        "dropped_knobs": dropped,
        "L": int(L),
        "B": int(B),
        "final_total_points": int(total_points(knobs)),
        "per_knob_m": {k.name: int(k.m) for k in knobs},
    }
    return knobs, meta


def _make_knob_values(k: KnobRange) -> List[float]:
    if k.kind == "choice":
        ch = list(k.choices or [])
        if not ch:
            return []
        if k.m <= 0 or k.m >= len(ch):
            return [float(x) for x in ch]
        idxs = np.linspace(0, len(ch) - 1, num=int(k.m)).round().astype(int).tolist()
        out: List[float] = []
        for ii in idxs:
            v = float(ch[int(ii)])
            if v not in out:
                out.append(v)
        return out

    if k.lo is None or k.hi is None:
        raise RuntimeError(f"[HPO] float knob range missing bounds for {k.name}")
    lo = float(k.lo)
    hi = float(k.hi)
    m = int(max(2, k.m))
    vals = np.linspace(lo, hi, num=m).astype(float).tolist()
    out: List[float] = []
    for v in vals:
        v = float(v)
        if v not in out:
            out.append(v)
    return out


# -----------------------------
# Bayes optimizer (GP + EI)
# -----------------------------
def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, length: float, var: float) -> np.ndarray:
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    s1 = np.sum(X1 * X1, axis=1, keepdims=True)
    s2 = np.sum(X2 * X2, axis=1, keepdims=True).T
    D2 = s1 + s2 - 2.0 * (X1 @ X2.T)
    return var * np.exp(-0.5 * D2 / max(1e-12, length * length))


def _gp_posterior(
    X: np.ndarray,
    y: np.ndarray,
    Xcand: np.ndarray,
    *,
    length: float = 1.0,
    var: float = 1.0,
    noise: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    Xcand = np.asarray(Xcand, dtype=float)

    n = X.shape[0]
    if n == 0:
        mu0 = np.zeros((Xcand.shape[0],), dtype=float)
        sg0 = np.ones((Xcand.shape[0],), dtype=float)
        return mu0, sg0

    K = _rbf_kernel(X, X, length=length, var=var)
    K = K + float(noise) * np.eye(n, dtype=float)
    Ks = _rbf_kernel(X, Xcand, length=length, var=var)
    Kss = _rbf_kernel(Xcand, Xcand, length=length, var=var)

    L = np.linalg.cholesky(K + 1e-12 * np.eye(n))
    v = np.linalg.solve(L, y)
    alpha = np.linalg.solve(L.T, v)

    mu = (Ks.T @ alpha).reshape(-1)

    w = np.linalg.solve(L, Ks)
    cov = Kss - (w.T @ w)
    cov = (cov + cov.T) * 0.5
    var_diag = np.clip(np.diag(cov), 1e-12, None)
    sigma = np.sqrt(var_diag)
    return mu, sigma


def _phi(x: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)


def _Phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    imp = mu - best - float(xi)
    Z = imp / np.clip(sigma, 1e-12, None)
    ei = imp * _Phi(Z) + sigma * _phi(Z)
    ei[sigma < 1e-12] = 0.0
    return ei


def _encode_config(
    *,
    lr: float,
    knobs: Dict[str, float],
    knob_space: List[KnobRange],
    choice_maps: Dict[str, List[float]],
    lr_min: float,
    lr_max: float,
) -> List[float]:
    v: List[float] = []
    lrl = math.log10(max(1e-20, float(lr)))
    mn = math.log10(max(1e-20, float(lr_min)))
    mx = math.log10(max(1e-20, float(lr_max)))
    if abs(mx - mn) < 1e-12:
        v.append(0.5)
    else:
        v.append((lrl - mn) / (mx - mn))

    for k in knob_space:
        name = k.name
        if k.kind == "choice":
            ch = choice_maps.get(name, list(k.choices or []))
            if not ch:
                v.append(0.0)
            else:
                try:
                    idx = int(np.argmin(np.abs(np.array(ch, dtype=float) - float(knobs.get(name, ch[0])))))
                except Exception:
                    idx = 0
                if len(ch) == 1:
                    v.append(0.0)
                else:
                    v.append(float(idx) / float(len(ch) - 1))
        else:
            lo = float(k.lo if k.lo is not None else 0.0)
            hi = float(k.hi if k.hi is not None else lo + 1.0)
            x = float(knobs.get(name, lo))
            if abs(hi - lo) < 1e-12:
                v.append(0.5)
            else:
                v.append((x - lo) / (hi - lo))
    return [float(max(0.0, min(1.0, x))) for x in v]


def _random_sample_config(
    rng: np.random.Generator,
    *,
    lr_candidates: List[float],
    knob_space: List[KnobRange],
) -> Tuple[float, Dict[str, float]]:
    lr = float(rng.choice(lr_candidates))
    assign: Dict[str, float] = {}
    for k in knob_space:
        if k.kind == "choice":
            ch = list(k.choices or [])
            if not ch:
                continue
            assign[k.name] = float(rng.choice(ch))
        else:
            if k.lo is None or k.hi is None:
                raise RuntimeError(f"[HPO] float knob range missing bounds for {k.name}")
            lo = float(k.lo)
            hi = float(k.hi)
            assign[k.name] = float(rng.uniform(lo, hi))
    return lr, assign


# -----------------------------
# Alpha probe helpers
# -----------------------------
def _median_from_knob_range(k: KnobRange) -> float:
    if k.kind == "choice":
        ch = list(k.choices or [])
        if not ch:
            return 0.0
        ch2 = sorted([float(x) for x in ch])
        return float(ch2[len(ch2) // 2])
    lo = float(k.lo if k.lo is not None else 0.0)
    hi = float(k.hi if k.hi is not None else lo)
    return float((lo + hi) / 2.0)


def _median_from_spec(spec: Dict[str, Any]) -> float:
    if "kind" not in spec:
        raise RuntimeError("[HPO] knob spec missing key: kind")
    kind = str(spec["kind"])
    if kind == "choice":
        choices = spec.get("choices", None)
        if isinstance(choices, list) and choices:
            ch = sorted([float(x) for x in choices])
            return float(ch[len(ch) // 2])
        raise RuntimeError("[HPO] choice knob spec must provide non-empty choices")
    lo = float(spec["lo"])
    hi = float(spec["hi"])
    return float((lo + hi) / 2.0)


def _score_mean_for_tags(trials_csv: Path, tags: List[str]) -> Optional[float]:
    rows = _read_all_rows(trials_csv)
    vals: List[float] = []
    for t in tags:
        r = _row_by_trial_tag(rows, t)
        if r is None:
            return None
        try:
            vals.append(float(r.get("score", "-inf")))
        except Exception:
            return None
    if not vals:
        return None
    return float(np.mean(np.array(vals, dtype=float)))


# -----------------------------
# Main entry
# -----------------------------
def run_hpo(cfg: Dict[str, Any], base_config_path: str, schedule_path: Optional[str], set_args: List[str]) -> None:
    _assert_capacity_unified(cfg)

    hpo_run_dir = _ensure_hpo_run_dir(cfg)
    trials_csv = hpo_run_dir / "trials.csv"
    best_path = hpo_run_dir / "best_hparams.json"
    plan_path = hpo_run_dir / "grid_plan.json"
    status_path = hpo_run_dir / "plan_status.json"
    bundle_path = hpo_run_dir / "hpo_bundle.json"
    best_curves_dir = hpo_run_dir / "best_curves"
    snapshot_path = hpo_run_dir / "config_snapshot.json"

    # provenance: keep exact configs used for this HPO run
    try:
        used_dir = hpo_run_dir / "configs_used"
        used_dir.mkdir(parents=True, exist_ok=True)
        bp = Path(str(base_config_path))
        if bp.exists() and bp.is_file():
            shutil.copy2(bp, used_dir / bp.name)
        if schedule_path:
            sp = Path(str(schedule_path))
            if sp.exists() and sp.is_file():
                shutil.copy2(sp, used_dir / sp.name)
        dump_json(
            hpo_run_dir / "config_provenance.json",
            {
                "base_config_path": str(base_config_path),
                "schedule_path": str(schedule_path) if schedule_path else None,
                "cli_set_args": list(set_args or []),
            },
        )
    except Exception:
        pass

    _atomic_write_json(snapshot_path, cfg)
    (hpo_run_dir / "base_merged.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    dump_json(hpo_run_dir / "cli_overrides.json", {"set_args": list(set_args or [])})

    plan = build_grid_plan(cfg)
    _atomic_write_json(plan_path, plan)

    hpo_cfg = cfg["hpo"]
    bandit = hpo_cfg["bandit"]
    score_cfg = bandit["score"]
    w_max = float(score_cfg["w_max"])
    w_final = float(score_cfg["w_final"])
    w_avg = float(score_cfg["w_avg"])
    score_weights = (w_max, w_final, w_avg)

    fixed_wu = float(bandit["fixed_warmup_ratio"])

    refine_seeds = [int(x) for x in bandit["refine_seeds"]]
    if len(refine_seeds) < 2:
        raise RuntimeError("[HPO] hpo.bandit.refine_seeds must have at least 2 seeds")
    refine_seeds = refine_seeds[:2]
    n_seeds = max(1, len(refine_seeds))

    tags = _baseline_tags(cfg)

    grid_cfg = hpo_cfg["grid"]
    sens_epochs = int(grid_cfg["sensitivity_epochs"])
    grid_epochs = int(grid_cfg["grid_epochs"])
    baseline_sweep_epochs = int(grid_cfg["baseline_sweep_epochs"])
    baseline_refine_epochs = int(grid_cfg["baseline_refine_epochs"])

    max_retries = int(grid_cfg["max_retries"])

    full_lrs = make_lr_candidates_from_plan(plan)

    forced_knobs = grid_cfg["knobs"]
    if not isinstance(forced_knobs, list) or not forced_knobs:
        raise ValueError("cfg.hpo.grid.knobs must be a non-empty list")
    forced_knobs = [str(x) for x in forced_knobs]
    if any(k == "lora" for k in forced_knobs):
        raise RuntimeError("[HPO] forbidden knob name: 'lora'")

    drop_if_weight_lt = float(grid_cfg["drop_if_weight_lt"])
    sens_seed = int(grid_cfg["sens_seed"])
    rng = np.random.default_rng(int(grid_cfg["rng_seed"]))
    knob_specs_all = grid_cfg["knob_specs"]
    if not isinstance(knob_specs_all, dict):
        raise RuntimeError("[HPO] hpo.grid.knob_specs must be a dict")

    def _sample_value(kn: str) -> float:
        spec = knob_specs_all.get(kn, None)
        if not isinstance(spec, dict):
            raise RuntimeError(f"[HPO] missing knob spec: hpo.grid.knob_specs.{kn}")
        kind = str(spec["kind"])
        if kind == "choice":
            choices = spec.get("choices", None)
            if not isinstance(choices, list) or not choices:
                raise RuntimeError(f"[HPO] choice knob '{kn}' must provide non-empty choices")
            return float(rng.choice([float(x) for x in choices]))
        lo = float(spec["lo"])
        hi = float(spec["hi"])
        step = spec.get("step", None)
        v = float(rng.uniform(lo, hi))
        if step is not None:
            try:
                st = float(step)
                if st > 0:
                    v = float(np.round(v / st) * st)
            except Exception:
                pass
        return v

    # -----------------------
    # Build plan_status items deterministically
    # -----------------------
    st = _load_plan_status(status_path)

    items: List[PlanItem] = []
    idx = 0

    # baseline sweep (seed=12 fixed)
    for tag in tags:
        lora = _baseline_lora_from_method(cfg, tag)
        for i, lr in enumerate(full_lrs):
            trial_tag = f"bl_sweep__{tag}__i{i}"
            run_dir = str(hpo_run_dir / "trial_runs" / "baseline_sweep" / tag / f"i{i}" / "s12")
            override = {
                "method": {"name": tag, tag: {"lora": {"r": int(lora["r"]), "alpha": float(lora["alpha"]), "dropout": float(lora["dropout"])}}},
                "train": {"lr": float(lr), "warmup_ratio": float(fixed_wu), "epochs": int(baseline_sweep_epochs), "seed": 12},
            }
            items.append(PlanItem(idx=idx, stage="baseline_sweep", trial_tag=trial_tag, seed=12, run_dir=run_dir, override=override))
            idx += 1

    if not st.get("items"):
        for it in items:
            _update_item(st, it)
        _save_plan_status(status_path, st)

    # -----------------------
    # Executor loop (resume-safe)
    # -----------------------
    def _execute_one(it: Dict[str, Any]) -> None:
        nonlocal st
        trial_tag = str(it["trial_tag"])
        stage = str(it["stage"])
        run_dir = Path(str(it["run_dir"]))
        override = it.get("override", {})
        tries = int(it.get("tries", 0))

        if _is_done_in_trials_csv(trials_csv, trial_tag):
            it["status"] = "done"
            it["last_update"] = int(time.time())
            _save_plan_status(status_path, st)
            return

        if tries >= max_retries:
            return

        it["status"] = "running"
        it["tries"] = tries + 1
        it["last_update"] = int(time.time())
        _save_plan_status(status_path, st)

        rc = _run_one_trial(
            base_config_path=base_config_path,
            schedule_path=schedule_path,
            base_set_args=set_args,
            trial_override=override,
            trial_tag=trial_tag,
            stage=stage,
            hpo_run_dir=hpo_run_dir,
            score_weights=score_weights,
            run_dir=run_dir,
            overwrite_mode="resume",
        )
        it["last_rc"] = int(rc)
        it["last_update"] = int(time.time())

        if _is_done_in_trials_csv(trials_csv, trial_tag):
            it["status"] = "done"
        else:
            it["status"] = "failed"
        _save_plan_status(status_path, st)

    # -----------------------
    # 0) Run baseline sweep
    # -----------------------
    while True:
        st = _load_plan_status(status_path)
        nxt = _next_pending_item(st)
        if nxt is None:
            break
        if str(nxt.get("stage")) != "baseline_sweep":
            break
        _execute_one(nxt)

    st = _load_plan_status(status_path)
    pending_sweep = [it for it in st.get("items", []) if str(it.get("stage")) == "baseline_sweep" and str(it.get("status")) != "done"]
    if pending_sweep:
        print(f"[HPO][STOP] baseline_sweep not finished. pending={len(pending_sweep)}")
        return

    # -----------------------
    # 0.5) baseline refine materialize + run
    # -----------------------
    rows = _read_all_rows(trials_csv)

    best_lr_by_tag: Dict[str, float] = {}
    for tag in tags:
        best = None
        best_s = float("-inf")
        for r in rows:
            if str(r.get("stage", "")) != "baseline_sweep":
                continue
            if not str(r.get("trial_tag", "")).startswith(f"bl_sweep__{tag}__"):
                continue
            try:
                s = float(r.get("score", "-inf"))
            except Exception:
                continue
            if s > best_s:
                best_s = s
                best = r
        if best is None:
            best_lr_by_tag[tag] = float(cfg["train"]["lr"])
        else:
            try:
                cfgj = json.loads(best.get("trial_cfg_json", "{}"))
                best_lr_by_tag[tag] = float(cfgj["train"]["lr"])
            except Exception as e:
                raise RuntimeError(f"[HPO] failed to parse baseline sweep lr for tag={tag}") from e

    refine_lr_lists: Dict[str, List[float]] = {}
    for tag in tags:
        refine_lr_lists[tag] = lr_neighborhood_from_plan(plan, best_lr_by_tag[tag])

    st = _load_plan_status(status_path)
    existing = _plan_index(st)
    idx0 = max([int(it.get("idx", 0)) for it in st.get("items", [])] + [0]) + 1

    for tag in tags:
        lora = _baseline_lora_from_method(cfg, tag)
        for i, lr in enumerate(refine_lr_lists[tag]):
            for sd in refine_seeds:
                trial_tag = f"bl_refine__{tag}__i{i}__s{sd}"
                key = f"{trial_tag}__seed_{sd}"
                if key in existing:
                    continue
                run_dir = str(hpo_run_dir / "trial_runs" / "baseline_refine" / tag / f"i{i}" / f"s{sd}")
                override = {
                    "method": {"name": tag, tag: {"lora": {"r": int(lora["r"]), "alpha": float(lora["alpha"]), "dropout": float(lora["dropout"])}}},
                    "train": {"lr": float(lr), "warmup_ratio": float(fixed_wu), "epochs": int(baseline_refine_epochs), "seed": int(sd)},
                }
                _update_item(st, PlanItem(idx=idx0, stage="baseline_refine", trial_tag=trial_tag, seed=int(sd), run_dir=run_dir, override=override))
                idx0 += 1
    _save_plan_status(status_path, st)

    while True:
        st = _load_plan_status(status_path)
        nxt = _next_pending_item(st)
        if nxt is None:
            break
        if str(nxt.get("stage")) != "baseline_refine":
            break
        _execute_one(nxt)

    st = _load_plan_status(status_path)
    pending_ref = [it for it in st.get("items", []) if str(it.get("stage")) == "baseline_refine" and str(it.get("status")) != "done"]
    if pending_ref:
        print(f"[HPO][STOP] baseline_refine not finished. pending={len(pending_ref)}")
        return

    rows = _read_all_rows(trials_csv)
    best_refined_lr_by_tag: Dict[str, float] = {}
    for tag in tags:
        best_lr = best_lr_by_tag[tag]
        best_score = float("-inf")
        for i, lr in enumerate(refine_lr_lists[tag]):
            seed_tags = [f"bl_refine__{tag}__i{i}__s{sd}" for sd in refine_seeds]
            m = _score_mean_for_tags(trials_csv, seed_tags)
            if m is None:
                continue
            if float(m) > best_score:
                best_score = float(m)
                best_lr = float(lr)
        best_refined_lr_by_tag[tag] = float(best_lr)

    lrs_union: List[float] = []
    for tag in tags:
        lrs_union.extend(lr_neighborhood_from_plan(plan, best_refined_lr_by_tag[tag]))
    out_lr: List[float] = []
    seen_lr = set()
    for x in lrs_union:
        fx = float(x)
        if fx not in seen_lr:
            out_lr.append(fx)
            seen_lr.add(fx)
    lrs_union = out_lr or lr_neighborhood_from_plan(plan, float(cfg["train"]["lr"]))

    # -----------------------
    # 1) sensitivity stage
    # -----------------------
    budget_alloc = plan["budget"]["alloc"]
    sens_trials = int(budget_alloc["sensitivity_trials"])

    st = _load_plan_status(status_path)
    existing = _plan_index(st)
    idx0 = max([int(it.get("idx", 0)) for it in st.get("items", [])] + [0]) + 1

    emitted = 0
    j = 0

    ours_base_knobs = json.loads(json.dumps(cfg["method"]["ours"]))

    while emitted < sens_trials:
        for kn in forced_knobs:
            if emitted >= sens_trials:
                break

            trial_tag = f"sens__{kn}__j{j}__s{sens_seed}"
            key = f"{trial_tag}__seed_{sens_seed}"

            if key not in existing:
                lr = float(rng.choice(lrs_union))
                v = float(_sample_value(kn))
                run_dir = str(hpo_run_dir / "trial_runs" / "sensitivity" / kn / f"j{j}" / f"s{sens_seed}")

                ours_block = dict(ours_base_knobs)
                ours_block[kn] = float(v)

                override = {
                    "method": {"name": "ours", "ours": ours_block},
                    "train": {
                        "lr": float(lr),
                        "warmup_ratio": float(fixed_wu),
                        "epochs": int(sens_epochs),
                        "seed": int(sens_seed),
                    },
                }

                _update_item(
                    st,
                    PlanItem(
                        idx=idx0,
                        stage="sensitivity",
                        trial_tag=trial_tag,
                        seed=int(sens_seed),
                        run_dir=run_dir,
                        override=override,
                    ),
                )
                idx0 += 1
                existing[key] = 1  # ✅ 就在这里

            emitted += 1
        j += 1

    _save_plan_status(status_path, st)

    while True:
        st = _load_plan_status(status_path)
        nxt = _next_pending_item(st)
        if nxt is None:
            break
        if str(nxt.get("stage")) != "sensitivity":
            break
        _execute_one(nxt)

    st = _load_plan_status(status_path)
    pending_sens = [it for it in st.get("items", []) if str(it.get("stage")) == "sensitivity" and str(it.get("status")) != "done"]
    if pending_sens:
        print(f"[HPO][STOP] sensitivity not finished. pending={len(pending_sens)}")
        return

    # derive knob ranges + weights from sensitivity results
    rows = _read_all_rows(trials_csv)
    sens_records: List[Dict[str, Any]] = []
    for r in rows:
        if not str(r.get("trial_tag", "")).startswith("sens__"):
            continue
        try:
            cfg_json = json.loads(r.get("trial_cfg_json", "{}"))
            ours = cfg_json.get("method", {}).get("ours", {})
            if not isinstance(ours, dict):
                continue
            for kn in forced_knobs:
                if kn not in ours:
                    continue
                sens_records.append({"knob": str(kn), "value": float(ours[kn]), "score": float(r.get("score", "-inf"))})
        except Exception:
            continue

    knob_ranges: List[KnobRange] = []
    for kn in forced_knobs:
        spec = knob_specs_all.get(kn, None)
        if not isinstance(spec, dict):
            raise RuntimeError(f"[HPO] missing knob spec: hpo.grid.knob_specs.{kn}")
        kind = str(spec["kind"])
        recs = [x for x in sens_records if x["knob"] == kn]
        if not recs:
            continue

        scores = np.array([float(x["score"]) for x in recs], dtype=float)
        weight = float(np.max(scores) - float(np.median(scores)))
        if weight < drop_if_weight_lt:
            continue

        if kind == "choice":
            choices = spec.get("choices", None)
            if not isinstance(choices, list) or not choices:
                raise RuntimeError(f"[HPO] choice knob '{kn}' must provide non-empty choices")
            knob_ranges.append(KnobRange(name=kn, kind="choice", choices=[float(x) for x in choices], weight=weight))
            continue

        top_q = float(spec["top_quantile"])
        top_q = min(0.9, max(0.05, top_q))
        padding_ratio = float(spec["padding_ratio"])

        recs_sorted = sorted(recs, key=lambda x: float(x["score"]), reverse=True)
        take = max(1, int(math.ceil(len(recs_sorted) * top_q)))
        vals = np.array([float(x["value"]) for x in recs_sorted[:take]], dtype=float)

        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if lo > hi:
            lo, hi = hi, lo
        span = max(1e-12, hi - lo)
        lo2 = lo - padding_ratio * span
        hi2 = hi + padding_ratio * span

        clamp_lo = spec.get("clamp_lo", None)
        clamp_hi = spec.get("clamp_hi", None)
        if clamp_lo is not None:
            lo2 = max(float(clamp_lo), lo2)
        if clamp_hi is not None:
            hi2 = min(float(clamp_hi), hi2)

        knob_ranges.append(KnobRange(name=kn, kind="float", lo=float(lo2), hi=float(hi2), weight=weight))

    # -----------------------
    # 1.5) alpha probe (NOT budgeted)
    # -----------------------
    alpha_probe_cfg = grid_cfg["alpha_probe"]
    alpha_probe_enabled = bool(alpha_probe_cfg["enabled"])

    chosen_ours_alpha: Optional[float] = None
    alpha_probe_report: Dict[str, Any] = {"enabled": bool(alpha_probe_enabled), "stage": "alpha_probe"}

    if alpha_probe_enabled:
        a_r = float(_baseline_lora_from_method(cfg, "baseline_r")["alpha"])
        a_R = float(_baseline_lora_from_method(cfg, "baseline_R")["alpha"])
        alpha_cands = [a_r, a_R]

        lr_cands = [
            float(best_refined_lr_by_tag["baseline_r"]),
            float(best_refined_lr_by_tag["baseline_R"]),
        ]

        ap_seeds = [int(x) for x in alpha_probe_cfg["seeds"]]
        if len(ap_seeds) < 2:
            raise RuntimeError("[HPO] hpo.grid.alpha_probe.seeds must have at least 2 seeds")
        ap_seeds = ap_seeds[:2]

        ap_epochs = int(alpha_probe_cfg["epochs"])

        range_map: Dict[str, KnobRange] = {k.name: k for k in knob_ranges}
        fixed_assign: Dict[str, float] = {}
        for kn in forced_knobs:
            if kn in range_map:
                fixed_assign[kn] = float(_median_from_knob_range(range_map[kn]))
            else:
                spec = knob_specs_all.get(kn, None)
                if not isinstance(spec, dict):
                    raise RuntimeError(f"[HPO] missing knob spec: hpo.grid.knob_specs.{kn}")
                fixed_assign[kn] = float(_median_from_spec(spec))

        alpha_probe_report.update(
            {
                "alpha_candidates": [float(x) for x in alpha_cands],
                "lr_candidates": [float(x) for x in lr_cands],
                "seeds": [int(x) for x in ap_seeds],
                "epochs": int(ap_epochs),
                "fixed_assign": fixed_assign,
            }
        )

        st = _load_plan_status(status_path)
        existing = _plan_index(st)
        idx0 = max([int(it.get("idx", 0)) for it in st.get("items", [])] + [0]) + 1

        alpha_tags_by_aidx: List[List[str]] = [[] for _ in range(len(alpha_cands))]

        for ai, a in enumerate(alpha_cands):
            for lr in lr_cands:
                for sd in ap_seeds:
                    payload = {"a_idx": int(ai), "alpha": float(a), "lr": float(lr), "seed": int(sd)}
                    hh = _stable_hash(payload)
                    trial_tag = f"alpha_probe__a{ai}__lr{float(lr):.3e}__s{int(sd)}__{hh}"
                    alpha_tags_by_aidx[ai].append(trial_tag)

                    key = f"{trial_tag}__seed_{sd}"
                    if key in existing:
                        continue

                    run_dir = str(hpo_run_dir / "trial_runs" / "alpha_probe" / f"a{ai}" / f"lr_{float(lr):.3e}" / f"s{int(sd)}")

                    ours_block = json.loads(json.dumps(cfg["method"]["ours"]))
                    for k_assign, v_assign in fixed_assign.items():
                        if k_assign == "lora":
                            raise RuntimeError("[HPO] forbidden knob name: 'lora'")
                        ours_block[k_assign] = float(v_assign)
                    ours_block["lora"]["alpha"] = float(a)

                    override = {
                        "method": {"name": "ours", "ours": ours_block},
                        "train": {"lr": float(lr), "warmup_ratio": float(fixed_wu), "epochs": int(ap_epochs), "seed": int(sd)},
                    }

                    _update_item(st, PlanItem(idx=idx0, stage="alpha_probe", trial_tag=trial_tag, seed=int(sd), run_dir=run_dir, override=override))
                    idx0 += 1
        _save_plan_status(status_path, st)

        while True:
            st = _load_plan_status(status_path)
            nxt = _next_pending_item(st)
            if nxt is None:
                break
            if str(nxt.get("stage")) != "alpha_probe":
                break
            _execute_one(nxt)

        st = _load_plan_status(status_path)
        pending_ap = [it for it in st.get("items", []) if str(it.get("stage")) == "alpha_probe" and str(it.get("status")) != "done"]
        if pending_ap:
            print(f"[HPO][STOP] alpha_probe not finished. pending={len(pending_ap)}")
            return

        alpha_scores: List[Dict[str, Any]] = []
        for ai, a in enumerate(alpha_cands):
            m = _score_mean_for_tags(trials_csv, alpha_tags_by_aidx[ai])
            alpha_scores.append({"alpha": float(a), "tags": alpha_tags_by_aidx[ai], "mean_score": m})
        alpha_probe_report["results"] = alpha_scores

        best_m = float("-inf")
        best_a: Optional[float] = None
        for rec in alpha_scores:
            m = rec.get("mean_score", None)
            if m is None:
                continue
            if float(m) > best_m:
                best_m = float(m)
                best_a = float(rec.get("alpha", 0.0))
        chosen_ours_alpha = best_a
        alpha_probe_report["chosen_ours_alpha"] = chosen_ours_alpha
        alpha_probe_report["chosen_mean_score"] = best_m

    # -----------------------
    # 2) grid2 cartesian (budgeted)
    # -----------------------
    budget_alloc = plan["budget"]["alloc"]
    grid_trials = int(budget_alloc["grid_trials"])
    grid_configs = int(budget_alloc["grid_configs"])

    L = max(1, len(lrs_union))
    B = max(1, int(grid_configs))

    max_m_float = int(grid_cfg["max_m_float"])
    max_m_choice = int(grid_cfg["max_m_choice"])

    active_knobs, solve_meta = _solve_grid_points(L=L, B=B, knobs=knob_ranges, max_m_float=max_m_float, max_m_choice=max_m_choice)

    knob_values: Dict[str, List[float]] = {}
    for k in active_knobs:
        vs = _make_knob_values(k)
        if vs:
            knob_values[k.name] = vs

    knob_names = list(knob_values.keys())
    knob_lists = [knob_values[k] for k in knob_names]
    if not knob_names:
        knob_lists = [[]]

    if knob_names:
        all_assigns = [{k: float(v) for k, v in zip(knob_names, vals)} for vals in itertools.product(*knob_lists)]
    else:
        all_assigns = [dict()]

    st = _load_plan_status(status_path)
    existing = _plan_index(st)
    idx0 = max([int(it.get("idx", 0)) for it in st.get("items", [])] + [0]) + 1

    ours_base_knobs = json.loads(json.dumps(cfg["method"]["ours"]))

    def _grid_trial_base_tag(lr: float, assign: Dict[str, float]) -> str:
        payload = {"lr": float(lr), "assign": assign, "alpha": chosen_ours_alpha}
        return f"grid2__lr{lr:.3e}__{_stable_hash(payload)}"

    emitted = 0
    for lr in lrs_union:
        for assign in all_assigns:
            if emitted >= B:
                break
            base_tag = _grid_trial_base_tag(float(lr), assign)

            for sd in refine_seeds:
                trial_tag = f"{base_tag}__s{sd}"
                key = f"{trial_tag}__seed_{sd}"
                if key in existing:
                    continue
                run_dir = str(hpo_run_dir / "trial_runs" / "grid2" / base_tag / f"s{sd}")

                method_block: Dict[str, Any] = {"name": "ours", "ours": dict(ours_base_knobs, **assign)}
                if chosen_ours_alpha is not None:
                    method_block["ours"].setdefault("lora", {})
                    method_block["ours"]["lora"]["alpha"] = float(chosen_ours_alpha)

                override = {
                    "method": method_block,
                    "train": {"lr": float(lr), "warmup_ratio": float(fixed_wu), "epochs": int(grid_epochs), "seed": int(sd)},
                }

                _update_item(st, PlanItem(idx=idx0, stage="grid2", trial_tag=trial_tag, seed=int(sd), run_dir=run_dir, override=override))
                idx0 += 1

            agg_tag = base_tag
            keyagg = f"{agg_tag}__agg"
            if keyagg not in existing:
                run_dir = str(hpo_run_dir / "trial_runs" / "grid2" / base_tag / "agg")

                # ✅ override 也统一写入 method.ours.lora.alpha（别再 method.lora 了）
                method_block2: Dict[str, Any] = {"name": "ours", "ours": dict(ours_base_knobs, **assign)}
                if chosen_ours_alpha is not None:
                    method_block2["ours"].setdefault("lora", {})
                    method_block2["ours"]["lora"]["alpha"] = float(chosen_ours_alpha)

                override2 = {"method": method_block2, "train": {"lr": float(lr), "warmup_ratio": float(fixed_wu), "epochs": int(grid_epochs)}}
                _update_item(st, PlanItem(idx=idx0, stage="grid2_agg", trial_tag=agg_tag, seed=None, run_dir=run_dir, override=override2))
                idx0 += 1

            emitted += 1
        if emitted >= B:
            break

    _save_plan_status(status_path, st)

    while True:
        st = _load_plan_status(status_path)
        nxt = _next_pending_item(st)
        if nxt is None:
            break
        if str(nxt.get("stage")) not in {"grid2"}:
            break
        _execute_one(nxt)

    st = _load_plan_status(status_path)
    items_list = st.get("items", []) if isinstance(st.get("items", []), list) else []
    grid_agg_items = [it for it in items_list if str(it.get("stage")) == "grid2_agg"]

    for it in sorted(grid_agg_items, key=lambda x: int(x.get("idx", 10**9))):
        base_tag = str(it["trial_tag"])
        if _is_done_in_trials_csv(trials_csv, base_tag):
            it["status"] = "done"
            it["last_update"] = int(time.time())
            _save_plan_status(status_path, st)
            continue

        seeds_done = True
        for sd in refine_seeds:
            seed_tag = f"{base_tag}__s{sd}"
            if not _is_done_in_trials_csv(trials_csv, seed_tag):
                seeds_done = False
                break
        if not seeds_done:
            continue

        ov = it.get("override", {})
        trial_cfg_json = json.dumps(ov, sort_keys=True)

        _ = _aggregate_seed_trial(
            trials_csv=trials_csv,
            base_trial_tag=base_tag,
            stage="grid2.agg",
            seeds=refine_seeds,
            trial_cfg_json=trial_cfg_json,
        )
        if _is_done_in_trials_csv(trials_csv, base_tag):
            it["status"] = "done"
            it["last_update"] = int(time.time())
            _save_plan_status(status_path, st)

    st = _load_plan_status(status_path)
    items_list = st.get("items", []) if isinstance(st.get("items", []), list) else []
    pending_grid = [it for it in items_list if str(it.get("stage")) in {"grid2", "grid2_agg"} and str(it.get("status")) != "done"]
    if pending_grid:
        print(f"[HPO][STOP] grid2 not finished. pending={len(pending_grid)}")
        return

    # -----------------------
    # 3) bayes refine (optional, budgeted)
    # -----------------------
    use_bayes = bool(cfg["hpo"]["use_bayes"])
    bayes_trials = int(budget_alloc["bayes_trials"])
    bayes_configs = int(budget_alloc["bayes_configs"])

    bayes_report: Dict[str, Any] = {"enabled": bool(use_bayes), "configs": int(bayes_configs)}
    bayes_best_row: Optional[Dict[str, Any]] = None

    if use_bayes and bayes_configs > 0:
        rows = _read_all_rows(trials_csv)
        data_X: List[List[float]] = []
        data_y: List[float] = []
        seen_cfg = set()

        knob_space = active_knobs
        choice_maps = {k.name: list(k.choices or []) for k in knob_space if k.kind == "choice"}

        lr_min = float(plan["lr"]["min_lr_ext"])
        lr_max = float(plan["lr"]["max_lr_ext"])

        def _parse_cfg_from_agg_row(r: Dict[str, Any]) -> Optional[Tuple[float, Dict[str, float], float]]:
            try:
                cfgj = json.loads(r.get("trial_cfg_json", "{}"))
                tr = cfgj.get("train", {})
                md = cfgj.get("method", {})
                lr = float(tr.get("lr"))
                ours = md.get("ours", {})
                if not isinstance(ours, dict):
                    ours = {}
                assign = {k.name: float(ours.get(k.name)) for k in knob_space if k.name in ours}
                score = float(r.get("score", "-inf"))
                return lr, assign, score
            except Exception:
                return None

        for r in rows:
            if str(r.get("stage", "")) != "grid2.agg":
                continue
            parsed = _parse_cfg_from_agg_row(r)
            if parsed is None:
                continue
            lr, assign, score = parsed
            key = json.dumps({"lr": lr, "assign": assign}, sort_keys=True)
            if key in seen_cfg:
                continue
            seen_cfg.add(key)
            vec = _encode_config(lr=lr, knobs=assign, knob_space=knob_space, choice_maps=choice_maps, lr_min=lr_min, lr_max=lr_max)
            data_X.append(vec)
            data_y.append(score)

        X = np.asarray(data_X, dtype=float)
        y = np.asarray(data_y, dtype=float)

        bo_rng = np.random.default_rng(int(grid_cfg["bayes_rng_seed"]))
        pool = int(grid_cfg["bayes_pool"])
        length = float(grid_cfg["bayes_gp_length"])
        var = float(grid_cfg["bayes_gp_var"])
        noise = float(grid_cfg["bayes_gp_noise"])
        xi = float(grid_cfg["bayes_ei_xi"])

        lr_candidates = list(lrs_union)

        st = _load_plan_status(status_path)
        existing = _plan_index(st)
        idx0 = max([int(it.get("idx", 0)) for it in st.get("items", [])] + [0]) + 1

        proposed: List[Dict[str, Any]] = []
        for t in range(bayes_configs):
            pool_lrs: List[float] = []
            pool_assigns: List[Dict[str, float]] = []
            pool_vecs: List[List[float]] = []

            for _ in range(pool):
                lr_s, assign_s = _random_sample_config(bo_rng, lr_candidates=lr_candidates, knob_space=knob_space)
                vec = _encode_config(lr=lr_s, knobs=assign_s, knob_space=knob_space, choice_maps=choice_maps, lr_min=lr_min, lr_max=lr_max)
                pool_lrs.append(lr_s)
                pool_assigns.append(assign_s)
                pool_vecs.append(vec)

            Xcand = np.asarray(pool_vecs, dtype=float)
            best_so_far = float(np.max(y)) if y.size else float("-inf")
            mu, sigma = _gp_posterior(X, y, Xcand, length=length, var=var, noise=noise)
            ei = _expected_improvement(mu, sigma, best=best_so_far, xi=xi)

            pick = int(np.argmax(ei))
            lr_pick = float(pool_lrs[pick])
            assign_pick = dict(pool_assigns[pick])

            payload = {"lr": lr_pick, "assign": assign_pick, "alpha": chosen_ours_alpha, "t": t}
            base_tag = f"bayes__t{t}__{_stable_hash(payload)}"

            for sd in refine_seeds:
                trial_tag = f"{base_tag}__s{sd}"
                key = f"{trial_tag}__seed_{sd}"
                if key not in existing:
                    run_dir = str(hpo_run_dir / "trial_runs" / "bayes" / base_tag / f"s{sd}")

                    method_block: Dict[str, Any] = {"name": "ours", "ours": dict(ours_base_knobs, **assign_pick)}
                    if chosen_ours_alpha is not None:
                        method_block["ours"].setdefault("lora", {})
                        method_block["ours"]["lora"]["alpha"] = float(chosen_ours_alpha)

                    override = {
                        "method": method_block,
                        "train": {"lr": float(lr_pick), "warmup_ratio": float(fixed_wu), "epochs": int(grid_epochs), "seed": int(sd)},
                    }
                    _update_item(st, PlanItem(idx=idx0, stage="bayes", trial_tag=trial_tag, seed=int(sd), run_dir=run_dir, override=override))
                    idx0 += 1

            agg_tag = base_tag
            keyagg = f"{agg_tag}__agg"
            if keyagg not in existing:
                run_dir = str(hpo_run_dir / "trial_runs" / "bayes" / base_tag / "agg")

                method_block2: Dict[str, Any] = {"name": "ours", "ours": dict(ours_base_knobs, **assign_pick)}
                if chosen_ours_alpha is not None:
                    method_block2["ours"].setdefault("lora", {})
                    method_block2["ours"]["lora"]["alpha"] = float(chosen_ours_alpha)

                override2 = {"method": method_block2, "train": {"lr": float(lr_pick), "warmup_ratio": float(fixed_wu), "epochs": int(grid_epochs)}}
                _update_item(st, PlanItem(idx=idx0, stage="bayes_agg", trial_tag=agg_tag, seed=None, run_dir=run_dir, override=override2))
                idx0 += 1

            proposed.append({"base_tag": base_tag, "lr": lr_pick, "assign": assign_pick, "ei": float(ei[pick])})

        bayes_report["proposals"] = proposed
        _save_plan_status(status_path, st)

        while True:
            st = _load_plan_status(status_path)
            nxt = _next_pending_item(st)
            if nxt is None:
                break
            if str(nxt.get("stage")) != "bayes":
                break
            _execute_one(nxt)

        st = _load_plan_status(status_path)
        items_list = st.get("items", []) if isinstance(st.get("items", []), list) else []
        bayes_agg_items = [it for it in items_list if str(it.get("stage")) == "bayes_agg"]

        for it in sorted(bayes_agg_items, key=lambda x: int(x.get("idx", 10**9))):
            base_tag = str(it["trial_tag"])
            if _is_done_in_trials_csv(trials_csv, base_tag):
                it["status"] = "done"
                it["last_update"] = int(time.time())
                _save_plan_status(status_path, st)
                continue

            seeds_done = True
            for sd in refine_seeds:
                seed_tag = f"{base_tag}__s{sd}"
                if not _is_done_in_trials_csv(trials_csv, seed_tag):
                    seeds_done = False
                    break
            if not seeds_done:
                continue

            ov = it.get("override", {})
            trial_cfg_json = json.dumps(ov, sort_keys=True)

            _ = _aggregate_seed_trial(
                trials_csv=trials_csv,
                base_trial_tag=base_tag,
                stage="bayes.agg",
                seeds=refine_seeds,
                trial_cfg_json=trial_cfg_json,
            )
            if _is_done_in_trials_csv(trials_csv, base_tag):
                it["status"] = "done"
                it["last_update"] = int(time.time())
                _save_plan_status(status_path, st)

        st = _load_plan_status(status_path)
        items_list = st.get("items", []) if isinstance(st.get("items", []), list) else []
        pending_bayes = [it for it in items_list if str(it.get("stage")) in {"bayes", "bayes_agg"} and str(it.get("status")) != "done"]
        if pending_bayes:
            print(f"[HPO][STOP] bayes not finished. pending={len(pending_bayes)}")
            return

        rows = _read_all_rows(trials_csv)
        bayes_best_row = _best_row(rows, stage_prefix="bayes.agg", only_aggregate=True)

    # -----------------------
    # Final selection + persist best_hparams.json + bundle
    # -----------------------
    rows = _read_all_rows(trials_csv)
    best_grid = _best_row(rows, stage_prefix="grid2.agg", only_aggregate=True)

    best_overall = best_grid
    if bayes_best_row is not None:
        try:
            if float(bayes_best_row.get("score", "-inf")) >= float(best_grid.get("score", "-inf")) if best_grid else True:
                best_overall = bayes_best_row
        except Exception:
            best_overall = bayes_best_row

    best_curves_dir.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        best = None
        best_s = float("-inf")
        for r in rows:
            if not str(r.get("trial_tag", "")).startswith(f"bl_refine__{tag}__"):
                continue
            try:
                s = float(r.get("score", "-inf"))
            except Exception:
                continue
            if s > best_s:
                best_s = s
                best = r
        if best:
            _copy_metrics_csv(best, best_curves_dir / f"best_{tag}.csv")

    if best_overall:
        _copy_metrics_csv(best_overall, best_curves_dir / "best_ours.csv")

    if best_overall:
        best_obj = {
            "best": best_overall,
            "best_source": "bayes" if (bayes_best_row is not None and best_overall == bayes_best_row) else "grid2",
            "weights": {"w_max": w_max, "w_final": w_final, "w_avg": w_avg},
            "plan_path": str(plan_path),
            "status_path": str(status_path),
            "hpo_dir": str(hpo_run_dir),
            "baseline_best_lr_refined": best_refined_lr_by_tag,
            "lrs_union": [float(x) for x in lrs_union],
            "chosen_ours_alpha": chosen_ours_alpha,
            "alpha_probe": alpha_probe_report,
            "grid_solve_meta": solve_meta,
            "use_bayes": bool(use_bayes),
            "bayes_report": bayes_report,
        }
        _atomic_write_json(best_path, best_obj)
        print(f"[HPO][OK] best saved: {best_path}")

        bundle = {
            "hpo_dir": str(hpo_run_dir),
            "best_path": str(best_path),
            "plan_path": str(plan_path),
            "status_path": str(status_path),
            "trials_csv": str(trials_csv),
            "best_curves_dir": str(best_curves_dir),
            "fixed_warmup_ratio": float(fixed_wu),
            "refine_seeds": refine_seeds,
            "baseline_tags": tags,
            "baseline_lora": {t: _baseline_lora_from_method(cfg, t) for t in tags},
            "baseline_best_lr_refined": best_refined_lr_by_tag,
            "lrs_union": [float(x) for x in lrs_union],
            "chosen_ours_alpha": chosen_ours_alpha,
            "alpha_probe": alpha_probe_report,
            "active_knobs": [
                {"name": k.name, "kind": k.kind, "lo": k.lo, "hi": k.hi, "choices": k.choices, "weight": float(k.weight), "m": int(k.m)}
                for k in active_knobs
            ],
            "knob_values": knob_values,
            "grid_solve_meta": solve_meta,
            "use_bayes": bool(use_bayes),
            "bayes_report": bayes_report,
            "config_snapshot": str(snapshot_path),
            "config_resolved": cfg,
        }
        _atomic_write_json(bundle_path, bundle)
        print(f"[HPO][OK] bundle saved: {bundle_path}")
    else:
        print("[HPO][WARN] No aggregate rows found; cannot choose best.")
        print(f"[HPO][INFO] trials.csv: {trials_csv}")

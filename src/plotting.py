# src/plotting.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# parsing helpers
# -------------------------
def _safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def _safe_int(x: object) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return int(float(s))
    except Exception:
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _is_long_format(rows: List[Dict[str, str]]) -> bool:
    if not rows:
        return False
    cols = set(rows[0].keys())
    return ("key" in cols) and ("value" in cols) and ("step" in cols)


def _collect_long(rows: List[Dict[str, str]]) -> Dict[str, Dict[int, List[float]]]:
    out: Dict[str, Dict[int, List[float]]] = {}
    for r in rows:
        step = _safe_int(r.get("step", ""))
        key = (r.get("key", "") or "").strip()
        val = _safe_float(r.get("value", ""))
        if step is None or not key or val is None:
            continue
        out.setdefault(key, {}).setdefault(step, []).append(val)
    return out


def _collect_wide(rows: List[Dict[str, str]]) -> Dict[str, Dict[int, List[float]]]:
    out: Dict[str, Dict[int, List[float]]] = {}
    if not rows:
        return out

    cols = [c for c in rows[0].keys() if c not in (None, "", "step")]
    cols = [str(c).strip() for c in cols if str(c).strip()]

    for r in rows:
        step = _safe_int(r.get("step", ""))
        if step is None:
            continue
        for c in cols:
            v = _safe_float(r.get(c, ""))
            if v is None:
                continue
            out.setdefault(c, {}).setdefault(step, []).append(v)
    return out


def _load_metrics_csv(path: Path) -> Dict[str, Dict[int, List[float]]]:
    rows = _read_csv_rows(path)
    if not rows:
        return {}
    if _is_long_format(rows):
        return _collect_long(rows)
    return _collect_wide(rows)


def _merge_metric_maps(maps: List[Dict[str, Dict[int, List[float]]]]) -> Dict[str, Dict[int, List[float]]]:
    merged: Dict[str, Dict[int, List[float]]] = {}
    for mp in maps:
        for metric, per_step in mp.items():
            m = merged.setdefault(metric, {})
            for step, vals in per_step.items():
                m.setdefault(step, []).extend(list(vals))
    return merged


def _fname(metric: str) -> str:
    return metric.replace("/", "_").replace(" ", "_").replace(":", "_")


# -------------------------
# grouping helpers
# -------------------------
def _infer_group_from_filename(p: Path) -> Optional[str]:
    # expected names:
    # baseline_r_s2.csv, baseline_R_s2.csv, ours_s2.csv
    # OR best_baseline_r.csv, best_baseline_R.csv, best_ours.csv
    name = p.name
    if name.startswith("baseline_r_") or name.startswith("best_baseline_r"):
        return "baseline_r"
    if name.startswith("baseline_R_") or name.startswith("best_baseline_R"):
        return "baseline_R"
    if name.startswith("ours_") or name.startswith("best_ours"):
        return "ours"
    return None


def _collect_group_metric_maps(curves_dir: Path) -> Dict[str, Dict[str, Dict[int, List[float]]]]:
    """
    Return:
      group -> (metric -> step -> [values])
    where values are pooled across seeds.
    """
    curves_dir = Path(curves_dir)
    group_files: Dict[str, List[Path]] = {"baseline_r": [], "baseline_R": [], "ours": []}

    for p in curves_dir.glob("*.csv"):
        g = _infer_group_from_filename(p)
        if g in group_files:
            group_files[g].append(p)

    group_metric_maps: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
    for g, files in group_files.items():
        maps = []
        for f in files:
            mp = _load_metrics_csv(f)
            if mp:
                maps.append(mp)
        group_metric_maps[g] = _merge_metric_maps(maps) if maps else {}
    return group_metric_maps


# -------------------------
# plotting
# -------------------------
def _plot_compare_one_metric(
    metric: str,
    group_to_per_step: Dict[str, Dict[int, List[float]]],
    out_path: Path,
) -> None:
    # union of steps across groups
    all_steps = set()
    for per_step in group_to_per_step.values():
        all_steps |= set(per_step.keys())
    steps = sorted(all_steps)
    if not steps:
        return

    plt.figure()

    # consistent order
    order = ["baseline_r", "baseline_R", "ours"]
    for g in order:
        per_step = group_to_per_step.get(g, {}) or {}
        if not per_step:
            continue

        means = []
        stds = []
        xs = []
        for s in steps:
            vals = per_step.get(s, None)
            if not vals:
                # skip missing step for this group
                continue
            xs.append(s)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        if not xs:
            continue

        mean = np.array(means, dtype=float)
        std = np.array(stds, dtype=float)

        plt.plot(xs, mean, label=f"{g} mean")
        # if only one seed, std will be 0 -> still ok
        plt.fill_between(xs, mean - std, mean + std, alpha=0.2, label=f"{g} ±std")

    plt.title(metric)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_compare(curves_dir: Path, out_dir: Path) -> None:
    """
    Build per-group metric maps and plot one figure per metric comparing 3 groups.
    """
    curves_dir = Path(curves_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_metric_maps = _collect_group_metric_maps(curves_dir)

    # metric union
    metrics = set()
    for gmap in group_metric_maps.values():
        metrics |= set(gmap.keys())
    metrics = sorted(metrics)

    for metric in metrics:
        g2ps = {
            g: group_metric_maps.get(g, {}).get(metric, {}) or {}
            for g in ["baseline_r", "baseline_R", "ours"]
        }
        out_path = out_dir / f"{_fname(metric)}.png"
        _plot_compare_one_metric(metric, g2ps, out_path)


def plot_run(run_dir: Path) -> None:
    """
    Smart plotting for your pipeline:
      - Prefer all_curves/ (final: per-seed curves) => plots_compare/
      - Else use best_curves/ => plots_compare_best/
      - Else fallback to old behavior (NOT recommended)
    """
    run_dir = Path(run_dir)
    all_curves = run_dir / "all_curves"
    best_curves = run_dir / "best_curves"

    if all_curves.exists():
        plot_compare(all_curves, run_dir / "plots_compare")
        return

    if best_curves.exists():
        plot_compare(best_curves, run_dir / "plots_compare_best")
        return

    # fallback: legacy (mix everything) - keep it for debugging
    # (but don't trust it for paper plots)
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_maps = []
    for p in run_dir.rglob("metrics.csv"):
        mp = _load_metrics_csv(p)
        if mp:
            metric_maps.append(mp)
    merged = _merge_metric_maps(metric_maps)

    for metric, per_step in merged.items():
        out_path = out_dir / f"{_fname(metric)}.png"
        # treat as "single group"
        _plot_compare_one_metric(metric, {"all": per_step}, out_path)


def run_plotting(runs_root: str) -> None:
    root = Path(runs_root)
    if not root.exists():
        print(f"[plotting] runs_root not found: {root}")
        return

    # If root itself looks like a run dir, plot it.
    if (root / "all_curves").exists() or (root / "best_curves").exists() or any(root.glob("**/metrics.csv")):
        plot_run(root)
        return

    # Otherwise treat root as runs folder
    for d in root.iterdir():
        if d.is_dir():
            plot_run(d)

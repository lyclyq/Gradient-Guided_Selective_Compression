# src/plotting.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _group_rows(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[int, List[float]]]:
    grouped: Dict[Tuple[str, str], Dict[int, List[float]]] = {}
    for row in rows:
        key = row.get("key", "")
        value = float(row.get("value", 0.0))
        step = int(float(row.get("step", 0)))
        # key like "val/acc" or "train/loss"
        if "/" in key:
            split, metric = key.split("/", 1)
        else:
            split, metric = "", key
        grouped.setdefault((split, metric), {}).setdefault(step, []).append(value)
    return grouped


def _plot_curve(steps: List[int], mean: np.ndarray, band: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure()
    plt.plot(steps, mean, label="mean")
    plt.fill_between(steps, mean - band, mean + band, alpha=0.2, label="±std")
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_run(run_dir: Path) -> None:
    plots_dir = run_dir / "plots"
    rows: List[Dict[str, str]] = []

    # recursive: supports hpo/trial/seed structures
    for m in run_dir.glob("**/metrics.csv"):
        rows.extend(_load_metrics(m))

    if not rows:
        return

    grouped = _group_rows(rows)
    for (split, metric), per_step in grouped.items():
        steps = sorted(per_step.keys())
        values = [per_step[s] for s in steps]
        mean = np.array([np.mean(v) for v in values])
        std = np.array([np.std(v) for v in values])
        title = f"{split}/{metric}" if split else metric
        out_path = plots_dir / f"{split}_{metric}.png" if split else plots_dir / f"{metric}.png"
        _plot_curve(steps, mean, std, out_path, title)

    _plot_convergence(run_dir, plots_dir, rows, threshold=0.9)


def _plot_convergence(run_dir: Path, plots_dir: Path, rows: List[Dict[str, str]], threshold: float) -> None:
    grouped = _group_rows(rows)
    if ("val", "acc") not in grouped and ("val", "accuracy") not in grouped:
        return
    key = ("val", "acc") if ("val", "acc") in grouped else ("val", "accuracy")
    val = grouped[key]
    steps = sorted(val.keys())
    first_hit = None
    for s in steps:
        if np.mean(val[s]) >= threshold:
            first_hit = s
            break
    if first_hit is None:
        return
    plt.figure()
    plt.axvline(first_hit, linestyle="--")
    plt.title(f"convergence step @ {threshold}")
    plt.xlabel("step")
    plt.ylabel("val acc")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "convergence.png")
    plt.close()


def run_plotting(runs_root: str) -> None:
    root = Path(runs_root)
    if not root.exists():
        return
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        plot_run(run_dir)

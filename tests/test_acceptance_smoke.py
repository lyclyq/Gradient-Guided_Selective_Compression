# tests/test_acceptance_smoke.py
from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import pytest


def _run(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ},
    )
    return p.returncode, p.stdout


def _collect_csvs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.csv"))


def _read_first_good_curve(csvs: List[Path]) -> pd.DataFrame:
    """
    Find first CSV that contains (step,val_acc) compatible columns.
    """
    for p in csvs[::-1]:  # newest first
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        cols = {str(c).strip(): c for c in df.columns}
        step_col = None
        for cand in ["step", "global_step", "iter", "iteration"]:
            if cand in cols:
                step_col = cols[cand]
                break

        val_col = None
        for cand in ["val/acc", "val_acc", "validation/acc", "validation_acc", "eval/acc", "eval_acc"]:
            if cand in cols:
                val_col = cols[cand]
                break

        if step_col is None or val_col is None:
            continue

        out = df[[step_col, val_col]].copy()
        out.columns = ["step", "val_acc"]
        out["step"] = pd.to_numeric(out["step"], errors="coerce")
        out["val_acc"] = pd.to_numeric(out["val_acc"], errors="coerce")
        out = out.dropna(subset=["step", "val_acc"]).sort_values("step")
        if len(out) > 0:
            return out

    raise AssertionError(
        "No CSV found with (step,val_acc). "
        "Check logger output columns and io.root path."
    )


@pytest.mark.parametrize(
    "name,extra_sets",
    [
        ("baseline_r32", {"method.name": "baseline", "method.lora.r": "32"}),
        ("baseline_R128", {"method.name": "baseline", "method.lora.r": "128"}),
        ("ours_full", {"method.name": "ours"}),
        (
            "ours_ablate",
            {
                "method.name": "ours",
                # 下面这俩 key 你要按你项目真实 key 改。
                # 我这里用你 yaml 里预留的 ablate 子树：
                "method.ours.ablate.cagrad": "true",
                "method.ours.ablate.interp": "true",
            },
        ),
    ],
)
def test_train_valid_smoke(tmp_path: Path, name: str, extra_sets: Dict[str, str]) -> None:
    """
    自动化验收（smoke）：
    - 能 train+valid 跑通（exit code = 0）
    - io.root 下产出至少一个 CSV
    - CSV 至少包含 step & val_acc（或兼容列名）
    """
    repo = Path(__file__).resolve().parents[1]
    io_root = tmp_path / "runs"
    io_root.mkdir(parents=True, exist_ok=True)

    base_sets: Dict[str, str] = {
        "task.name": "glue/rte",
        "train.epochs": "1",
        "train.batch_size": "8",
        "train.lr": "2e-5",
        "train.warmup_ratio": "0.1",
        "train.seed": "12",
        "io.root": str(io_root.as_posix()),
        "log.csv": "true",
        "debug.enabled": "true",
    }

    # only ours needs gate config
    # 5 degrees trigger => tau_D = 1 - cos(5deg) ~ 0.0038053
    # (use your real key; from your yaml it's method.ours.trigger_gate0.tau_D)
    extra = dict(extra_sets)
    if extra.get("method.name") == "ours":
        base_sets["method.ours.trigger_gate0.tau_D"] = "0.0038053"

    base_sets.update(extra)

    cmd = [
        sys.executable,
        "scripts/run.py",
        "train",
        "--config",
        "configs/base.yaml",
        "--trial_tag",
        f"smoke_{name}",
    ]
    for k, v in base_sets.items():
        cmd += ["--set", f"{k}={v}"]

    rc, out = _run(cmd, cwd=repo)

    assert rc == 0, f"Run failed ({name}). Output:\n{out}"

    csvs = _collect_csvs(io_root)
    assert len(csvs) > 0, f"No CSV produced for ({name}). Output:\n{out}"

    curve = _read_first_good_curve(csvs)
    assert "step" in curve.columns and "val_acc" in curve.columns
    assert len(curve) >= 1, f"Curve empty for ({name}). CSVs: {[str(x) for x in csvs[-5:]]}"

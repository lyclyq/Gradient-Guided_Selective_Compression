# src/artifacts.py
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _hash_config(cfg: Dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:8]


def make_run_name(cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> str:
    """
    Stable, audit-friendly run name.
    """
    extra = extra or {}
    stage = str(cfg.get("stage", "debug"))
    task = str(cfg.get("task", {}).get("name", "task")).replace("/", "_")
    model = str(cfg.get("model", {}).get("name", "model")).replace("/", "_")
    method = str(cfg.get("method", {}).get("name", "method"))

    lr = cfg.get("train", {}).get("lr", None)
    ep = cfg.get("train", {}).get("epochs", None)
    bs = cfg.get("train", {}).get("batch_size", None)

    # optional tags
    trial_tag = extra.get("trial_tag")
    seed = extra.get("seed")
    kind = extra.get("kind")

    ts = time.strftime("%Y%m%d-%H%M%S")
    h = _hash_config(cfg)

    parts = [
        stage,
        task,
        model,
        method,
        f"ep{ep}" if ep is not None else None,
        f"bs{bs}" if bs is not None else None,
        f"lr{lr}" if lr is not None else None,
        f"seed{seed}" if seed is not None else None,
        f"{kind}" if kind else None,
        f"{trial_tag}" if trial_tag else None,
        ts,
        h,
    ]
    parts = [p for p in parts if p]
    return "__".join(parts)


def prepare_run_dir(root: str, run_name: str, overwrite: str = "ask") -> Tuple[Path, bool]:
    """
    Returns (run_dir, resumed)
    overwrite: ask | force | resume
    """
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    run_dir = root_p / run_name

    if run_dir.exists():
        if overwrite == "resume":
            return run_dir, True
        if overwrite == "force":
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir, False
        # ask
        ans = input(f"[artifacts] {run_dir} exists. Overwrite? [y/N] ").strip().lower()
        if ans in {"y", "yes"}:
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir, False
        return run_dir, True

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, False


# Compatibility helper (some code may call ensure_run_dir)
def ensure_run_dir(cfg: Dict[str, Any]) -> Path:
    io = cfg.get("io", {})
    root = io.get("root", "runs")
    overwrite = io.get("overwrite", "ask")
    run_name = make_run_name(cfg)
    run_dir, _ = prepare_run_dir(root=root, run_name=run_name, overwrite=overwrite)
    return run_dir

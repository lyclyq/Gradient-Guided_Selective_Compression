# /home/lyclyq/Optimization/grad-shake-align/src/artifacts.py
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
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
    STRICT: no fallbacks. Missing keys => crash.

    NOTE:
      - stage is injected by scripts/run.py (command context), or set by schedule YAML.
    """
    extra = extra or {}

    stage = str(cfg["stage"])
    task = str(cfg["task"]["name"]).replace("/", "_")
    model = str(cfg["model"]["name"]).replace("/", "_")
    method = str(cfg["method"]["name"])

    lr = cfg["train"]["lr"]
    ep = cfg["train"]["epochs"]
    bs = cfg["train"]["batch_size"]

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
        f"ep{ep}",
        f"bs{bs}",
        f"lr{lr}",
        f"seed{seed}" if seed is not None else None,
        f"{kind}" if kind else None,
        f"{trial_tag}" if trial_tag else None,
        ts,
        h,
    ]
    parts = [p for p in parts if p]
    return "__".join(parts)


def _is_interactive() -> bool:
    """True when we can safely prompt user."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def prepare_run_dir(root: str, run_name: str, overwrite: str = "ask") -> Tuple[Path, bool]:
    """
    Returns (run_dir, resumed)

    overwrite:
      - ask: prompt user if interactive; otherwise auto resume
      - force: delete and recreate
      - resume: keep existing
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

        # overwrite == "ask"
        if not _is_interactive():
            return run_dir, True

        ans = input(f"[artifacts] {run_dir} exists. Overwrite? [y/N] ").strip().lower()
        if ans in {"y", "yes"}:
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir, False
        return run_dir, True

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, False


def resolve_run_dir(cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Tuple[Path, bool, str]:
    """
    Resolve run_dir with priority:
      1) cfg.io.run_dir (absolute or relative)
      2) cfg.io.root + make_run_name(cfg, extra)

    STRICT: no root fallback. cfg must contain io.root/io.overwrite.
    Returns (run_dir, resumed, run_name)
    """
    extra = extra or {}
    io = cfg["io"]

    overwrite = str(io["overwrite"])
    run_dir_override = io.get("run_dir", None)

    if run_dir_override:
        run_dir = Path(str(run_dir_override)).expanduser()
        if not run_dir.is_absolute():
            run_dir = Path(os.getcwd()) / run_dir

        run_dir.parent.mkdir(parents=True, exist_ok=True)

        if run_dir.exists():
            if overwrite == "force":
                shutil.rmtree(run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                return run_dir, False, run_dir.name

            if overwrite == "ask" and _is_interactive():
                ans = input(f"[artifacts] {run_dir} exists. Overwrite? [y/N] ").strip().lower()
                if ans in {"y", "yes"}:
                    shutil.rmtree(run_dir)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    return run_dir, False, run_dir.name
                return run_dir, True, run_dir.name

            # resume OR ask(non-interactive)
            return run_dir, True, run_dir.name

        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, False, run_dir.name

    root = str(io["root"])
    run_name = make_run_name(cfg, extra=extra)
    run_dir, resumed = prepare_run_dir(root=root, run_name=run_name, overwrite=overwrite)
    return run_dir, resumed, run_name


def ensure_run_dir(cfg: Dict[str, Any]) -> Path:
    run_dir, _resumed, _name = resolve_run_dir(cfg, extra=None)
    return run_dir

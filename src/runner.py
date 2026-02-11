# /home/lyclyq/Optimization/grad-shake-align/src/runner.py
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader

from .artifacts import make_run_name, prepare_run_dir
from .data_glue import load_glue
from .lora_layers import inject_lora
from .loggingx import CSVLogger, RunLogger, SwanLabLogger
from .models_hf import build_model
from .trainer import train_one
from .utils import infer_device, set_seed


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _append_trials_csv(hpo_dir: Path, row: Dict[str, Any]) -> None:
    import csv

    hpo_dir.mkdir(parents=True, exist_ok=True)
    csv_path = hpo_dir / "trials.csv"
    header = [
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
    need_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def _prepare_run_dir_from_cfg(cfg: Dict[str, Any], run_name: str) -> Tuple[Path, bool]:
    """
    Priority:
      1) io.run_dir (absolute OR relative to cwd) -> deterministic
      2) io.root + run_name via artifacts.prepare_run_dir -> auto naming

    Returns:
      (run_dir, resumed)
    """
    io = cfg["io"]  # strict
    overwrite = str(io["overwrite"])  # strict

    run_dir_cfg = io.get("run_dir", None)
    if run_dir_cfg:
        p = Path(str(run_dir_cfg))
        if not p.is_absolute():
            p = Path(os.getcwd()) / p
        p.parent.mkdir(parents=True, exist_ok=True)

        if p.exists():
            if overwrite == "resume":
                return p, True
            if overwrite == "force":
                shutil.rmtree(p)
                p.mkdir(parents=True, exist_ok=True)
                return p, False

            ans = input(f"[artifacts] {p} exists. Overwrite? [y/N] ").strip().lower()
            if ans in {"y", "yes"}:
                shutil.rmtree(p)
                p.mkdir(parents=True, exist_ok=True)
                return p, False
            return p, True

        p.mkdir(parents=True, exist_ok=True)
        return p, False

    root = str(io["root"])  # strict
    return prepare_run_dir(root=root, run_name=run_name, overwrite=overwrite)


def _persist_repro_bundle(run_dir: Path, cfg: Dict[str, Any]) -> None:
    """
    Best-effort: write provenance + copy base/schedule configs if present.
    """
    try:
        prov = cfg.get("_provenance", {})
        if not isinstance(prov, dict):
            prov = {}

        _write_json(run_dir / "config_provenance.json", prov)

        used_dir = run_dir / "configs_used"
        used_dir.mkdir(parents=True, exist_ok=True)

        base_path = prov.get("base_config_path", None)
        schedule_path = prov.get("schedule_path", None)

        if base_path:
            p = Path(str(base_path))
            if p.exists() and p.is_file():
                shutil.copy2(p, used_dir / p.name)

        if schedule_path:
            p = Path(str(schedule_path))
            if p.exists() and p.is_file():
                shutil.copy2(p, used_dir / p.name)
    except Exception:
        pass


def _resolve_method_lora(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    STRICT method selector:
      cfg.method.name ∈ {baseline_r, baseline_R, ours}
      lora truth lives in cfg.method.<name>.lora
    """
    method = cfg["method"]
    mode = str(method["name"]).strip()
    if mode not in {"baseline_r", "baseline_R", "ours"}:
        raise RuntimeError(f"[runner] invalid method.name={mode!r} (must be baseline_r/baseline_R/ours)")

    sub = method[mode]
    lora = sub["lora"]
    # lora keys are validated by config.py; here we just normalize
    out = {
        "mode": mode,
        "r": int(lora["r"]),
        "alpha": float(lora["alpha"]),
        "dropout": float(lora["dropout"]),
    }
    if mode == "ours":
        out["R"] = int(lora["R"])
    else:
        out["R"] = int(lora["r"])  # baseline uses single-rank injection; keep signature stable
    return out


def run_train(
    cfg: Dict[str, Any],
    run_id: Optional[str] = None,
    trial_id: Optional[str] = None,
    trial_tag: Optional[str] = None,
) -> Dict[str, Any]:
    device = infer_device()

    # strict: no fallback
    seed = int(cfg["train"]["seed"])
    set_seed(seed)

    dataset = str(cfg["task"]["name"])
    assert dataset.startswith("glue/"), "Only glue/<task> supported in this scaffold"
    glue_task = dataset.split("/", 1)[1]

    model_name = str(cfg["model"]["name"])
    max_len = int(cfg["task"]["max_len"])

    data = load_glue(glue_task, model_name, max_len)

    train_loader = DataLoader(
        data.train,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        collate_fn=data.collator,
    )
    val_loader = DataLoader(
        data.validation,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=data.collator,
    )

    model = build_model(model_name, num_labels=data.num_labels)

    # --------------------------
    # STRICT method resolution
    # --------------------------
    ml = _resolve_method_lora(cfg)
    mode = ml["mode"]

    # inject_lora mode mapping:
    # - ours => "ours"
    # - baselines => "baseline"
    inject_mode = "ours" if mode == "ours" else "baseline"

    # IMPORTANT: scaling should NOT be touched here.
    inject_lora(
        model,
        mode=inject_mode,
        r=int(ml["r"]),
        R=int(ml["R"]),
        alpha=float(ml["alpha"]),
        dropout=float(ml["dropout"]),
        target_substrings=None,
    )

    model.to(device)

    extra = {"seed": seed, "method": mode}
    if trial_tag:
        extra["trial_tag"] = trial_tag
    if trial_id:
        extra["trial_id"] = trial_id
    if run_id:
        extra["run_id"] = run_id

    run_name = make_run_name(cfg, extra=extra)

    run_dir, _resumed = _prepare_run_dir_from_cfg(cfg, run_name=run_name)
    _persist_repro_bundle(run_dir, cfg)

    metrics_csv = run_dir / "metrics.csv"
    csv_logger = CSVLogger(metrics_csv)

    sw_cfg = cfg["log"]["swanlab"]
    sw_enabled = bool(sw_cfg["enabled"])
    sw_project = str(sw_cfg["project"])
    swan = SwanLabLogger(
        enabled=sw_enabled,
        project=sw_project,
        run_name=run_name,
        config=cfg,
    )

    logger = RunLogger(csv=csv_logger, swan=swan)

    _write_json(run_dir / "config_resolved.json", cfg)

    summary = train_one(cfg, model, train_loader, val_loader, logger)

    _write_json(run_dir / "summary.json", summary)
    logger.close()

    # --------------------------
    # HPO integration (append only if training finished)
    # --------------------------
    hpo_dir_env = os.environ.get("GSA_HPO_RUN_DIR", "")
    stage_env = os.environ.get("GSA_STAGE", "")
    if hpo_dir_env:
        hpo_dir = Path(hpo_dir_env)

        w_max = float(os.environ["GSA_W_MAX"])
        w_final = float(os.environ["GSA_W_FINAL"])
        w_avg = float(os.environ["GSA_W_AVG"])

        val_max = float(summary["val_max"])
        val_final = float(summary["val_final"])
        val_avg = float(summary["val_avg"])

        score = w_max * val_max + w_final * val_final + w_avg * val_avg

        row = {
            "trial_id": trial_id or "",
            "trial_tag": trial_tag or run_name,
            "stage": stage_env or (trial_tag or ""),
            "score": score,
            "val_max": val_max,
            "val_final": val_final,
            "val_avg": val_avg,
            "trial_cfg_json": json.dumps(cfg, sort_keys=True),
            "seeds": str(seed),
            "run_dir": str(run_dir),
            "metrics_csv": str(metrics_csv),
        }
        _append_trials_csv(hpo_dir, row)

    return summary

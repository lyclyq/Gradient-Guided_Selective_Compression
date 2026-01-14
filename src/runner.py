# src/runner.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader

from .artifacts import make_run_name, prepare_run_dir, dump_json as dump_json_art
from .data_glue import load_glue
from .lora_layers import inject_lora, lora_block_names
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
    ]
    need_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def run_train(
    cfg: Dict[str, Any],
    run_id: Optional[str] = None,
    trial_id: Optional[str] = None,
    trial_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single training run (one seed in current scaffold).

    - Creates a run_dir under io.root
    - Writes: config_resolved.json, metrics.csv, summary.json
    - If env GSA_HPO_RUN_DIR is set, appends a row to <HPO_DIR>/trials.csv for HPO selection.
    """
    device = infer_device()

    # seed (single-seed scaffold for now)
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    # data
    dataset = str(cfg["task"]["name"])
    assert dataset.startswith("glue/"), "Only glue/<task> supported in this scaffold"
    glue_task = dataset.split("/", 1)[1]
    data = load_glue(glue_task, cfg["model"]["name"], int(cfg["task"]["max_len"]))

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

    # model + lora injection
    model = build_model(cfg["model"]["name"], num_labels=data.num_labels)

    mode = str(cfg["method"]["name"])
    lora_cfg = cfg["method"]["lora"]
    replaced = inject_lora(
        model,
        mode=mode,
        r=int(lora_cfg["r"]),
        R=int(lora_cfg["R"]),
        alpha=float(lora_cfg["alpha"]),
        dropout=float(lora_cfg.get("dropout", 0.0)),
        target_substrings=None,  # you can restrict later
    )
    _ = replaced  # keep for debug if needed

    model.to(device)

    # run dir
    io = cfg.get("io", {})
    root = io.get("root", "runs")
    overwrite = io.get("overwrite", "ask")

    extra = {"seed": seed}
    if trial_tag:
        extra["trial_tag"] = trial_tag
    if trial_id:
        extra["trial_id"] = trial_id
    if run_id:
        extra["run_id"] = run_id

    run_name = make_run_name(cfg, extra=extra)
    run_dir, _resumed = prepare_run_dir(root=root, run_name=run_name, overwrite=overwrite)

    # logger
    metrics_csv = run_dir / "metrics.csv"
    csv_logger = CSVLogger(metrics_csv)
    sw_cfg = cfg.get("log", {}).get("swanlab", {}) or {}
    sw_enabled = bool(sw_cfg.get("enabled", False))
    sw_project = str(sw_cfg.get("project", "ShakeAlign_LoRA"))
    swan = SwanLabLogger(
        enabled=sw_enabled,
        project=sw_project,
        run_name=run_name,
        config=cfg,
    )
    logger = RunLogger(csv=csv_logger, swan=swan)

    # persist resolved config
    _write_json(run_dir / "config_resolved.json", cfg)

    # train
    summary = train_one(cfg, model, train_loader, val_loader, logger)

    # save summary
    _write_json(run_dir / "summary.json", summary)
    logger.close()

    # If launched by HPO: append to <HPO_DIR>/trials.csv
    hpo_dir_env = os.environ.get("GSA_HPO_RUN_DIR", "")
    stage_env = os.environ.get("GSA_STAGE", "")
    if hpo_dir_env:
        hpo_dir = Path(hpo_dir_env)

        # score weights: from env if present, otherwise default
        w_max = float(os.environ.get("GSA_W_MAX", "0.5"))
        w_final = float(os.environ.get("GSA_W_FINAL", "0.4"))
        w_avg = float(os.environ.get("GSA_W_AVG", "0.1"))

        val_max = float(summary.get("val_max", summary.get("best_val_acc", -1.0)))
        val_final = float(summary.get("val_final", val_max))
        val_avg = float(summary.get("val_avg", val_max))
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
        }
        _append_trials_csv(hpo_dir, row)

    return summary

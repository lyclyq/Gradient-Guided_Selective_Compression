#!/usr/bin/env python3
# /home/lyclyq/Optimization/grad-shake-align/scripts/run.py
from __future__ import annotations

import sys
from pathlib import Path

# --- ensure project root is on sys.path (so `import src` always works) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from src.cli import parse_args
    from src.config import load_config_with_cli_overrides, validate_config
    from src.runner import run_train
    from src.hpo import run_hpo

    args = parse_args()

    cfg = load_config_with_cli_overrides(
        config_path=args.config,
        schedule_path=getattr(args, "schedule", None),
        set_args=getattr(args, "set", None),
    )

    cmd = args.command

    # ✅ stage is runtime context (not a hyperparam fallback)
    # schedule may override it; we enforce command truth here.
    cfg["stage"] = cmd

    # ✅ strict validation: missing keys => crash NOW
    validate_config(cfg, cmd=cmd)

    if cmd == "train":
        run_train(cfg, run_id=args.run_id, trial_id=args.trial_id, trial_tag=args.trial_tag)
        return 0

    if cmd == "resume":
        cfg["io"]["overwrite"] = "resume"
        run_train(cfg, run_id=args.run_id, trial_id=args.trial_id, trial_tag=args.trial_tag)
        return 0

    if cmd == "hpo":
        run_hpo(
            cfg,
            base_config_path=args.config,
            schedule_path=args.schedule,
            set_args=getattr(args, "set", None) or [],
        )
        return 0

    if cmd == "plot":
        from src.plotting import run_plotting
        runs_dir = args.runs_dir if args.runs_dir is not None else cfg["io"]["root"]
        run_plotting(runs_dir)
        return 0

    if cmd == "final":
        from src.final import run_final
        run_final(
            cfg,
            base_config_path=args.config,
            schedule_path=args.schedule,
            set_args=getattr(args, "set", None) or [],
            best_path=args.best,
        )
        return 0

    raise ValueError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

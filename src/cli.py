# /home/lyclyq/Optimization/grad-shake-align/src/cli.py
from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Gradient Shake-to-Align Experiment Runner")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)

    # ✅ strict: missing => argparse error
    common.add_argument("--config", type=str, required=True)
    common.add_argument("--schedule", type=str, default=None, help="Optional schedule yaml to merge on top of base")

    # ✅ --set may appear multiple times; empty list if none
    common.add_argument("--set", action="append", default=[], help="Override keys: --set train.lr=3e-5")

    # identifiers (optional)
    common.add_argument("--run_id", type=str, default=None)
    common.add_argument("--trial_id", type=str, default=None)
    common.add_argument("--trial_tag", type=str, default=None)

    sub.add_parser("train", parents=[common], help="Single run")
    sub.add_parser("hpo", parents=[common], help="Hyperparameter search")
    sub.add_parser("resume", parents=[common], help="Resume a run (use --set io.run_dir=... and overwrite=resume)")

    plot = sub.add_parser("plot", parents=[common], help="Aggregate and plot runs")
    plot.add_argument("--runs_dir", type=str, default=None)

    final = sub.add_parser("final", parents=[common], help="Run final eval using best_hparams.json")
    final.add_argument("--best", type=str, required=True, help="Path to best_hparams.json (required)")

    return p.parse_args()

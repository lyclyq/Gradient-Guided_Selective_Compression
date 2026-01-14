# src/cli.py
from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Gradient Shake-to-Align Experiment Runner")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, default="configs/base.yaml")
    common.add_argument("--schedule", type=str, default=None, help="Optional schedule yaml to merge on top of base")
    common.add_argument("--set", action="append", default=None, help="Override keys: --set train.lr=3e-5")

    common.add_argument("--run_id", type=str, default=None)
    common.add_argument("--trial_id", type=str, default=None)
    common.add_argument("--trial_tag", type=str, default=None)

    sub.add_parser("train", parents=[common], help="Single run")
    sub.add_parser("hpo", parents=[common], help="Hyperparameter search")
    sub.add_parser("resume", parents=[common], help="Resume a run")

    plot = sub.add_parser("plot", parents=[common], help="Aggregate and plot runs")
    plot.add_argument("--runs_dir", type=str, default=None)

    final = sub.add_parser("final", parents=[common], help="Run final eval using best_hparams.json")
    final.add_argument("--best", type=str, default=None, help="Path to best_hparams.json. If omitted, auto-pick latest under io.root.")

    return p.parse_args()

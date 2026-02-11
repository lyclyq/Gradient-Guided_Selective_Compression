#/home/lyclyq/Optimization/grad-shake-align/scripts/plot_final_4lines_abs.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_epoch_curve(metrics_csv: Path, ykey: str):
    df = pd.read_csv(metrics_csv)

    if "epoch" not in df.columns:
        return None, f"[MISS] no 'epoch' col in {metrics_csv}"

    # robust epoch parse: numeric coercion
    ep = pd.to_numeric(df["epoch"], errors="coerce")
    df = df[ep.notna()].copy()
    df["epoch"] = ep[ep.notna()].astype(int)

    if df.empty:
        return None, f"[MISS] no epoch-end rows in {metrics_csv}"

    if ykey not in df.columns:
        tell = f"[MISS] no '{ykey}' in {metrics_csv} (cols={list(df.columns)[:12]}...)"
        return None, tell

    y = pd.to_numeric(df[ykey], errors="coerce")
    df = df[y.notna()].copy()
    if df.empty:
        return None, f"[MISS] epoch rows exist but '{ykey}' all NaN in {metrics_csv}"

    # sort by epoch and drop duplicates (keep last)
    df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last")
    epochs = df["epoch"].to_numpy(dtype=int)
    ys = pd.to_numeric(df[ykey], errors="coerce").to_numpy(dtype=float)
    return (epochs, ys), f"[OK] {metrics_csv} points={len(epochs)}"

def collect_variant_curves(trial_runs_dir: Path, variant: str, ykey: str, verbose=True):
    vdir = trial_runs_dir / variant
    files = sorted(vdir.glob("s*/metrics.csv"))
    if verbose:
        print(f"\n== {variant} | looking for {ykey} ==")
        print(f"dir: {vdir}")
        print(f"found {len(files)} files")

    curves = []
    for f in files:
        out, msg = read_epoch_curve(f, ykey=ykey)
        if verbose:
            print(msg)
        if out is not None:
            curves.append(out)
    return curves

def align_and_stack(curves):
    if not curves:
        return None, None
    common = None
    for ep, _ in curves:
        s = set(map(int, ep.tolist()))
        common = s if common is None else (common & s)
    if not common:
        return None, None
    epochs = np.array(sorted(common), dtype=int)
    Ys = []
    for ep, y in curves:
        mp = {int(e): float(v) for e, v in zip(ep, y)}
        Ys.append([mp[int(e)] for e in epochs])
    return epochs, np.array(Ys, dtype=float)

def mean_std_plot(epochs, Ys, label):
    mu = Ys.mean(axis=0)
    sd = Ys.std(axis=0)
    plt.plot(epochs, mu, marker="o", label=label)
    plt.fill_between(epochs, mu - sd, mu + sd, alpha=0.18)

def plot_4lines(trial_runs_dir: str, metric: str = "val/acc"):
    tr = Path(trial_runs_dir).expanduser().resolve()
    assert tr.exists(), f"trial_runs_dir not found: {tr}"

    print(f"[INFO] trial_runs_dir = {tr}")

    plt.figure()

    # 1) baseline_r
    curves = collect_variant_curves(tr, "baseline_r", metric, verbose=True)
    e, Ys = align_and_stack(curves)
    if e is not None:
        mean_std_plot(e, Ys, f"baseline_r ({metric}) encouraging")
    else:
        print("[WARN] baseline_r: nothing to plot (alignment failed or no curves)")

    # 2) baseline_R
    curves = collect_variant_curves(tr, "baseline_R", metric, verbose=True)
    e, Ys = align_and_stack(curves)
    if e is not None:
        mean_std_plot(e, Ys, f"baseline_R ({metric})")
    else:
        print("[WARN] baseline_R: nothing to plot (alignment failed or no curves)")

    # 3) ours full
    curves = collect_variant_curves(tr, "ours", metric, verbose=True)
    e, Ys = align_and_stack(curves)
    if e is not None:
        mean_std_plot(e, Ys, f"ours full-R ({metric})")
    else:
        print("[WARN] ours(full): nothing to plot (alignment failed or no curves)")

    # 4) ours r-only
    r_only_key = metric.replace("/acc", "/acc_r_only")
    curves = collect_variant_curves(tr, "ours", r_only_key, verbose=True)
    e, Ys = align_and_stack(curves)
    if e is not None:
        mean_std_plot(e, Ys, f"ours r-only ({r_only_key})")
    else:
        print(f"[WARN] ours(r-only): nothing to plot (alignment failed or no curves)")

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title("Final comparison (mean±std over seeds)")
    plt.legend()
    plt.tight_layout()

    out_dir = tr / "_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_4lines_{metric.replace('/','_')}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"\n[OK] saved figure -> {out_path}")

if __name__ == "__main__":
    import sys
    TR = "/home/lyclyq/Optimization/grad-shake-align/runs/final__glue_rte__bert-base-uncased__ours__ep7__bs32__lr8.046330436316312e-05__final__20260129-164311__1b79c799/trial_runs"
    metric = "val/acc"
    if len(sys.argv) >= 2:
        TR = sys.argv[1]
    if len(sys.argv) >= 3:
        metric = sys.argv[2]
    plot_4lines(TR, metric)

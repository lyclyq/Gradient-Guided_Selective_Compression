import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ 用 “主 CSV” 来推导三个 stage CSV 路径
MASTER_CSV = "rte_lora_r128_lr0p002_cancel_valloss_logpcsv"

# ---- plotting config ----
X_COL = "avg_conf_cancel_interval"
Y_COL = "delta_val_probe_loss"

# 你的 y 是 Δval loss（正=变差）
TITLE_TMPL = "{stage}: ΔValProbeLoss vs Avg Conflict (Cancel)"
XLABEL_TMPL = "Avg conflict over last {interval}-step interval (Cancel)"
YLABEL = "ΔValProbeLoss (current eval - previous eval)  (positive = worse)"

OUT_TMPL = "scatter_{stage}_delta_valloss_vs_cancel.png"


def scatter_reg(x, y, title, xlabel, ylabel, out_png):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # drop NaN/Inf
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    if len(x) < 5:
        print(f"Not enough points for regression: {out_png} (n={len(x)})")
        return

    # linear regression y = a*x + b
    a, b = np.polyfit(x, y, deg=1)
    y_hat = a * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    xs = np.linspace(x.min(), x.max(), 200)
    ys = a * xs + b

    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    plt.plot(xs, ys)  # regression line (default color)
    plt.title(f"{title}\nfit: y={a:.6g}x+{b:.6g}, R^2={r2:.6g}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Saved:", out_png)
    print(f"[{out_png}] slope={a:.6g}, intercept={b:.6g}, R^2={r2:.6g}, n={len(x)}")


def stage_paths_from_master(master_csv: str):
    base = master_csv
    if base.endswith(".csv"):
        base = base[:-4]
    return {
        "low":  base + "_stage_low.csv",
        "mid":  base + "_stage_mid.csv",
        "high": base + "_stage_high.csv",
    }


def load_eval_points(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"[skip] file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # stage csv 里理应全是 eval，但稳妥一点还是过滤
    if "event" in df.columns:
        df = df[df["event"] == "eval"].copy()

    # 必要列检查
    for col in [X_COL, Y_COL]:
        if col not in df.columns:
            print(f"[skip] missing column {col} in {csv_path}")
            return None

    x = df[X_COL].astype(float)
    y = df[Y_COL].astype(float)

    # interval 步长：尽量从列里读，没有就默认 2
    interval = None
    if "eval_every_steps" in df.columns:
        try:
            interval = int(float(df["eval_every_steps"].dropna().iloc[0]))
        except Exception:
            interval = None
    if interval is None:
        interval = 2

    return x, y, interval


def main():
    stage_paths = stage_paths_from_master(MASTER_CSV)

    for stage, path in stage_paths.items():
        loaded = load_eval_points(path)
        if loaded is None:
            continue

        x, y, interval = loaded

        scatter_reg(
            x=x,
            y=y,
            title=TITLE_TMPL.format(stage=stage.upper()),
            xlabel=XLABEL_TMPL.format(interval=interval),
            ylabel=YLABEL,
            out_png=OUT_TMPL.format(stage=stage),
        )


if __name__ == "__main__":
    main()

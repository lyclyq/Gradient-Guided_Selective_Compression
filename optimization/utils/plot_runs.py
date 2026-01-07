# utils/plot_runs.py
"""
python ./optimization/utils/plot_runs.py --tags vit_baseline_c10 vit_darts_c10 vit_darts_unfreeze_c10 \
  --log_dir ./optimization/logs --out_dir ./optimization/plots
"""

import os
import argparse
from typing import List, Dict, Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 纯保存，不弹窗
import matplotlib.pyplot as plt


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _load_epoch_csv(log_dir: str, tag: str) -> pd.DataFrame:
    path = os.path.join(log_dir, f"{tag}_epoch.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"epoch csv not found: {path}")
    df = pd.read_csv(path)

    # —— 统一字段类型（第一重保险）——
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.lower()
    for c in ("epoch", "loss", "acc", "ece"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _load_quickval_csv(log_dir: str, tag: str) -> Optional[pd.DataFrame]:
    path = os.path.join(log_dir, f"{tag}_quickval.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for c in ("step", "epoch", "loss", "acc", "ece"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _finalize_series(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """标准化：只保留 epoch/metric 两列，数值化 + 排序 + 同 epoch 取最后一条。"""
    dd = df[["epoch", metric]].copy()
    dd["epoch"] = pd.to_numeric(dd["epoch"], errors="coerce")
    dd[metric] = pd.to_numeric(dd[metric], errors="coerce")
    dd = dd.dropna(subset=["epoch", metric]).sort_values("epoch")
    dd = dd.drop_duplicates(subset=["epoch"], keep="last")
    return dd


def plot_epoch_compare(tags: List[str], log_dir: str, out_dir: str, title_suffix: str = ""):
    """
    多方法对比：
    - valid-acc / valid-loss：折线对比（按 epoch）
    - test-acc / test-loss：折线接口画“单点 + marker”（通常只有一次 test）
    """
    _ensure_dir(out_dir)
    valid_curves: Dict[str, pd.DataFrame] = {}
    test_points: Dict[str, Optional[pd.Series]] = {}

    for t in tags:
        df = _load_epoch_csv(log_dir, t)

        # —— 第二重保险：函数内部再按 split 过滤 —— #
        v = df[df["split"] == "valid"].copy()
        v = v.dropna(subset=["epoch"]).sort_values("epoch")
        if not v.empty:
            valid_curves[t] = v
        print(f"[plot][valid] {t} splits used -> {sorted(v['split'].unique().tolist()) if not v.empty else []}")

        te_df = df[df["split"] == "test"].copy()
        te_df = te_df.dropna(subset=["epoch"]).sort_values("epoch")
        if not te_df.empty:
            test_points[t] = te_df.iloc[-1]
            print(f"[plot][test]  {t} -> epoch={test_points[t].get('epoch')}, "
                  f"acc={test_points[t].get('acc')}, loss={test_points[t].get('loss')}")
        else:
            test_points[t] = None
            print(f"[plot][test]  {t} -> missing")

    # ---- valid acc 折线对比 ----
    if valid_curves:
        plt.figure(figsize=(8, 5))
        for t, v in valid_curves.items():
            dd = _finalize_series(v, "acc")
            if dd.empty: 
                continue
            plt.plot(dd["epoch"], dd["acc"], label=t, linewidth=2)
        plt.xlabel("Epoch"); plt.ylabel("Valid Acc")
        plt.title("Valid Accuracy vs Epoch" + title_suffix)
        plt.grid(True, alpha=0.3); plt.legend()
        out_acc = os.path.join(out_dir, "valid_acc_compare.png")
        plt.savefig(out_acc, dpi=180, bbox_inches="tight"); plt.close()
        print("[plot] saved:", out_acc)
    else:
        print("[plot] no valid curves to plot for acc.")

    # ---- valid loss 折线对比 ----
    if valid_curves:
        plt.figure(figsize=(8, 5))
        for t, v in valid_curves.items():
            dd = _finalize_series(v, "loss")
            if dd.empty: 
                continue
            plt.plot(dd["epoch"], dd["loss"], label=t, linewidth=2)
        plt.xlabel("Epoch"); plt.ylabel("Valid Loss")
        plt.title("Valid Loss vs Epoch" + title_suffix)
        plt.grid(True, alpha=0.3); plt.legend()
        out_loss = os.path.join(out_dir, "valid_loss_compare.png")
        plt.savefig(out_loss, dpi=180, bbox_inches="tight"); plt.close()
        print("[plot] saved:", out_loss)
    else:
        print("[plot] no valid curves to plot for loss.")

    # ---- test acc 折线（单点 + marker）----
    has_test_acc = False
    plt.figure(figsize=(8, 5))
    for t, te in test_points.items():
        if te is not None and pd.notna(te.get("acc")) and pd.notna(te.get("epoch")):
            plt.plot([float(te["epoch"])], [float(te["acc"])], label=t, linewidth=2, marker="o")
            has_test_acc = True
    if has_test_acc:
        plt.xlabel("Epoch"); plt.ylabel("Test Acc")
        plt.title("Test Accuracy vs Epoch (Single Point per Run)" + title_suffix)
        plt.grid(True, alpha=0.3); plt.legend()
        out_ta = os.path.join(out_dir, "test_acc_compare.png")
        plt.savefig(out_ta, dpi=180, bbox_inches="tight"); plt.close()
        print("[plot] saved:", out_ta)
    else:
        plt.close()
        print("[plot] no test acc to compare.")

    # ---- test loss 折线（单点 + marker）----
    has_test_loss = False
    plt.figure(figsize=(8, 5))
    for t, te in test_points.items():
        if te is not None and pd.notna(te.get("loss")) and pd.notna(te.get("epoch")):
            plt.plot([float(te["epoch"])], [float(te["loss"])], label=t, linewidth=2, marker="o")
            has_test_loss = True
    if has_test_loss:
        plt.xlabel("Epoch"); plt.ylabel("Test Loss")
        plt.title("Test Loss vs Epoch (Single Point per Run)" + title_suffix)
        plt.grid(True, alpha=0.3); plt.legend()
        out_tl = os.path.join(out_dir, "test_loss_compare.png")
        plt.savefig(out_tl, dpi=180, bbox_inches="tight"); plt.close()
        print("[plot] saved:", out_tl)
    else:
        plt.close()
        print("[plot] no test loss to compare.")


def plot_train_vs_valid(tag: str, log_dir: str, out_dir: str):
    """
    单个 run：
    - train_loss.png
    - valid_loss.png
    - train_acc.png
    - valid_acc.png
    完全分离，不再交错绘制。
    """
    _ensure_dir(out_dir)
    df = _load_epoch_csv(log_dir, tag)

    # —— 第二重保险：再次按 split 过滤 —— #
    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "valid"].copy()
    print(f"[plot][single] {tag} train splits used -> {sorted(tr['split'].unique().tolist()) if not tr.empty else []}")
    print(f"[plot][single] {tag} valid splits used -> {sorted(va['split'].unique().tolist()) if not va.empty else []}")

    def _plot_one(split_df: pd.DataFrame, metric: str, split_name: str):
        if split_df.empty or metric not in split_df.columns:
            print(f"[plot] {tag} {split_name} missing or no {metric}."); return
        # —— 只保留这个 split，防混入 —— #
        dd = split_df.copy()
        dd = _finalize_series(dd, metric)
        if dd.empty:
            print(f"[plot] {tag} {split_name} {metric} empty after cleaning."); return
        plt.figure(figsize=(8, 5))
        plt.plot(dd["epoch"], dd[metric], linewidth=2)
        plt.xlabel("Epoch"); plt.ylabel(metric.upper() if metric == "acc" else metric.capitalize())
        plt.title(f"{tag} {split_name.capitalize()} {metric.upper() if metric=='acc' else metric.capitalize()}")
        plt.grid(True, alpha=0.3)
        out_p = os.path.join(out_dir, f"{tag}_{split_name}_{metric}.png")
        plt.savefig(out_p, dpi=180, bbox_inches="tight"); plt.close()
        print("[plot] saved:", out_p)

    # 四张独立图
    _plot_one(tr, "loss", "train")
    _plot_one(va, "loss", "valid")
    _plot_one(tr, "acc",  "train")
    _plot_one(va, "acc",  "valid")


def plot_quickval(tag: str, log_dir: str, out_dir: str, smooth: int = 1):
    """
    画 VALOR/VALOR-RL 的 quick-valid 曲线（按 step）。
    smooth>1 时用简单滑动平均平滑。
    """
    _ensure_dir(out_dir)
    qv = _load_quickval_csv(log_dir, tag)
    if qv is None or qv.empty:
        print("[plot] quickval missing for tag={}, skip.".format(tag))
        return

    qv = qv.sort_values("step")
    if smooth and smooth > 1:
        qv["acc_s"] = qv["acc"].rolling(window=smooth, min_periods=1).mean()
        qv["loss_s"] = qv["loss"].rolling(window=smooth, min_periods=1).mean()
        acc_col, loss_col = "acc_s", "loss_s"
    else:
        acc_col, loss_col = "acc", "loss"

    # acc
    plt.figure(figsize=(8, 5))
    plt.plot(qv["step"], qv[acc_col], label="{}-quickval-acc".format(tag), linewidth=1.8)
    plt.xlabel("Step"); plt.ylabel("Quick-Valid Acc"); plt.grid(True, alpha=0.3); plt.legend()
    plt.title("{} Quick-Valid Accuracy".format(tag))
    out_acc = os.path.join(out_dir, "{}_quickval_acc.png".format(tag))
    plt.savefig(out_acc, dpi=180, bbox_inches="tight"); plt.close()
    print("[plot] saved: {}".format(out_acc))

    # loss
    plt.figure(figsize=(8, 5))
    plt.plot(qv["step"], qv[loss_col], label="{}-quickval-loss".format(tag), linewidth=1.8)
    plt.xlabel("Step"); plt.ylabel("Quick-Valid Loss"); plt.grid(True, alpha=0.3); plt.legend()
    plt.title("{} Quick-Valid Loss".format(tag))
    out_loss = os.path.join(out_dir, "{}_quickval_loss.png".format(tag))
    plt.savefig(out_loss, dpi=180, bbox_inches="tight"); plt.close()
    print("[plot] saved: {}".format(out_loss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="要对比的 run_tag 列表（与 CSV 前缀一致）")
    ap.add_argument("--log_dir", default="./logs", help="CSV 所在目录")
    ap.add_argument("--out_dir", default="./plots", help="输出图目录")
    ap.add_argument("--smooth", type=int, default=1, help="quick-valid 平滑窗口（>1 开启滑动平均）")
    args = ap.parse_args()

    # 1) 多方法 valid 对比（acc / loss）+ test 单点折线
    plot_epoch_compare(args.tags, args.log_dir, args.out_dir, title_suffix="")

    # 2) 每个 run 的单独图（train 与 valid 各自成图）
    for t in args.tags:
        try:
            plot_train_vs_valid(t, args.log_dir, args.out_dir)
        except FileNotFoundError as e:
            print(str(e))

    # 3) 对 VALOR/VALOR-RL 的 quick-valid（若存在）
    for t in args.tags:
        try:
            plot_quickval(t, args.log_dir, args.out_dir, smooth=args.smooth)
        except Exception as e:
            print("[plot] quickval for {} skipped: {}".format(t, e))


if __name__ == "__main__":
    main()

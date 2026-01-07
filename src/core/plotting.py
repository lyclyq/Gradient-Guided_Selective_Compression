import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_compare(curves, out_png, title):
    # curves: list of (label, csv_path)
    plt.figure()
    for label, path in curves:
        if not os.path.exists(path):
            continue
        df=pd.read_csv(path)
        if len(df)==0: 
            continue
        plt.plot(df["step"], df["val_acc"], label=f"{label}-val")
        plt.plot(df["step"], df["test_acc"], label=f"{label}-test")
    plt.legend()
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("acc")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

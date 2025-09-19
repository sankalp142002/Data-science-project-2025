

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = "results/metrics"       # where CSVs are stored
OUT_DIR  = "figures/error_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS   = ["lstm", "bilstm", "gru", "cnnlstm", "tcn", "tft", "informer", "sgp4_monotonic"]
HORIZONS = [1, 3, 7, 30, 90]       # in days


def plot_ecdf(data, horizon):
    plt.figure(figsize=(7,5))
    for model in MODELS:
        if model not in data: continue
        x = np.sort(data[model])
        y = np.arange(1, len(x)+1) / len(x)
        plt.step(x, y, where="post", label=model)

    plt.xscale("log")
    plt.xlabel(f"RMSE at {horizon} d (geo-mean over core features)")
    plt.ylabel("Empirical CDF")
    plt.title(f"Per-satellite robustness — eCDF @ {horizon} d")
    plt.legend(fontsize=8, loc="lower right", ncol=2)
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(OUT_DIR, f"ecdf_{horizon}d.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[saved] {out_path}")


def plot_residuals(data, horizon):
    plt.figure(figsize=(8,5))
    for model in MODELS:
        if model not in data: continue
        sns.histplot(data[model], bins=40, kde=True, stat="density", element="step", label=model, alpha=0.5)

    plt.xlabel(f"Residuals at {horizon} d")
    plt.ylabel("Density")
    plt.title(f"Residual error distribution — {horizon} d horizon")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(OUT_DIR, f"residuals_{horizon}d.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    for horizon in HORIZONS:
        # Load your per-satellite RMSE/residuals from CSV
        # Expected format: one CSV per horizon with columns = model names
        file_path = os.path.join(DATA_DIR, f"errors_{horizon}d.csv")
        if not os.path.exists(file_path):
            print(f"[skip] Missing {file_path}")
            continue

        df = pd.read_csv(file_path)
        data = {m: df[m].dropna().values for m in MODELS if m in df.columns}

        # Plot ECDF + Residuals
        plot_ecdf(data, horizon)
        plot_residuals(data, horizon)

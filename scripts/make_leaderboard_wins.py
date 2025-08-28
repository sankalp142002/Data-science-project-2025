#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 'best-in-class' leaderboard counts (ties shared) and a clean,
aesthetic horizontal bar chart.

Inputs
------
- <root>/bench_metrics_log.csv  (columns include: model, file, feature, span, h, rmse_log, ...)

Output
------
- <root>/figures/Q3_leaderboards/leaderboard_wins.png
- <root>/figures/Q3_leaderboards/leaderboard_wins.pdf
- <root>/tables/E3_best_in_class_counts.csv

Usage
-----
python -m scripts.make_leaderboard_wins --root results/benchmark_v2 --features core --tolerance 0.0
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------- aesthetics -------------------------
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
})

CORE = [
    "eccentricity",
    "mean_motion",
    "semi_major_axis",
    "inclination_deg",
    "mean_anomaly_deg",
]

def make_out(root: Path, sub: str) -> Path:
    p = root / sub
    p.mkdir(parents=True, exist_ok=True)
    return p

def savefig(fig, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    print(f"🖼️  wrote {out_png}")
    plt.close(fig)

def best_in_class_counts(dlog: pd.DataFrame, use_core: bool, tol: float) -> pd.DataFrame:
    """
    Compute fractional win counts by model across matched cells.
    Ties within |delta rmse_log| <= tol share the win equally.
    Only cells where **all models present** are used (matched cells).
    """
    need = {"model","file","feature","span","h","rmse_log"}
    if not need.issubset(dlog.columns):
        raise RuntimeError(f"bench_metrics_log.csv is missing columns: {sorted(need - set(dlog.columns))}")

    df = dlog.copy()
    df = df[df["h"].notna()]

    if use_core:
        df = df[df["feature"].isin(CORE)]
        if df.empty:
            raise RuntimeError("No rows for the core feature set in bench_metrics_log.csv")

    # Matched-cell filtering: keep (file, feature, span, h) where ALL models appear
    models = sorted(df["model"].unique())
    key = ["file","feature","span","h"]
    ct = (df.groupby(key)["model"].nunique().reset_index(name="n_models"))
    complete_keys = ct[ct["n_models"] == len(models)][key]
    dfm = df.merge(complete_keys, on=key, how="inner")

    # For each cell, find minimal rmse_log and award fractional wins to models within tolerance
    def award(group: pd.DataFrame) -> pd.DataFrame:
        m = group["rmse_log"].min()
        winners = np.isclose(group["rmse_log"].values, m, rtol=0, atol=tol)
        k = winners.sum()
        frac = np.where(winners, 1.0 / k, 0.0)
        return pd.DataFrame({
            "model": group["model"].values,
            "win": frac
        })

    wins = dfm.groupby(key, group_keys=False).apply(award, include_groups=False).reset_index(drop=True)

    counts = (
        wins.groupby("model", as_index=False)["win"].sum()
        .sort_values("win", ascending=False)
    )


    # Add 'total_cells' and 'coverage_note' for context
    total_cells = dfm[key].drop_duplicates().shape[0]
    counts["total_cells"] = total_cells
    counts["share_%"] = 100.0 * counts["win"] / counts["total_cells"]

    return counts

def plot_wins(counts: pd.DataFrame, out: Path, title_suffix: str):
    counts = counts.sort_values("win", ascending=True)  # for barh (bottom = highest)
    fig, ax = plt.subplots(figsize=(8.6, max(3.2, 0.5*len(counts)+1.2)))

    # Bars
    palette = sns.color_palette("crest", n_colors=len(counts))
    ax.barh(counts["model"], counts["win"], color=palette, edgecolor="white", linewidth=0.6)

    # Annotate numeric values: absolute wins and percentage of matched cells
    for i, (w, pct) in enumerate(zip(counts["win"].values, counts["share_%"].values)):
        ax.text(w + 0.02 * counts["win"].max(), i,
                f"{w:,.1f}  ({pct:,.1f}%)",
                va="center", ha="left", fontsize=11)

    ax.set_xlabel("Best-in-class wins (ties share)", labelpad=6)
    ax.set_ylabel("Model")
    ttl = "Best-in-class counts — aggregate" + title_suffix
    ax.set_title(ttl)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    ax.grid(False, axis="y")
    ax.set_xlim(left=0)

    # Tight layout and save
    fig.tight_layout()
    savefig(fig, out / "leaderboard_wins.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/benchmark_v2"),
                    help="Benchmark root containing bench_metrics_log.csv")
    ap.add_argument("--features", choices=["core","all"], default="core",
                    help="Use the five core features or all available features")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="Tie tolerance in log-space for rmse_log (abs difference).")
    args = ap.parse_args()

    root = args.root
    flog = root / "bench_metrics_log.csv"
    if not flog.exists():
        raise FileNotFoundError(f"Cannot find {flog}")

    dlog = pd.read_csv(flog)

    use_core = (args.features == "core")
    counts = best_in_class_counts(dlog, use_core=use_core, tol=args.tolerance)

    # Write table
    TBL = make_out(root, "tables")
    out_csv = TBL / "E3_best_in_class_counts.csv"
    counts.to_csv(out_csv, index=False)
    print(f"📝 wrote {out_csv}")

    # Plot
    FIG = make_out(root, "figures/Q3_leaderboards")
    suffix = " (core features)" if use_core else " (all features)"
    plot_wins(counts, FIG, suffix)

if __name__ == "__main__":
    main()

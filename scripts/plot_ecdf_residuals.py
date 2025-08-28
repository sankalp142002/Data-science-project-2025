#!/usr/bin/env python3
# scripts/plot_ecdf_residuals.py
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CORE = ["eccentricity","mean_motion","semi_major_axis","inclination_deg","mean_anomaly_deg"]

def make_out(root: Path, name: str) -> Path:
    p = root / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def savefig_loud(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"🖼️  wrote {out_path}")
    plt.close(fig)

def clip_pos(a, eps=1e-12):
    a = np.asarray(a, float).copy()
    a[a <= 0] = eps
    return a

def gmean_pos(x):
    x = clip_pos(np.asarray(x, float))
    return np.exp(np.log(x).mean())

def perfile_from_any(dsamp: pd.DataFrame, dlog: pd.DataFrame) -> pd.DataFrame:
    """Return tidy per-file RMSE with columns [model,file,h,feature,RMSE]."""
    if not dsamp.empty and {"err","h","model","file","feature"}.issubset(dsamp.columns):
        pf = (dsamp.assign(err2=lambda d: d.err**2)
                    .groupby(["model","file","h","feature"]).err2.mean()
                    .pipe(lambda s: np.sqrt(s)).reset_index()
                    .rename(columns={"err2":"RMSE"}))
        return pf
    # fallback from dlog
    need = {"model","file","h","feature","rmse_log"}
    if need.issubset(dlog.columns):
        pf = dlog[dlog.feature.isin(CORE)][["model","file","h","feature","rmse_log"]].dropna(subset=["h"]).copy()
        pf["RMSE"] = 10**pf["rmse_log"]
        return pf
    return pd.DataFrame()

def build_ratio_frame(dlog: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Return frame with RMSE ratio to baseline per cell."""
    keys = ["file","feature","h","span"]
    if not {"model","rmse_log"}.issubset(dlog.columns):
        return pd.DataFrame()
    base = (dlog[dlog.model==baseline][keys+["rmse_log"]]
            .rename(columns={"rmse_log":"rmse_log_b"}))
    oth  = dlog[dlog.model!=baseline][keys+["model","rmse_log"]]
    m = oth.merge(base, on=keys, how="inner").copy()
    m["ratio"] = clip_pos(10**(m.rmse_log - m.rmse_log_b))
    return m

def ecdf_plots(perfile: pd.DataFrame, out_dir: Path, horizons=(7,30,90)):
    """eCDF of per-satellite geo-mean RMSE at selected horizons."""
    if perfile.empty:
        print("ℹ️  ECDF skipped: per-file frame is empty.")
        return
    for H in horizons:
        dH = perfile[(perfile.h==H) & (perfile.feature.isin(CORE))]
        if dH.empty: 
            print(f"ℹ️  ECDF {H}d skipped: no rows.")
            continue

        # geo-mean across CORE per (model,file)
        tidy = (dH.groupby(["model","file"])["RMSE"].apply(gmean_pos)
                   .reset_index(name="RMSE"))
        if tidy.empty: 
            print(f"ℹ️  ECDF {H}d skipped: tidy empty.")
            continue

        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for m, g in tidy.groupby("model"):
            x = np.sort(clip_pos(g["RMSE"].values))
            if x.size == 0: 
                continue
            y = np.linspace(0, 1, x.size, endpoint=False)
            ax.step(x, y, where="post", label=m, lw=1.6)
        ax.set_xscale("log")
        ax.set_xlabel(f"RMSE at {H} d (geo-mean over core features)")
        ax.set_ylabel("Empirical CDF")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        # legend in single row above to avoid overlapping data
        leg = ax.legend(ncol=3, fontsize=9, frameon=False, loc="lower right")
        for lh in leg.legend_handles:
            lh.set_linewidth(2.0)
        ax.set_title(f"Per-satellite robustness — eCDF @ {H} d")
        savefig_loud(fig, out_dir / f"ecdf_{H}d.png")

def residual_distribution_plots(ratio_df: pd.DataFrame,
                                out_dir: Path,
                                baseline: str,
                                horizons=(7,30,90)):
    """Horizontal boxplots of RMSE ratios by model, per feature and horizon."""
    if ratio_df.empty:
        print("ℹ️  Residual distributions skipped: ratio frame is empty.")
        return

    # Clean up model order by median ratio across all
    model_order = (ratio_df.groupby("model").ratio.median()
                   .sort_values().index.tolist())

    for feat in CORE:
        subf = ratio_df[ratio_df.feature==feat]
        if subf.empty: 
            print(f"ℹ️  Residuals for {feat} skipped: no rows.")
            continue

        # Make a compact grid: one row, len(horizons) columns
        ncols = len(horizons)
        fig, axes = plt.subplots(
            1, ncols, figsize=(4.0*ncols, 5.2), sharex=True, sharey=True
        )
        if ncols == 1:
            axes = [axes]

        x_all = []
        for j, H in enumerate(horizons):
            ax = axes[j]
            dH = subf[subf.h==H]
            if dH.empty:
                ax.text(0.5, 0.5, f"No data @ {H}d", ha="center", va="center")
                ax.axis("off")
                continue

            # Horizontal boxplot, log x-axis with a reference line at ratio=1
            # Use clip to avoid zero/negatives
            dH = dH.copy()
            dH["ratio"] = clip_pos(dH["ratio"].values)

            sns.boxplot(
                data=dH, x="ratio", y="model",
                order=model_order,
                ax=ax, orient="h",
                whis=(5,95), fliersize=1.5, linewidth=1.0
            )
            ax.axvline(1.0, ls="--", lw=1.2, c="k", alpha=0.7)
            ax.set_xscale("log")
            ax.set_xlabel(f"RMSE ratio vs {baseline} @ {H} d")
            if j == 0:
                ax.set_ylabel("Model")
            else:
                ax.set_ylabel("")
            ax.grid(True, which="both", axis="x", alpha=0.25, linewidth=0.6)

            # Annotate group medians at the right edge to add numbers compactly
            med = (dH.groupby("model")["ratio"].median()
                     .reindex(model_order))
            y_pos = np.arange(len(model_order))
            for yk, mv in zip(y_pos, med.values):
                if np.isfinite(mv):
                    ax.text(mv, yk, f"  {mv:,.2f}",
                            va="center", ha="left", fontsize=8,
                            color="tab:gray")

            x_all.append(dH["ratio"].values)

        # Harmonise x-limits across columns to avoid visual bias
        if x_all:
            vals = np.concatenate(x_all)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                lo, hi = np.quantile(vals, [0.01, 0.99])
                lo = max(lo/1.5, 1e-3)
                hi = min(hi*1.5, 1e3)
                for ax in axes:
                    ax.set_xlim(lo, hi)

        fig.suptitle(f"Residual distributions — {feat} (ratio to {baseline}, log scale)",
                     y=0.98, fontsize=13)
        fig.tight_layout()
        savefig_loud(fig, out_dir / f"residual_dist_{feat}.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/benchmark_v2", type=Path,
                    help="Benchmark root containing bench_*.csv")
    ap.add_argument("--baseline", default="sgp4_monotonic",
                    help="Baseline model name in the CSVs")
    ap.add_argument("--horizons", nargs="*", type=int, default=[7,30,90],
                    help="Horizons to plot for ECDF and residuals")
    args = ap.parse_args()

    root: Path = args.root
    dlog  = pd.read_csv(root / "bench_metrics_log.csv")
    dsamp = (pd.read_csv(root / "bench_samples.csv")
             if (root / "bench_samples.csv").exists() else pd.DataFrame())

    # Output directories (match existing layout)
    FIG = make_out(root, "figures")
    F_RB = make_out(FIG, "ROBUSTNESS")
    F_Q2 = make_out(FIG, "Q2_horizon")

    # Build sources
    perfile = perfile_from_any(dsamp, dlog)
    ratio_df = build_ratio_frame(dlog, baseline=args.baseline)

    # Plots
    ecdf_plots(perfile, F_RB, horizons=tuple(args.horizons))
    residual_distribution_plots(ratio_df, F_Q2, baseline=args.baseline,
                                horizons=tuple(args.horizons))

    print("✅  ECDF and residual distribution figures generated.")

if __name__ == "__main__":
    main()

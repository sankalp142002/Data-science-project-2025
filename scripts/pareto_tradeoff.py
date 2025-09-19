

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_family(name: str) -> str:
    """Map raw model names to a family key."""
    s = str(name).strip().lower()
    if "sgp4" in s:
        return "sgp4"
    if "bilstm" in s or ("bi" in s and "lstm" in s):
        return "bilstm"
    if "cnnlstm" in s or ("cnn" in s and "lstm" in s):
        return "cnnlstm"
    if "informer" in s:
        return "informer"
    if "tft" in s or "temporal fusion" in s:
        return "tft"
    if "tcn" in s:
        return "tcn"
    if "gru" in s:
        return "gru"
    if "lstm" in s:
        return "lstm"
    # fallback: first alpha chunk
    m = re.findall(r"[a-z]+", s)
    return m[0] if m else s


def standardize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns: horizon, feature, model; parse ratios/RMSE fields if present."""
    cmap = {c.lower(): c for c in df.columns}

    # Horizon
    if "horizon" in cmap:
        df["horizon"] = pd.to_numeric(df[cmap["horizon"]], errors="coerce").astype("Int64")
    elif "horizon_days" in cmap:
        df["horizon"] = pd.to_numeric(df[cmap["horizon_days"]], errors="coerce").astype("Int64")
    elif "h" in cmap:
        df["horizon"] = pd.to_numeric(df[cmap["h"]], errors="coerce").astype("Int64")
    else:
        raise ValueError("Missing horizon column (expected one of: horizon, horizon_days, h).")

    # Feature
    if "feature" not in cmap:
        raise ValueError("Missing 'feature' column.")
    df["feature"] = df[cmap["feature"]].astype(str).str.strip().str.lower()

    # Model
    if "model" not in cmap:
        raise ValueError("Missing 'model' column.")
    df["model"] = df[cmap["model"]].astype(str)
    df["model_family"] = df["model"].apply(to_family)

    # Metrics (ratio preferred)
    if "median_ratio" in cmap:
        df["median_ratio"] = pd.to_numeric(df[cmap["median_ratio"]], errors="coerce")
    if "rmse" in cmap:
        df["RMSE"] = pd.to_numeric(df[cmap["rmse"]], errors="coerce")

    if "median_ratio" not in df.columns and "RMSE" not in df.columns:
        raise ValueError("Need either 'median_ratio' or 'RMSE' in metrics CSV.")

    return df


def build_y_metric(df: pd.DataFrame):

    # Case 1: direct ratio is present
    if "median_ratio" in df.columns and df["median_ratio"].notna().any():
        out = df.copy()
        out["metric_y"] = out["median_ratio"]
        return out, "Median RMSE ratio vs SGP4 (↓ better)"

    # Case 2: compute ratio from RMSE with fallback to raw RMSE row-wise
    if "RMSE" not in df.columns:
        raise ValueError("No 'RMSE' column available to compute ratios.")

    # Build per-(feature,horizon) SGP4 baseline
    is_sgp4 = df["model_family"].eq("sgp4")
    if is_sgp4.any():
        base = (
            df[is_sgp4]
            .groupby(["feature", "horizon"], as_index=False)["RMSE"]
            .median()
            .rename(columns={"RMSE": "RMSE_sgp4"})
        )
        merged = pd.merge(df, base, on=["feature", "horizon"], how="left")
        # If RMSE_sgp4 is missing for a row, fall back to raw RMSE for that row
        merged["metric_y"] = np.where(
            merged["RMSE_sgp4"].notna(),
            merged["RMSE"] / merged["RMSE_sgp4"],
            merged["RMSE"],
        )
        ylabel = "RMSE ratio vs SGP4 (↓) / RMSE if baseline missing"
        return merged, ylabel

    # No SGP4 anywhere -> pure RMSE
    out = df.copy()
    out["metric_y"] = out["RMSE"]
    return out, "RMSE (↓ better)"


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Return a boolean mask for non-dominated points (minimize both axes)."""
    n = points.shape[0]
    is_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_front[i]:
            continue
        dom = (
            (points[:, 0] <= points[i, 0]) & (points[:, 1] <= points[i, 1]) &
            ((points[:, 0] < points[i, 0]) | (points[:, 1] < points[i, 1]))
        )
        if np.any(dom & (np.arange(n) != i)):
            is_front[i] = False
    return is_front


def enforce_single_point_per_family(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the best (lowest y) entry per model_family for cleaner labeling."""
    return (
        df.sort_values("metric_y", ascending=True)
          .groupby("model_family", as_index=False)
          .first()
    )



def plot_one_feature(merged: pd.DataFrame, horizon: int, feature: str,
                     cost_metric: str, ylabel: str, outdir: str):
    sub = merged[
        (merged["horizon"].astype("Int64") == int(horizon)) &
        (merged["feature"] == feature)
    ].copy()

    # Y needed; allow missing cost -> treat as zero
    sub = sub[sub["metric_y"].notna()].copy()
    if cost_metric not in sub.columns:
        raise ValueError(f"Cost metric '{cost_metric}' not found after merge.")
    sub[cost_metric] = pd.to_numeric(sub[cost_metric], errors="coerce").fillna(0.0)

    if sub.empty:
        avail_h = sorted(merged["horizon"].dropna().unique().tolist())
        avail_f = sorted(merged["feature"].dropna().unique().tolist())
        raise ValueError(
            f"Empty panel for feature='{feature}', horizon={horizon}. "
            f"Available horizons: {avail_h}. Available features: {avail_f}."
        )

    sub = enforce_single_point_per_family(sub)

    costs = sub[cost_metric].astype(float).to_numpy()
    ys = sub["metric_y"].astype(float).to_numpy()
    names = sub["model_family"].tolist()
    pts = np.column_stack([costs, ys])

    mask = pareto_front(pts) if len(pts) > 1 else np.array([True]*len(pts))
    frontier = pts[mask]
    if len(frontier) > 1:
        order = np.argsort(frontier[:, 0])
        frontier = frontier[order]

    plt.figure(figsize=(7.5, 6))
    plt.scatter(costs, ys, s=70, alpha=0.85, label="Models")
    if len(frontier) >= 1:
        plt.scatter(frontier[:, 0], frontier[:, 1], s=100, marker="D", label="Pareto frontier")
        if len(frontier) > 1:
            plt.plot(frontier[:, 0], frontier[:, 1], linewidth=2)

    # nudge labels upward if many points are near one another
    for x, y, n in zip(costs, ys, names):
        plt.text(x, y - 0.015 if len(ys) > 6 else y - 0.01, n, fontsize=9, ha="center", va="top")

    xlab = {
        "params_m":  "Parameter count (millions)",
        "mem_mb":    "Parameter memory (MB)",
        "latency_ms":"Inference latency (ms)"
    }.get(cost_metric, cost_metric)

    plt.xlabel(xlab)
    plt.ylabel(ylabel)
    plt.title(f"Pareto trade-off • feature={feature}, horizon={int(horizon)}d")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best")

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"pareto_{feature}_h{int(horizon)}_{cost_metric}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[OK] Saved: {out}")
    return out



def main():
    ap = argparse.ArgumentParser(description="Pareto trade-off plots (merge by model_family; robust fallbacks).")
    ap.add_argument("--metrics", required=True, help="Path to results/benchmark_v2/bench_metrics.csv")
    ap.add_argument("--resources", required=True, help="Path to results/benchmark_v2/resources.csv")
    ap.add_argument("--horizon", required=True, type=int)
    ap.add_argument("--feature", default=None, type=str, help="If omitted, plot all features found in metrics.")
    ap.add_argument("--cost_metric", required=True, choices=["params_m", "mem_mb", "latency_ms"])
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    # 1) Load & standardize metrics
    met = pd.read_csv(args.metrics)
    met = standardize_metrics(met)
    met, ylabel = build_y_metric(met)

    # 2) Load resources and map to families
    res = pd.read_csv(args.resources)
    if "model" not in res.columns:
        raise ValueError("resources.csv must contain a 'model' column.")
    if args.cost_metric not in res.columns:
        raise ValueError(f"resources.csv must contain the chosen cost metric column '{args.cost_metric}'.")
    res["model_family"] = res["model"].apply(to_family)

    # Keep only family + chosen cost metric
    res_fam = (
        res.drop_duplicates(subset=["model_family"])[["model_family", args.cost_metric]]
    )

    # 3) Auto-expand resources across (family, feature, horizon)
    combos = met[["model_family", "feature", "horizon"]].drop_duplicates()
    res_expanded = combos.merge(res_fam, on="model_family", how="left")

    # 4) Merge with metrics
    merged = pd.merge(met, res_expanded, on=["model_family", "feature", "horizon"], how="inner")

    # 5) Select features and plot
    feats = [args.feature.strip().lower()] if args.feature else \
            sorted(merged["feature"].dropna().unique().tolist())
    print(f"[INFO] Auto-detected features: {feats}")

    for f in feats:
        try:
            plot_one_feature(merged, args.horizon, f, args.cost_metric, ylabel, args.outdir)
        except Exception as e:
            print(f"[WARN] Skipped feature '{f}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

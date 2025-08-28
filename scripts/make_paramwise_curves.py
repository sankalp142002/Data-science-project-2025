import argparse
import math
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

FEATURES = ["semi_major_axis", "mean_motion", "eccentricity"]
FEATURE_SHORT = {"semi_major_axis": "a", "mean_motion": "n", "eccentricity": "e"}

MODEL_ORDER = [
    "bilstm", "cnnlstm", "gru", "lstm", "tcn", "tft", "informer", "sgp4_monotonic"
]
MODEL_COLORS = {
    "bilstm": "#1f77b4",
    "cnnlstm": "#ff7f0e",
    "gru": "#2ca02c",
    "lstm": "#9467bd",
    "tcn": "#e377c2",
    "tft": "#7f7f7f",
    "informer": "#d62728",
    "sgp4_monotonic": "#9e9e9e",
}

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "RMSE" in df.columns and "rmse" not in df.columns:
        df = df.rename(columns={"RMSE": "rmse"})
    df["h"] = pd.to_numeric(df["h"], errors="coerce")
    df["w"] = pd.to_numeric(df["w"], errors="coerce")
    df["span"] = df["span"].fillna("misc")
    df["model"] = df["model"].str.lower()
    df["feature"] = df["feature"].str.strip()
    df = df[df["feature"].isin(FEATURES)]
    return df

def median_rmse_by_cell(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    d = df[df["h"] == horizon]
    if d.empty:
        return pd.DataFrame(columns=["span", "w", "feature", "model", "rmse_med"])
    return (
        d.groupby(["span", "w", "feature", "model"], dropna=False)["rmse"]
        .median()
        .reset_index(name="rmse_med")
    )

def sgp4_baseline_by_feature(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    d = df[(df["h"] == horizon) & (df["model"] == "sgp4_monotonic")]
    return (
        d.groupby(["feature"], dropna=False)["rmse"]
        .median()
        .reset_index()
        .rename(columns={"rmse": "rmse_sgp4"})
    )

def add_ratio_to_sgp4(g: pd.DataFrame, base_feat: pd.DataFrame) -> pd.DataFrame:
    out = g.merge(base_feat, on=["feature"], how="left")
    out["ratio"] = out["rmse_med"] / out["rmse_sgp4"]
    return out

def ensure_order(values, desired):
    seen = [v for v in desired if v in values]
    extra = [v for v in values if v not in desired]
    return seen + extra

def radar_axes(ax, title=None):
    # Base polar config
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Remove degree labels completely
    ax.set_xticks([])               # no angular tick marks or labels
    ax.set_yticklabels([])          # hide radial tick labels
    # Radial limits and grid
    ax.set_rlim(0.4, 2.5)
    ax.set_yticks([0.5, 1.0, 2.0])
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=8, pad=8)

def draw_category_labels(ax, labels=("a","n","e")):
    # Place the 3 labels manually around the circle
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    r = 2.55  # just outside outer ring
    for ang, lab in zip(angles, labels):
        ax.text(ang, r, lab, ha="center", va="center", fontsize=9, weight="medium")

def plot_cell(ax, cell_df: pd.DataFrame):
    # Reference unit circle for SGP4
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(theta, np.ones_like(theta), "--", lw=1.2,
            color=MODEL_COLORS["sgp4_monotonic"], alpha=0.9)

    labels = ["a","n","e"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    for m in MODEL_ORDER:
        if m == "sgp4_monotonic":
            continue
        mdf = cell_df[cell_df["model"] == m]
        if mdf.empty:
            continue
        ratios = []
        for f in ["semi_major_axis","mean_motion","eccentricity"]:
            row = mdf[mdf["feature"] == f]
            if row.empty or not np.isfinite(row["ratio"].values[0]):
                ratios.append(np.nan)
            else:
                r = float(row["ratio"].values[0])
                r = max(0.4, min(2.5, r))      # clip for readability
                ratios.append(r)
        ratios += ratios[:1]
        if all(np.isnan(ratios[:-1])):
            continue
        ax.plot(
            angles, ratios, lw=1.8, marker="o", ms=2.8,
            color=MODEL_COLORS.get(m, None), label=m
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--spans", type=str,
                    default="last90d,last180d,last365d,last730d,full")
    ap.add_argument("--windows", type=str, default="30,60,90")
    ap.add_argument("--dpi", type=int, default=250)
    args = ap.parse_args()

    raw = pd.read_csv(args.input)
    df = clean(raw)

    g = median_rmse_by_cell(df, args.horizon)
    if g.empty:
        raise SystemExit(f"No rows at h={args.horizon}.")
    base = sgp4_baseline_by_feature(df, args.horizon)
    if base.empty:
        raise SystemExit("No SGP4 rows for baseline.")

    gr = add_ratio_to_sgp4(g, base)

    spans_in = list(pd.unique(gr["span"]))
    wins_in  = [w for w in pd.unique(gr["w"]) if pd.notna(w)]

    spans = ensure_order(spans_in, args.spans.split(","))
    windows = ensure_order(wins_in, [float(x) for x in args.windows.split(",")])

    mpl.rcParams.update({
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#666666",
        "grid.color": "#DDDDDD",
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
    })

    fig_w = max(8.0, 3.2 * len(windows))
    fig_h = max(6.0, 2.8 * len(spans))
    fig, axs = plt.subplots(len(spans), len(windows),
                            figsize=(fig_w, fig_h), subplot_kw={"projection": "polar"})
    if len(spans) == 1 and len(windows) == 1:
        axs = np.array([[axs]])

    for r, span in enumerate(spans):
        for c, w in enumerate(windows):
            ax = axs[r, c]
            radar_axes(ax, title=f"span={span}, w={int(w) if not math.isnan(w) else 'misc'}")
            sub = gr[(gr["span"] == span) & (gr["w"] == w)]
            plot_cell(ax, sub)
            draw_category_labels(ax, labels=("a","n","e"))

    # Legend with explicit color → model mapping
    handles = []
    labels = []
    # SGP4 reference handle
    handles.append(plt.Line2D([0],[0], color=MODEL_COLORS["sgp4_monotonic"], lw=1.6, ls="--"))
    labels.append("sgp4_monotonic")
    for m in MODEL_ORDER:
        if m == "sgp4_monotonic":
            continue
        handles.append(plt.Line2D([0],[0], color=MODEL_COLORS[m], lw=2.2, marker="o", ms=4))
        labels.append(m)

    leg = fig.legend(handles, labels, title="Models",
                     ncol=min(5, len(labels)), loc="lower center",
                     frameon=True, fontsize=9, title_fontsize=9)
    leg.get_frame().set_edgecolor("#bbbbbb")
    leg.get_frame().set_linewidth(0.6)

    fig.subplots_adjust(bottom=0.09)
    os.makedirs(args.output, exist_ok=True)
    out = os.path.join(args.output, f"param_radar_grid_h{args.horizon}.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"[done] Saved {out}")

if __name__ == "__main__":
    main()

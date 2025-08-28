#!/usr/bin/env python3
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("results/benchmark_v2")
d = pd.read_csv(ROOT/"bench_metrics_log.csv")

core = d[d["feature"].isin(["eccentricity","mean_motion","semi_major_axis","inclination_deg","mean_anomaly_deg"])]

OUT = ROOT/"figures_min"; OUT.mkdir(parents=True, exist_ok=True)

# Q1 overall lollipop
best = core.groupby("model")["rmse_log"].mean().sort_values()
fig, ax = plt.subplots(figsize=(9,3))
ax.hlines(best.index, 0, 10**best.values)
ax.set_xscale("log"); ax.set_title("Q1 – overall (geo-mean on log scale)")
fig.savefig(OUT/"q1_overall.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# Q2 horizon curves (median per horizon)
for feat in ["eccentricity","mean_motion","semi_major_axis","inclination_deg","mean_anomaly_deg"]:
    sub = core[core.feature==feat]
    if sub.empty: continue
    fig, ax = plt.subplots(figsize=(8,3))
    for mdl, g in sub.groupby("model"):
        med = g.groupby("h")["rmse_log"].median()
        ax.plot(med.index, 10**med.values, marker="o", label=mdl)
    ax.set_yscale("log"); ax.set_title(f"Q2 – median vs horizon ({feat})"); ax.legend()
    fig.savefig(OUT/f"q2_horizon_{feat}.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# Q4 window heatmaps (ratio to best per horizon)
sub = core.dropna(subset=["w"]).copy()
if not sub.empty:
    sub["w"] = sub["w"].astype(int)
    g = (sub.groupby(["model","h","w"])["rmse_log"].median().reset_index())
    for mdl, gm in g.groupby("model"):
        best_w = gm.loc[gm.groupby("h")["rmse_log"].idxmin()][["h","rmse_log"]].rename(columns={"rmse_log":"best"})
        m = gm.merge(best_w, on="h")
        m["ratio"] = 10**(m["rmse_log"] - m["best"])
        piv = m.pivot(index="h", columns="w", values="ratio")
        import seaborn as sns
        fig = sns.heatmap(piv, cmap="viridis_r", annot=True, fmt=".2f", cbar_kws=dict(label="× best")).get_figure()
        fig.suptitle(f"Q4 – window×horizon ratio ({mdl})")
        fig.savefig(OUT/f"q4_window_{mdl}.png", dpi=160, bbox_inches="tight"); plt.close(fig)

print(f"✅ wrote plots to {OUT.resolve()}")

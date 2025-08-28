#!/usr/bin/env python3
# scripts/plot_overall.py
from __future__ import annotations
from pathlib import Path
import argparse, json, warnings, re
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from scipy import stats

sns.set_theme(style="whitegrid", context="talk")

def clip_pos(a, eps=1e-12):
    a = np.asarray(a, float).copy()
    a[a <= 0] = eps
    return a

def gmean_pos(x):
    x = clip_pos(np.asarray(x, float))
    return np.exp(np.log(x).mean())

def make_out(root: Path, name: str) -> Path:
    p = root / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def savefig_loud(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"🖼️  wrote {out_path}")
    plt.close(fig)

def load_profiles(prof_root: Path) -> pd.DataFrame:
    rows = []
    if not prof_root.exists():
        return pd.DataFrame()
    for f in prof_root.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            d["model"] = f.stem
            rows.append(d)
        except Exception:
            pass
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="results/benchmark_v2", type=Path)
ap.add_argument("--baseline", default="sgp4_monotonic")
ap.add_argument("--span-order", nargs="*", default=["last90d","last180d","last365d","last730d","full"])
args = ap.parse_args()

dmet  = pd.read_csv(args.root / "bench_metrics.csv")
dlog  = pd.read_csv(args.root / "bench_metrics_log.csv")
dsamp = (pd.read_csv(args.root / "bench_samples.csv")
         if (args.root / "bench_samples.csv").exists() else pd.DataFrame())

# try to auto-detect baseline
if args.baseline not in set(dmet.model):
    cand = next((m for m in dmet.model.unique() if re.search("sgp4", m, re.I)), None)
    if cand:
        args.baseline = cand

FIG = make_out(args.root, "figures")
F_Q1 = make_out(FIG, "Q1_overall")
F_Q2 = make_out(FIG, "Q2_horizon")
F_RB = make_out(FIG, "ROBUSTNESS")
F_ST = make_out(FIG, "STATS")
F_TS = make_out(FIG, "TIME_RASTERS")
F_RTN= make_out(FIG, "RTN_ENERGY")
F_CST= make_out(FIG, "COST")
F_UNC= make_out(FIG, "UNCERTAINTY")
F_OT = make_out(FIG, "OUTLIERS")
F_OVF= make_out(FIG, "OVERFIT")

CORE = ["eccentricity","mean_motion","semi_major_axis","inclination_deg","mean_anomaly_deg"]

# ---------------- Q1 (rankings) ----------------
core_met = dmet[dmet.feature.isin(CORE)].copy()
if core_met.empty:
    raise RuntimeError("No core features in bench_metrics.csv – nothing to plot.")
best = core_met.groupby("model").RMSE.apply(gmean_pos).sort_values()

fig, ax = plt.subplots(figsize=(8, 4))
xmin = max(best.min() * 0.9, 1e-12)
ax.hlines(best.index, xmin=xmin, xmax=best.values, color="#bcd", lw=4)
ax.plot(best.values, best.index, "o", color="#2369bd")
ax.set_xscale("log")
ax.set_xlabel("Geo-mean RMSE (log)")
ax.set_title("Overall ranking")
savefig_loud(fig, F_Q1 / "overall_lollipop.png")

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(best.index, best.values)
ax.set_xscale("log")
ax.set_xlabel("Geo-mean RMSE (log)")
ax.set_title("Overall ranking (bar)")
savefig_loud(fig, F_Q1 / "overall_bar.png")

# ---------------- Q2 (horizon curves) -----------------------------
def plot_horizon_from_samples(dsamp: pd.DataFrame, span: str | None):
    sub = dsamp[dsamp.feature.isin(CORE)].dropna(subset=["h"]).copy()
    if "span" in sub.columns and span is not None:
        sub = sub[sub.span == span]
        if sub.empty: return False

    perfile = (sub.assign(err2=lambda d: d.err**2)
                  .groupby(["model","file","h","feature"]).err2.mean()
                  .pipe(lambda s: np.sqrt(s)).reset_index()
                  .rename(columns={"err2": "RMSE"}))

    def bootstrap_ci(x, B=800, alpha=0.05):
        if len(x) == 0: return (np.nan, np.nan, np.nan)
        rng = np.random.default_rng(42)
        boots = [gmean_pos(rng.choice(x, size=len(x), replace=True)) for _ in range(B)]
        lo, hi = np.quantile(boots, [alpha/2, 1 - alpha/2])
        return (gmean_pos(x), lo, hi)

    rows = []
    for (m, h), grp in perfile.groupby(["model", "h"]):
        x = grp[grp.feature.isin(CORE)].RMSE.values
        mu, lo, hi = bootstrap_ci(x)
        rows.append({"model": m, "h": h, "rmse_gmean": mu, "lo": lo, "hi": hi})
    agg = pd.DataFrame(rows).sort_values("h")
    if agg.empty: return False

    fig, ax = plt.subplots(figsize=(9, 4))
    for m, grp in agg.groupby("model"):
        ax.plot(grp.h, clip_pos(grp.rmse_gmean), label=m, lw=2 if m == args.baseline else 1.5)
        if grp["lo"].notna().all() and grp["hi"].notna().all():
            ax.fill_between(grp.h, clip_pos(grp.lo), clip_pos(grp.hi), alpha=.15)
    ax.set_yscale("log"); ax.set_xlabel("Horizon (d)"); ax.set_ylabel("Geo-mean RMSE")
    ttl = f"Horizon curves with 95% CIs – {span}" if span else "Horizon curves with 95% CIs"
    ax.set_title(ttl); ax.legend(ncol=2, fontsize=9)
    name = f"horizon_CI_{span}.png" if span else "horizon_CI.png"
    savefig_loud(fig, F_Q2 / name)
    return True

def plot_horizon_from_log(dlog: pd.DataFrame, span: str | None):
    sub = dlog[dlog.feature.isin(CORE)].dropna(subset=["h"]).copy()
    if "span" in sub.columns and span is not None:
        sub = sub[sub.span == span]
        if sub.empty: return False
    agg = (sub.groupby(["model", "h"])["rmse_log"].median().reset_index())
    if agg.empty: return False

    fig, ax = plt.subplots(figsize=(9, 4))
    for m, grp in agg.groupby("model"):
        ax.plot(grp.h, 10**grp.rmse_log, marker="o", label=m, lw=2 if m == args.baseline else 1.5)
    ax.set_yscale("log"); ax.set_xlabel("Horizon (d)"); ax.set_ylabel("Median RMSE")
    ttl = f"Horizon curves (median) – {span}" if span else "Horizon curves (median)"
    ax.set_title(ttl); ax.legend(ncol=2, fontsize=9)
    name = f"horizon_median_{span}.png" if span else "horizon_median.png"
    savefig_loud(fig, F_Q2 / name)
    return True

for span in args.span_order:
    ok = False
    if not dsamp.empty:
        ok = plot_horizon_from_samples(dsamp, span)
    if not ok:
        plot_horizon_from_log(dlog, span)
if dsamp.empty:
    plot_horizon_from_log(dlog, span=None)
else:
    plot_horizon_from_samples(dsamp, span=None)

# ---------- helpers built from any source ----------
def perfile_from_any(dsamp: pd.DataFrame, dlog: pd.DataFrame) -> pd.DataFrame:
    # Prefer samples
    if not dsamp.empty:
        pf = (dsamp.assign(err2=lambda d: d.err**2)
                    .groupby(["model","file","h","feature"]).err2.mean()
                    .pipe(lambda s: np.sqrt(s)).reset_index()
                    .rename(columns={"err2":"RMSE"}))
        return pf
    # Fallback from dlog
    need = {"model","file","h","feature","rmse_log"}
    if not need.issubset(dlog.columns): return pd.DataFrame()
    pf = dlog[dlog.feature.isin(CORE)][["model","file","h","feature","rmse_log"]].dropna(subset=["h"]).copy()
    pf["RMSE"] = 10**pf["rmse_log"]
    return pf

perfile = perfile_from_any(dsamp, dlog)

# ---------- STATS: Wilcoxon ----------
if not perfile.empty:
    pivot_key = ["file","h","feature"]
    common = perfile[pivot_key].drop_duplicates()
    models = sorted(perfile.model.unique())
    P = pd.DataFrame(np.nan, index=models, columns=models)
    for i, m1 in enumerate(models):
        v1 = perfile[perfile.model == m1].merge(common, on=pivot_key)["RMSE"]
        for j, m2 in enumerate(models):
            if j <= i: continue
            v2 = perfile[perfile.model == m2].merge(common, on=pivot_key)["RMSE"]
            k = min(len(v1), len(v2))
            if k < 10: continue
            try:
                p = stats.wilcoxon(v1[:k], v2[:k], zero_method="wilcox", alternative="two-sided").pvalue
                P.loc[m1, m2] = P.loc[m2, m1] = p
            except Exception:
                pass
    if P.notna().any().any():
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(P, annot=True, fmt=".2g", cmap="mako_r", cbar_kws=dict(label="p-value"), ax=ax)
        ax.set_title("Paired Wilcoxon (RMSE per file/h/feature)")
        savefig_loud(fig, F_ST / "pvalues_wilcoxon.png")

# ---------- Q2 residual ratios vs baseline ----------
keys = ["file","feature","h","span"]
base_df = (dlog[dlog.model == args.baseline][keys + ["rmse_log"]]
           .rename(columns={"rmse_log": "rmse_log_b"}))
oth     = dlog[dlog.model != args.baseline][keys + ["model","rmse_log"]]
mrg = oth.merge(base_df, on=keys, how="inner")
mrg["ratio"] = clip_pos(10**(mrg.rmse_log - mrg.rmse_log_b))
for feat in CORE:
    d = mrg[mrg.feature == feat]
    if d.empty: continue
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, g in d.groupby("model"):
        med = g.groupby("h").ratio.median()
        ax.plot(med.index, med.values, marker="o", label=m)
    ax.axhline(1, ls="--", c="k"); ax.set_yscale("log")
    ax.set_title(f"Residual (RMSE / {args.baseline}) – {feat}")
    ax.set_xlabel("Horizon (d)"); ax.set_ylabel("Median ratio")
    ax.legend(ncol=2, fontsize=9)
    savefig_loud(fig, F_Q2 / f"residual_ratio_{feat}.png")

# ---------- ROBUSTNESS: eCDFs (fallbacks work) ----------
if not perfile.empty:
    for H in [7, 30, 90]:
        dH = perfile[perfile.h == H]
        if dH.empty: continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for m, g in dH.groupby("model"):
            # geo-mean over features for each file
            per_file = (g.groupby(["file"])["RMSE"].apply(gmean_pos).sort_values().values)
            if len(per_file) == 0: continue
            y = np.linspace(0, 1, len(per_file), endpoint=False)
            ax.step(clip_pos(per_file), y, where="post", label=m)
        ax.set_xscale("log"); ax.set_xlabel(f"RMSE at {H} d"); ax.set_ylabel("eCDF")
        ax.set_title(f"Per-satellite robustness – eCDF @ {H} d")
        ax.legend()
        savefig_loud(fig, F_RB / f"ecdf_{H}d.png")

# ---------- TIME_RASTERS: fallback to window×horizon from dlog ----------
if not dsamp.empty and "t0" in dsamp.columns and dsamp["t0"].notna().any():
    dt = dsamp.copy()
    dt["t0"] = pd.to_datetime(dt["t0"], errors="coerce")
    for m in dt.model.unique():
        for feat in ["mean_anomaly_deg","semi_major_axis"]:
            dmf = dt[(dt.model == m) & (dmf.feature == feat)]
            if dmf.empty: continue
            rm = (dmf.assign(err2=lambda d: d.err**2)
                     .groupby([pd.Grouper(key="t0", freq="7D"), "h"]).err2.mean()
                     .pipe(lambda s: np.sqrt(s)).reset_index())
            if rm.empty: continue
            piv = rm.pivot_table(index="h", columns="t0", values="err2", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(10, 3.6))
            sns.heatmap(np.sqrt(piv), cmap="magma_r", cbar_kws=dict(label="RMSE"), ax=ax)
            ax.set_title(f"Time×Horizon raster – {m} – {feat}")
            savefig_loud(fig, F_TS / f"raster_{m}_{feat}.png")
else:
    # Fallback: use window index 'w' as time axis from dlog
    need = {"model","feature","h","w","rmse_log"}
    if need.issubset(dlog.columns):
        sub = dlog[dlog.feature.isin(["mean_anomaly_deg","semi_major_axis"])].copy()
        for (m, feat), g in sub.groupby(["model","feature"]):
            piv = (g.pivot_table(index="h", columns="w", values="rmse_log", aggfunc="median"))
            if piv.empty: continue
            fig, ax = plt.subplots(figsize=(10, 3.6))
            sns.heatmap(10**piv, cmap="magma_r", cbar_kws=dict(label="RMSE"), ax=ax)
            ax.set_title(f"Window×Horizon raster – {m} – {feat}")
            savefig_loud(fig, F_TS / f"window_raster_{m}_{feat}.png")
    else:
        print("ℹ️  TIME_RASTERS skipped (no t0 in samples and no w in logs).")

# ---------- COST ----------
prof = load_profiles(Path("results/profiles"))
if not prof.empty:
    acc = core_met.groupby("model").RMSE.apply(gmean_pos).reset_index()
    p = prof.merge(acc, on="model", how="left").dropna(subset=["RMSE"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(p["params"], p["RMSE"], s=80)
    for _, r in p.iterrows():
        ax.annotate(r["model"], (r["params"], r["RMSE"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Parameters"); ax.set_ylabel("Geo-mean RMSE")
    ax.set_title("Accuracy vs Model Size (Pareto)")
    savefig_loud(fig, F_CST / "pareto_params_accuracy.png")

    if "latency_ms" in p.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=p, x="model", y="latency_ms", ax=ax)
        ax.set_ylabel("Latency (ms / batch)"); ax.set_title("Latency by model")
        savefig_loud(fig, F_CST / "latency_bars.png")
else:
    # Fallback: accuracy-only chart lives here so the folder isn't empty
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(best.index, best.values)
    ax.set_xscale("log"); ax.set_xlabel("Geo-mean RMSE (log)")
    ax.set_title("Accuracy (profiles not found)")
    savefig_loud(fig, F_CST / "accuracy_only.png")

# ---------- UNCERTAINTY ----------
if not dsamp.empty and {"y_pred_mu","y_pred_sigma"}.issubset(dsamp.columns):
    z_grid = np.linspace(0.5, 2.5, 9)
    rows = []
    for z in z_grid:
        cover = np.mean(np.abs(dsamp.y_true - dsamp.y_pred_mu) <= z * dsamp.y_pred_sigma)
        rows.append({"nominal": 2 * stats.norm.cdf(z) - 1, "empirical": cover})
    cov = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(cov.nominal, cov.empirical, marker="o")
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage")
    ax.set_title("Reliability diagram (all models pooled)")
    savefig_loud(fig, F_UNC / "reliability.png")

# ---------- OUTLIERS ----------
if not dsamp.empty and set(["alt_km","ecc","tle_age_d","f107"]).issubset(dsamp.columns):
    tri = (dsamp.assign(err2=lambda d: d.err**2)
                 .groupby(["model","file","h"])
                 .agg(rmse=("err2", lambda s: np.sqrt(np.mean(s))),
                      alt_km=("alt_km","mean"),
                      ecc=("ecc","mean"),
                      tle_age_d=("tle_age_d","mean"),
                      f107=("f107","mean")).reset_index())
    for x in ["alt_km","ecc","tle_age_d","f107"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=tri, x=x, y="rmse", hue="model", ax=ax, alpha=.5, legend=False)
        ax.set_yscale("log"); ax.set_title(f"RMSE vs {x}")
        savefig_loud(fig, F_OT / f"scatter_rmse_vs_{x}.png")

# ---------- OVERFIT ----------
hist_root = Path("results/history")
if hist_root.exists():
    frames = []
    for f in hist_root.glob("*.csv"):
        d = pd.read_csv(f); d["model"] = f.stem
        frames.append(d)
    if frames:
        H = pd.concat(frames, ignore_index=True)
        if {"epoch","val_rmse"}.issubset(H.columns):
            fig, ax = plt.subplots(figsize=(7, 4))
            for m, g in H.groupby("model"):
                ax.plot(g.epoch, clip_pos(g.val_rmse), label=m)
            ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Val RMSE")
            ax.set_title("Validation curve – overfitting check")
            ax.legend(fontsize=9, ncol=2)
            savefig_loud(fig, F_OVF / "val_curves.png")

print(f"✅  Figures written to {FIG.resolve()}")

#!/usr/bin/env python3
# scripts/make_benchmark_tables_ext.py
from __future__ import annotations
import argparse, json, math, os, re, sys, subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ------------------------- helpers & constants -------------------------

CORE = ["eccentricity","mean_motion","semi_major_axis","inclination_deg","mean_anomaly_deg"]

def clip_pos(a, eps=1e-12):
    a = np.asarray(a, float).copy()
    a[a <= 0] = eps
    return a

def gmean_pos(x):
    x = np.asarray(x, float)
    x = clip_pos(x)
    return float(np.exp(np.log(x).mean())) if x.size else np.nan

def escape_tex(s: str) -> str:
    if s is None: return ""
    s = str(s)
    # Escape a few common LaTeX specials
    repl = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for k,v in repl.items():
        s = s.replace(k, v)
    return s

def fmt_num(x):
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    # choose compact formatting
    if abs(xf) >= 100 or (abs(xf) > 0 and abs(xf) < 0.01):
        return f"{xf:.3g}"
    return f"{xf:.3f}"

def fmt_pct(x):
    try:
        return f"{float(x):+.1f}\\%"
    except Exception:
        return str(x)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_table(df: pd.DataFrame,
               out_dir: Path,
               name: str,
               *,
               index: bool = False,
               latex_bold_min_cols: list[str] | None = None,
               round_cols: list[str] | None = None,
               caption: str | None = None,
               label: str | None = None,
               percent_cols: list[str] | None = None):
    """
    Save df to CSV and LaTeX.
    - Bold minimum per row across latex_bold_min_cols (if provided).
    - Format % columns with fmt_pct; others with fmt_num.
    """
    ensure_dir(out_dir)
    csv_path = out_dir / f"{name}.csv"
    tex_path = out_dir / f"{name}.tex"

    df_to_write = df.copy()

    # formatting
    percent_cols = percent_cols or []
    round_cols = round_cols or []

    for c in df_to_write.columns:
        if c in percent_cols:
            df_to_write[c] = df_to_write[c].map(fmt_pct)
        elif df_to_write[c].dtype.kind in "fi":
            df_to_write[c] = df_to_write[c].map(fmt_num)
        else:
            df_to_write[c] = df_to_write[c].astype(str)

    # bold minima across selected columns (row-wise)
    if latex_bold_min_cols:
        intersect = [c for c in latex_bold_min_cols if c in df.columns]
        if intersect:
            # Work on original numeric values to find mins
            numblock = df[intersect].apply(pd.to_numeric, errors="coerce")
            mins = numblock.idxmin(axis=1)
            # Apply \textbf{} to those cells in df_to_write
            for r, cmin in mins.items():
                if pd.notna(cmin):
                    val = df_to_write.at[r, cmin]
                    df_to_write.at[r, cmin] = r"\textbf{" + str(val) + "}"

    # CSV
    df.to_csv(csv_path, index=index)
    print(f"📄  wrote {csv_path}")

    # LaTeX (booktabs style)
    cap = caption or name.replace("_", " ")
    lab = label or f"tab:{name}"

    # Escape headers and index labels
    cols_tex = [escape_tex(c) for c in map(str, df_to_write.columns)]
    idx_name = escape_tex(df_to_write.index.name) if df_to_write.index.name else ""

    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\begin{tabular}{%s}\n" % ("l" + "r"*(len(cols_tex)) if index else "r"*len(cols_tex)))
        f.write("\\toprule\n")
        if index:
            header_line = " & ".join([idx_name] + cols_tex)
        else:
            header_line = " & ".join(cols_tex)
        f.write(header_line + " \\\\\n\\midrule\n")

        # write rows
        for ridx, row in df_to_write.iterrows():
            row_vals = [escape_tex(ridx)] if index else []
            row_vals += [str(v) for v in row.values.tolist()]
            f.write(" & ".join(row_vals) + " \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"\\caption{{{escape_tex(cap)}}}\n\\label{{{escape_tex(lab)}}}\n\\end{table}\n")
    print(f"📄  wrote {tex_path}")

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

def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """FDR BH correction; expects 1D series of p-values (NA ignored)."""
    s = pvals.dropna().sort_values()
    m = len(s)
    if m == 0: return pvals
    ranks = np.arange(1, m+1)
    q = s.values * m / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    out = pd.Series(np.nan, index=pvals.index)
    out.loc[s.index] = np.clip(q, 0, 1)
    return out

# ------------------------- table builders -------------------------

def build_accuracy_geo(dmet: pd.DataFrame) -> pd.DataFrame:
    if not set(["model","feature","RMSE"]).issubset(dmet.columns):
        return pd.DataFrame()
    acc = (dmet[dmet.feature.isin(CORE)]
           .groupby("model").RMSE.apply(gmean_pos).rename("RMSE_geo").sort_values())
    return acc.to_frame()

def build_G1_cost_summary(dmet: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    acc = build_accuracy_geo(dmet)
    if acc.empty or profiles.empty: return pd.DataFrame()
    keep = [c for c in ["model","params","latency_ms","training_time_h","infer_samples_s"] if c in profiles.columns]
    p = profiles[keep].drop_duplicates("model")
    out = p.merge(acc, on="model", how="left").sort_values("RMSE_geo")
    return out

def build_C5_sample_sizes(dlog: pd.DataFrame, dsamp: pd.DataFrame) -> pd.DataFrame:
    if not dlog.empty and set(["span","h","feature"]).issubset(dlog.columns):
        g = (dlog.assign(RMSE=lambda d: 10**d["rmse_log"] if "rmse_log" in d else np.nan)
                  .groupby(["span","h","feature"])
                  .agg(n_windows=("w", lambda s: s.nunique() if "w" in dlog.columns else np.nan),
                       n_files=("file", lambda s: s.nunique() if "file" in dlog.columns else np.nan))
                  .reset_index())
        return g
    if not dsamp.empty and set(["h","feature"]).issubset(dsamp.columns):
        cols = ["span","h","feature"] if "span" in dsamp.columns else ["h","feature"]
        g = (dsamp.groupby(cols)
                  .agg(n_windows=("file", "nunique"))
                  .reset_index())
        return g
    return pd.DataFrame()

def build_D2_improvement_vs_baseline(dlog: pd.DataFrame, baseline: str) -> pd.DataFrame:
    if dlog.empty or "rmse_log" not in dlog.columns or "model" not in dlog.columns:
        return pd.DataFrame()
    base = dlog[dlog.model == baseline][["file","feature","h","w","span","rmse_log"]]
    oth  = dlog[dlog.model != baseline]
    mrg = oth.merge(base, on=["file","feature","h","w","span"], suffixes=("","_b"))
    mrg["ratio"] = clip_pos(10**(mrg.rmse_log - mrg.rmse_log_b))
    # aggregate over CORE features
    sub = mrg[mrg.feature.isin(CORE)]
    if sub.empty: return pd.DataFrame()
    g = (sub.groupby(["model","span","h"]).ratio.apply(gmean_pos)
              .reset_index().rename(columns={"ratio":"improvement_pct"}))
    g["improvement_pct"] = (1 - g["improvement_pct"]) * 100.0
    # sort for readability
    g = g.sort_values(["span","h","improvement_pct"], ascending=[True, True, False])
    return g

def build_M2_bootstrap_ci_by_span(dsamp: pd.DataFrame, B: int = 1000, seed: int = 42) -> pd.DataFrame:
    if dsamp.empty or not set(["err","model"]).issubset(dsamp.columns):
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    # per (model,file,feature,span) RMSE
    cols = ["model","file","feature"]
    if "span" in dsamp.columns: cols.append("span")
    base = (dsamp.assign(err2=lambda d:d.err**2)
                  .groupby(cols).err2.mean().pipe(lambda s: np.sqrt(s)).reset_index())
    if "span" not in base.columns:
        base["span"] = "all"
    rows=[]
    for (m, sp), grp in base.groupby(["model","span"]):
        x = grp[grp.feature.isin(CORE)]["err2"]  # already sqrt in pipe -> err2 col is actually RMSE; reusing name
        x = grp[grp.feature.isin(CORE)]["err2"].values
        # the comment above: our column name is misnamed; compute x properly:
        x = grp[grp.feature.isin(CORE)]["err2"].values  # these are RMSE values from sqrt(mean(err^2))
        if len(x)==0: continue
        boots = []
        for _ in range(B):
            bs = rng.choice(x, size=len(x), replace=True)
            boots.append(gmean_pos(bs))
        rm = gmean_pos(x)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        rows.append({"model": m, "span": sp, "rmse_geo": rm, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    return out.sort_values(["span","rmse_geo"])

def build_J1_error_quantiles(dlog: pd.DataFrame) -> pd.DataFrame:
    if dlog.empty or "rmse_log" not in dlog.columns: return pd.DataFrame()
    df = dlog.copy()
    df["RMSE"] = 10**df["rmse_log"]
    group_keys = ["model","feature","h"] + (["span"] if "span" in df.columns else [])
    q = (df.groupby(group_keys)["RMSE"]
            .quantile([0.25,0.5,0.75,0.9,0.95])
            .unstack(-1).reset_index()
            .rename(columns={0.25:"q25",0.5:"q50",0.75:"q75",0.9:"q90",0.95:"q95"}))
    return q.sort_values(group_keys)

def build_E2b_wilcoxon_fdr(dlog: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats
    if dlog.empty or "rmse_log" not in dlog.columns: return pd.DataFrame()
    df = dlog.copy()
    df["RMSE"] = 10**df["rmse_log"]
    pivot_key = ["file","h","feature"]
    if "span" in df.columns: pivot_key.append("span")
    common = df[pivot_key].drop_duplicates()
    models = sorted(df.model.unique())
    rows=[]
    for i,m1 in enumerate(models):
        v1 = df[df.model==m1].merge(common, on=pivot_key)
        for j,m2 in enumerate(models):
            if j<=i: continue
            v2 = df[df.model==m2].merge(common, on=pivot_key)
            k = min(len(v1), len(v2))
            if k < 10: 
                continue
            # align by order
            r1 = v1["RMSE"].values[:k]
            r2 = v2["RMSE"].values[:k]
            try:
                p = stats.wilcoxon(r1, r2, zero_method="wilcox", alternative="two-sided").pvalue
            except Exception:
                p = np.nan
            rows.append({"model_i":m1,"model_j":m2,"p_raw":p})
    out = pd.DataFrame(rows)
    if out.empty: return out
    out["q_bh"] = benjamini_hochberg(out["p_raw"])
    return out.sort_values(["q_bh","p_raw"])

def bin_by_quantiles(series: pd.Series, q=4):
    try:
        return pd.qcut(series, q=q, duplicates="drop")
    except Exception:
        # fallback to cut
        return pd.cut(series, bins=q)

def build_R_bins(dsamp: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    if dsamp.empty or col not in dsamp.columns: return pd.DataFrame()
    if not set(["err","model","feature"]).issubset(dsamp.columns): return pd.DataFrame()
    d = dsamp.copy()
    d["bin"] = bin_by_quantiles(d[col], q=4)
    # per (model, bin, feature) RMSE
    rm = (d.assign(err2=lambda x:x.err**2)
            .groupby(["model","bin","feature"]).err2.mean()
            .pipe(lambda s: np.sqrt(s)).reset_index())
    rm = rm[rm.feature.isin(CORE)]
    g = (rm.groupby(["model","bin"]).err2.apply(gmean_pos)
            .reset_index().rename(columns={"err2":"rmse_geo"}))
    g["bin"] = g["bin"].astype(str)
    return g.sort_values(["bin","rmse_geo"]).rename(columns={"bin":f"{label}_bin"})

def build_RTN(dsamp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dsamp.empty or not set(["dr","dt","dn","h","model"]).issubset(dsamp.columns):
        return pd.DataFrame(), pd.DataFrame()
    d = dsamp.copy()
    d["at"] = np.abs(d["dt"])
    d["norm"] = np.sqrt(d["dr"]**2 + d["dt"]**2 + d["dn"]**2)
    at = (d.groupby(["model","h"]).at.apply(lambda s: float(np.sqrt(np.mean(s**2)))))
    at = at.reset_index().rename(columns={"at":"at_rmse_km"})
    nm = (d.groupby(["model","h"]).norm.apply(lambda s: float(np.sqrt(np.mean(s**2)))))
    nm = nm.reset_index().rename(columns={"norm":"norm_rmse_km"})
    return at.sort_values(["h","at_rmse_km"]), nm.sort_values(["h","norm_rmse_km"])

def build_F3_growth_slope(dlog: pd.DataFrame) -> pd.DataFrame:
    if dlog.empty or "rmse_log" not in dlog.columns: return pd.DataFrame()
    df = dlog.copy()
    # median across files/spans/windows for stability
    med = (df.groupby(["model","feature","h"]).rmse_log.median().reset_index())
    rows=[]
    for (m,feat), g in med.groupby(["model","feature"]):
        x = g["h"].values.astype(float)
        y = g["rmse_log"].values.astype(float)  # already log10(RMSE)
        if len(x) < 2: 
            continue
        # simple linear fit y = a*x + b
        a, b = np.polyfit(x, y, 1)
        # R^2
        yhat = a*x + b
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
        rows.append({"model":m,"feature":feat,"slope_log10_per_day":a,"r2":r2})
    out = pd.DataFrame(rows)
    return out.sort_values(["feature","slope_log10_per_day"])

def build_L1_leaderboard(dlog: pd.DataFrame) -> pd.DataFrame:
    if dlog.empty or "rmse_log" not in dlog.columns: return pd.DataFrame()
    df = dlog.copy()
    df["RMSE"] = 10**df["rmse_log"]
    # geo-mean across CORE features at each (model,h)
    # First, per (model,h,feature) median RMSE across cells
    med = (df[df.feature.isin(CORE)]
              .groupby(["model","h","feature"]).RMSE.median().reset_index())
    g = (med.groupby(["model","h"]).RMSE.apply(gmean_pos)
            .reset_index().rename(columns={"RMSE":"rmse_geo"}))
    rows=[]
    for h, grp in g.groupby("h"):
        grp = grp.sort_values("rmse_geo")
        grp["rank"] = np.arange(1, len(grp)+1)
        # margin to next best
        margin = []
        for i in range(len(grp)):
            if i == 0 and len(grp) > 1:
                nxt = grp.iloc[i+1].rmse_geo
                margin.append(100*(grp.iloc[i].rmse_geo/nxt - 1))
            else:
                margin.append(np.nan)
        grp["margin_pct_to_next"] = margin
        rows.append(grp[["h","rank","model","rmse_geo","margin_pct_to_next"]])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["h","rank"])

def build_U1_calibration(dsamp: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats
    need = {"y_pred_mu","y_pred_sigma","y_true"}
    if dsamp.empty or not need.issubset(dsamp.columns): return pd.DataFrame()
    z_grid = np.linspace(0.5, 2.5, 9)  # ~68–98%
    rows=[]
    y = dsamp["y_true"].values
    mu = dsamp["y_pred_mu"].values
    sig = dsamp["y_pred_sigma"].values
    for z in z_grid:
        cover = np.mean(np.abs(y - mu) <= z*sig)
        rows.append({
            "z": z,
            "nominal": 2*stats.norm.cdf(z)-1,
            "empirical": cover,
            "abs_error": abs((2*stats.norm.cdf(z)-1) - cover)
        })
    return pd.DataFrame(rows).sort_values("nominal")

def build_O1_outliers(dsamp: pd.DataFrame, topn: int = 30) -> pd.DataFrame:
    need = {"err","model","file","h","feature"}
    if dsamp.empty or not need.issubset(dsamp.columns): return pd.DataFrame()
    tri = (dsamp.assign(err2=lambda d:d.err**2)
                 .groupby(["model","file","h","feature"])
                 .agg(rmse=("err2", lambda s: float(np.sqrt(np.mean(s)))),
                      alt_km=("alt_km","mean") if "alt_km" in dsamp.columns else ("h", "size"),
                      ecc=("ecc","mean") if "ecc" in dsamp.columns else ("h","size"),
                      tle_age_d=("tle_age_d","mean") if "tle_age_d" in dsamp.columns else ("h","size"),
                      f107=("f107","mean") if "f107" in dsamp.columns else ("h","size")).reset_index())
    tri = tri.sort_values("rmse", ascending=False).head(topn)
    return tri

def build_H1_window_length(dsamp: pd.DataFrame) -> pd.DataFrame:
    # try a few possible column names that may encode window length
    cand_cols = [c for c in ["win_rev","window_rev","window_len_rev","window_days"] if c in dsamp.columns]
    if dsamp.empty or not cand_cols or "err" not in dsamp.columns:
        return pd.DataFrame()
    col = cand_cols[0]
    rm = (dsamp.assign(err2=lambda d:d.err**2)
                 .groupby(["model", col, "feature"]).err2.mean()
                 .pipe(lambda s: np.sqrt(s)).reset_index())
    rm = rm[rm.feature.isin(CORE)]
    g = (rm.groupby(["model", col]).err2.apply(gmean_pos).reset_index().rename(columns={"err2":"rmse_geo"}))
    return g.sort_values([col,"rmse_geo"]).rename(columns={col:"window_rev"})

def build_K1_reproducibility_digest(project_root: Path) -> pd.DataFrame:
    rec = {
        "start_ts": datetime.now().isoformat(timespec="seconds"),
        "numpy_version": None,
        "pandas_version": None,
        "pytorch_version": None,
        "git_commit": None,
        "git_branch": None,
        "python_version": sys.version.split()[0],
        "seed_policy": "deterministic seeds recommended (NumPy/PyTorch)"
    }
    try:
        import numpy as _np
        rec["numpy_version"] = _np.__version__
    except Exception: pass
    try:
        import pandas as _pd
        rec["pandas_version"] = _pd.__version__
    except Exception: pass
    try:
        import torch as _torch
        rec["pytorch_version"] = _torch.__version__
    except Exception: pass
    # git info
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=str(project_root)).decode().strip()
        rec["git_commit"] = commit
        branch = subprocess.check_output(["git","rev-parse","--abbrev-ref","HEAD"], cwd=str(project_root)).decode().strip()
        rec["git_branch"] = branch
    except Exception:
        pass
    return pd.DataFrame([rec])

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/benchmark_v2", type=Path)
    ap.add_argument("--baseline", default="sgp4_monotonic")
    args = ap.parse_args()

    ROOT = Path(args.root)
    TOUT = ensure_dir(ROOT / "tables_ext")

    # load
    dmet  = pd.read_csv(ROOT/"bench_metrics.csv") if (ROOT/"bench_metrics.csv").exists() else pd.DataFrame()
    dlog  = pd.read_csv(ROOT/"bench_metrics_log.csv") if (ROOT/"bench_metrics_log.csv").exists() else pd.DataFrame()
    dsamp = pd.read_csv(ROOT/"bench_samples.csv") if (ROOT/"bench_samples.csv").exists() else pd.DataFrame()
    profiles = load_profiles(Path("results/profiles"))

    # ---------------- G1: cost summary
    try:
        g1 = build_G1_cost_summary(dmet, profiles)
        if not g1.empty:
            save_table(g1, TOUT, "G1_cost_summary",
                       index=False,
                       latex_bold_min_cols=["RMSE_geo","latency_ms","params"])
    except Exception as e:
        print(f"⚠️  G1 failed: {e}")

    # ---------------- C5: sample sizes by grid
    try:
        c5 = build_C5_sample_sizes(dlog, dsamp)
        if not c5.empty:
            save_table(c5, TOUT, "C5_sample_sizes_by_grid", index=False)
    except Exception as e:
        print(f"⚠️  C5 failed: {e}")

    # ---------------- D2: improvement vs baseline (span × h)
    try:
        d2 = build_D2_improvement_vs_baseline(dlog, args.baseline)
        if not d2.empty:
            save_table(d2, TOUT, "D2_improvement_vs_baseline_by_span_h",
                       index=False, percent_cols=["improvement_pct"])
    except Exception as e:
        print(f"⚠️  D2 failed: {e}")

    # ---------------- M2: bootstrap CIs by span
    try:
        m2 = build_M2_bootstrap_ci_by_span(dsamp)
        if not m2.empty:
            save_table(m2, TOUT, "M2_rmse_bootstrap_ci_by_span",
                       index=False, latex_bold_min_cols=["rmse_geo"])
    except Exception as e:
        print(f"⚠️  M2 failed: {e}")

    # ---------------- J1: error quantiles by feature × h (× span if present)
    try:
        j1 = build_J1_error_quantiles(dlog)
        if not j1.empty:
            save_table(j1, TOUT, "J1_error_quantiles_by_feature_h",
                       index=False)
    except Exception as e:
        print(f"⚠️  J1 failed: {e}")

    # ---------------- E2b: Wilcoxon with BH FDR
    try:
        e2b = build_E2b_wilcoxon_fdr(dlog)
        if not e2b.empty:
            save_table(e2b, TOUT, "E2b_pairwise_wilcoxon_fdr_bh",
                       index=False, latex_bold_min_cols=["q_bh","p_raw"])
    except Exception as e:
        print(f"⚠️  E2b failed: {e}")

    # ---------------- R1..R4: robustness by bins
    try:
        r1 = build_R_bins(dsamp, "alt_km", "altitude")
        if not r1.empty:
            save_table(r1, TOUT, "R1_rmse_by_altitude_bin",
                       index=False, latex_bold_min_cols=["rmse_geo"])
    except Exception as e:
        print(f"⚠️  R1 failed: {e}")

    try:
        r2 = build_R_bins(dsamp, "ecc", "ecc")
        if not r2.empty:
            save_table(r2, TOUT, "R2_rmse_by_ecc_bin",
                       index=False, latex_bold_min_cols=["rmse_geo"])
    except Exception as e:
        print(f"⚠️  R2 failed: {e}")

    try:
        r3 = build_R_bins(dsamp, "tle_age_d", "tle_age_d")
        if not r3.empty:
            save_table(r3, TOUT, "R3_rmse_by_tle_age_bin",
                       index=False, latex_bold_min_cols=["rmse_geo"])
    except Exception as e:
        print(f"⚠️  R3 failed: {e}")

    try:
        r4 = build_R_bins(dsamp, "f107", "f107")
        if not r4.empty:
            save_table(r4, TOUT, "R4_rmse_by_f107_bin",
                       index=False, latex_bold_min_cols=["rmse_geo"])
    except Exception as e:
        print(f"⚠️  R4 failed: {e}")

    # ---------------- RTN1/RTN2: along-track & 3D norm
    try:
        rtn1, rtn2 = build_RTN(dsamp)
        if not rtn1.empty:
            save_table(rtn1, TOUT, "RTN1_alongtrack_rmse_by_h",
                       index=False, latex_bold_min_cols=["at_rmse_km"])
        if not rtn2.empty:
            save_table(rtn2, TOUT, "RTN2_norm_rmse_by_h",
                       index=False, latex_bold_min_cols=["norm_rmse_km"])
    except Exception as e:
        print(f"⚠️  RTN failed: {e}")

    # ---------------- F3: error growth slope
    try:
        f3 = build_F3_growth_slope(dlog)
        if not f3.empty:
            save_table(f3, TOUT, "F3_error_growth_slope",
                       index=False, latex_bold_min_cols=["slope_log10_per_day"])
    except Exception as e:
        print(f"⚠️  F3 failed: {e}")

    # ---------------- L1: leaderboard by horizon
    try:
        l1 = build_L1_leaderboard(dlog)
        if not l1.empty:
            save_table(l1, TOUT, "L1_leaderboard_by_horizon",
                       index=False, latex_bold_min_cols=["rmse_geo"])
    except Exception as e:
        print(f"⚠️  L1 failed: {e}")

    # ---------------- U1: calibration coverage
    try:
        u1 = build_U1_calibration(dsamp)
        if not u1.empty:
            save_table(u1, TOUT, "U1_calibration_coverage",
                       index=False, latex_bold_min_cols=["abs_error"])
    except Exception as e:
        print(f"⚠️  U1 failed: {e}")

    # ---------------- O1: top outlier windows
    try:
        o1 = build_O1_outliers(dsamp, topn=30)
        if not o1.empty:
            save_table(o1, TOUT, "O1_top_outlier_windows",
                       index=False)
    except Exception as e:
        print(f"⚠️  O1 failed: {e}")

    # ---------------- H1: window length ablation
    try:
        h1 = build_H1_window_length(dsamp)
        if not h1.empty:
            save_table(h1, TOUT, "H1_window_length_ablation",
                       index=False, latex_bold_min_cols=["rmse_geo"])
    except Exception as e:
        print(f"⚠️  H1 failed: {e}")

    # ---------------- K1: reproducibility digest
    try:
        k1 = build_K1_reproducibility_digest(Path.cwd())
        if not k1.empty:
            save_table(k1, TOUT, "K1_reproducibility_digest",
                       index=False)
    except Exception as e:
        print(f"⚠️  K1 failed: {e}")

    print(f"✅  Extra tables written to {TOUT.resolve()}")

if __name__ == "__main__":
    main()


"""
Compute Wilcoxon signed-rank tests and Cliff's delta vs. SGP4 baseline.

Works with bench_metrics.csv columns as provided:
  model, file, norad_id, feature, RMSE, MAPE%, h, w, span, ...

Key decisions for your dataset:
- Baseline is 'sgp4_monotonic' (one value per (feature, h)).
- Other models may have many values per (feature, h); script pairs them
  to the SGP4 baseline by broadcasting the baseline value to the same length.
- Features are mapped from full names to short codes (a/n/e) for compact tables.
- Benjamini–Hochberg FDR (q-values) is computed across all tests.

Outputs:
- CSV with all rows
- LaTeX table with vertical/horizontal lines (boxed style)

Usage:
  python -m scripts.make_significance_table \
    --input results/benchmark_v2/bench_metrics.csv \
    --output_csv results/benchmark_v2/significance_results.csv \
    --output_tex results/benchmark_v2/significance_table.tex \
    --baseline sgp4_monotonic \
    --features semi_major_axis,mean_motion,eccentricity \
    --horizons 1,3,7,15,30,90
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:

    if x.size == 0 or y.size == 0:
        return np.nan
    dif = x[:, None] - y[None, :]
    gt = np.sum(dif > 0)
    lt = np.sum(dif < 0)
    return (gt - lt) / (x.size * y.size)


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    p = pvals.values.astype(float)
    n = np.isfinite(p).sum()
    q = np.full_like(p, np.nan, dtype=float)
    if n == 0:
        return pd.Series(q, index=pvals.index)

    # sort finite p-values
    finite_idx = np.where(np.isfinite(p))[0]
    order = finite_idx[np.argsort(p[finite_idx])]
    ranked_p = p[order]
    m = len(ranked_p)

    # BH
    bh = ranked_p * m / (np.arange(m) + 1)
    # monotone non-increasing from the end
    bh = np.minimum.accumulate(bh[::-1])[::-1]

    q[order] = np.clip(bh, 0, 1)
    return pd.Series(q, index=pvals.index)


def main(args):
    # Load
    df = pd.read_csv(args.input)

    # Normalize column names
    cols_map = {c: c.strip() for c in df.columns}
    df.rename(columns=cols_map, inplace=True)

    # Ensure required columns
    needed = {"model", "feature", "RMSE", "h"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV must have columns {needed}, got {set(df.columns)}")

    # Clean types
    df["model"] = df["model"].astype(str)
    df["feature"] = df["feature"].astype(str)
    df["h"] = pd.to_numeric(df["h"], errors="coerce")
    df = df[np.isfinite(df["h"])].copy()
    df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
    df = df[np.isfinite(df["RMSE"])].copy()

    # Feature filter and mapping
    feature_list = [f.strip() for f in args.features.split(",") if f.strip()]
    feature_map = {
        "semi_major_axis": "a",
        "mean_motion": "n",
        "eccentricity": "e",
    }
    # keep only features we know how to map
    df = df[df["feature"].isin(feature_list)].copy()

    # Horizon filter
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    df = df[df["h"].isin(horizons)].copy()

    baseline = args.baseline  # e.g., 'sgp4_monotonic'

    # Build results
    rows = []
    models = sorted([m for m in df["model"].unique() if m != baseline])
    for feat in feature_list:
        # baseline vector(s) for this feature at each horizon
        for h in horizons:
            b = df[(df["model"] == baseline) & (df["feature"] == feat) & (df["h"] == h)]["RMSE"].values
            if b.size == 0:
                # no baseline for this cell, skip
                continue

            for model in models:
                x = df[(df["model"] == model) & (df["feature"] == feat) & (df["h"] == h)]["RMSE"].values
                if x.size == 0:
                    continue

                # Broadcast baseline to x length if needed
                if b.size == 1:
                    y = np.repeat(b[0], x.size)
                else:
                    # Truncate to min length if both have multiple values
                    n = min(x.size, b.size)
                    x = x[:n]
                    y = b[:n]

                # Clean any non-finite
                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]
                if x.size == 0:
                    continue

                # Median ratio vs SGP4
                ratio = np.median(x / y)

                # Wilcoxon signed-rank
                p = np.nan
                try:
                    # Pratt handles zero-diff pairs gracefully
                    stat, p = wilcoxon(x, y, zero_method="pratt", alternative="two-sided", method="auto")
                except Exception:
                    p = np.nan

                # Cliff's delta
                delta = cliffs_delta(x, y)

                rows.append({
                    "Model": model,
                    "Horizon": h,
                    "Feature": feature_map.get(feat, feat),
                    "MedianRatioToSGP4": ratio,
                    "CliffsDelta": delta,
                    "p_value": p,
                    "N_pairs": int(x.size),
                })

    if not rows:
        # Helpful diagnostic
        print("No rows were produced. Check that baseline name, features, and horizons match your CSV.")
        print(f"Baseline searched: {baseline}")
        print(f"Features searched: {feature_list}")
        print(f"Horizons searched: {horizons}")
        # Dump quick inventory
        print("Sample head of dataframe:")
        print(df.head())
        return

    res = pd.DataFrame(rows)

    # Add BH-FDR q-values across all tests
    res["q_value"] = benjamini_hochberg(res["p_value"])

    # Sort
    res.sort_values(["Feature", "Horizon", "MedianRatioToSGP4"], inplace=True, ignore_index=True)

    # Save CSV
    pd.options.display.float_format = "{:,.6g}".format
    res.to_csv(args.output_csv, index=False)

    # Build a boxed LaTeX table
    with open(args.output_tex, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("  \\centering\n")
        f.write("  \\caption{Wilcoxon signed-rank outcomes and Cliff’s $\\delta$ versus SGP4. Ratios are median error relative to SGP4 (values $<1$ indicate improvement). q-values are Benjamini--Hochberg FDR.}\n")
        f.write("  \\label{tab:significance}\n")
        f.write("  \\begin{tabular}{|l|c|c|c|c|c|c|}\n")
        f.write("    \\hline\n")
        f.write("    Model & Horizon (d) & Feature & Median Ratio & Cliff’s $\\delta$ & $p$-value & $q$-value \\\\\n")
        f.write("    \\hline\n")
        # Keep it tidy: optionally filter to rows with at least min_n pairs
        table_df = res[res["N_pairs"] >= args.min_n_pairs].copy()
        if table_df.empty:
            table_df = res.copy()
        for _, r in table_df.iterrows():
            f.write(f"    {r['Model']} & {int(r['Horizon'])} & {r['Feature']} & "
                    f"{r['MedianRatioToSGP4']:.2f} & {r['CliffsDelta']:.2f} & "
                    f"{(r['p_value'] if np.isfinite(r['p_value']) else np.nan):.2e} & "
                    f"{(r['q_value'] if np.isfinite(r['q_value']) else np.nan):.2e} \\\\\n")
        f.write("    \\hline\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"[OK] Wrote: {args.output_csv}")
    print(f"[OK] Wrote: {args.output_tex}")
    # Quick summary to console
    print(res.groupby(["Feature", "Horizon"]).size().rename("tests_per_cell"))
    print("Note: If some p-values are NaN, there may be all-zero differences or too-few pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to bench_metrics.csv")
    parser.add_argument("--output_csv", default="results/benchmark_v2/significance_results.csv")
    parser.add_argument("--output_tex", default="results/benchmark_v2/significance_table.tex")
    parser.add_argument("--baseline", default="sgp4_monotonic")
    parser.add_argument("--features", default="semi_major_axis,mean_motion,eccentricity")
    parser.add_argument("--horizons", default="1,3,7,15,30,90")
    parser.add_argument("--min_n_pairs", type=int, default=1, help="Only include rows with at least this many paired samples in the LaTeX table")
    args = parser.parse_args()
    main(args)

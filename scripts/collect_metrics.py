#!/usr/bin/env python3
# scripts/collect_matrix_v2.py
"""
Benchmark v2 collector:
- Keeps your per-file metrics flow
- ALSO ingests optional per-sample error files for CIs, paired tests, eCDFs
- Writes to results/benchmark_v2/

Usage
-----
python -m scripts.collect_matrix_v2 \
  --root results \
  --sat-filter "" \
  --out-root results/benchmark_v2
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, re, sys
import numpy as np
import pandas as pd

NUM = (int, float)

ALIAS = {
    "inclination":  "inclination_deg",
    "raan":         "raan_deg",
    "arg_perigee":  "arg_perigee_deg",
    "mean_anomaly": "mean_anomaly_deg",
}

RE_H  = re.compile(r"_h(?P<h>\d+)")
RE_W  = re.compile(r"_w(?P<w>\d+)")
RE_SP = re.compile(r"_(?P<span>last\d+d|last\d+yr|full)")
RE_SAT= re.compile(r"(?:sat|norad)?(?P<sat>\d{4,6})")  # best effort

def norm_feat(x:str)->str: return ALIAS.get(x, x)

def collect_file_metrics(root:Path, filt:str)->pd.DataFrame:
    rows = []
    for model_dir in root.iterdir():
        if not model_dir.is_dir(): continue
        model = model_dir.name
        for jf in model_dir.glob("metrics_*.json"):
            stem = jf.stem.replace("metrics_","")
            if filt and filt not in stem: continue
            try:
                data = json.loads(jf.read_text())
            except Exception as e:
                print(f"⚠️  skip {jf}: {e}")
                continue

            top_num = {k:v for k,v in data.items() if isinstance(v, NUM)}
            json_h = data.get("horizon_days")

            metrics = data.get("metrics", [])
            if not metrics: continue

            h  = RE_H.search(stem)
            w  = RE_W.search(stem)
            sp = RE_SP.search(stem)
            sat= RE_SAT.search(stem)

            for m in metrics:
                feat = norm_feat(m.get("feature",""))
                if not feat: continue
                rmse = m.get("RMSE", m.get("RMSE_deg"))
                if rmse is None: continue

                rows.append({
                    "model":   model,
                    "file":    stem,
                    "norad_id": int(sat.group("sat")) if sat else np.nan,
                    "feature": feat,
                    "RMSE":    float(rmse),
                    "MAPE%":   m.get("MAPE%"),
                    "h":       int(h.group("h")) if h else (json_h or np.nan),
                    "w":       int(w.group("w")) if w else np.nan,
                    "span":    sp.group("span") if sp else np.nan,
                    **top_num
                })
    if not rows:
        sys.exit("❌  No metrics_*.json found. Check --root.")
    df = pd.DataFrame(rows)
    for c in ("RMSE","MAPE%","h","w","overall_acc_percent"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def read_any(path:Path)->pd.DataFrame:
    if path.suffix == ".parquet": return pd.read_parquet(path)
    return pd.read_csv(path)

def collect_sample_errors(root:Path, filt:str)->pd.DataFrame:
    frames = []
    for model_dir in root.iterdir():
        if not model_dir.is_dir(): continue
        model = model_dir.name
        for f in list(model_dir.glob("samples/errors_*.parquet")) + \
                 list(model_dir.glob("samples/errors_*.csv")):
            if filt and filt not in f.stem: continue
            try:
                d = read_any(f)
            except Exception as e:
                print(f"⚠️  skip {f}: {e}")
                continue

            # expected columns (best effort normalisation)
            colmap = {c.lower():c for c in d.columns}
            def pick(*names):
                for n in names:
                    if n in d.columns: return n
                    if n in colmap: return colmap[n]
                return None

            need = {
                "file": pick("file","segment_id"),
                "norad_id": pick("norad_id","sat","sat_id"),
                "t0":   pick("t0","epoch","start_time"),
                "h":    pick("h","horizon","horizon_d","h_days"),
                "w":    pick("w","window","win"),
                "span": pick("span"),
                "feature": pick("feature","param"),
                "err":  pick("err","error","y_err","residual"),
                "model": None  # fill constant
            }
            missing = [k for k,v in need.items() if v is None and k!="model"]
            if missing:
                print(f"⚠️  {f.name}: missing cols {missing} – trying to infer")
            d2 = pd.DataFrame({
                "model":   model,
                "file":    d.get(need["file"], f.stem),
                "norad_id": pd.to_numeric(d.get(need["norad_id"], np.nan), errors="coerce"),
                "t0":      pd.to_datetime(d.get(need["t0"], pd.NaT), errors="coerce"),
                "h":       pd.to_numeric(d.get(need["h"], np.nan), errors="coerce"),
                "w":       pd.to_numeric(d.get(need["w"], np.nan), errors="coerce"),
                "span":    d.get(need["span"], np.nan),
                "feature": d.get(need["feature"], np.nan).map(norm_feat),
                "err":     pd.to_numeric(d.get(need["err"], np.nan), errors="coerce")
            })
            # optional extras pass-through if present
            for extra in ("y_true","y_pred","y_base","dr","dt","dn",
                          "y_pred_mu","y_pred_sigma",
                          "alt_km","ecc","tle_age_d","f107"):
                if extra in d.columns:
                    d2[extra] = d[extra]
            frames.append(d2)
    if not frames:
        print("ℹ️  No per-sample error files found – advanced plots will be skipped.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results", type=Path)
    ap.add_argument("--sat-filter", default="")
    ap.add_argument("--out-root", default="results/benchmark_v2", type=Path)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    df_metrics = collect_file_metrics(args.root, args.sat_filter)
    df_metrics.to_csv(args.out_root / "bench_metrics.csv", index=False)

    df_log = df_metrics[df_metrics["RMSE"]>0].copy()
    df_log["rmse_log"] = np.log10(df_log["RMSE"])
    df_log.to_csv(args.out_root / "bench_metrics_log.csv", index=False)

    df_samples = collect_sample_errors(args.root, args.sat_filter)
    if not df_samples.empty:
        # Normalise feature names and basic types
        df_samples["feature"] = df_samples["feature"].map(norm_feat)
        df_samples.to_csv(args.out_root / "bench_samples.csv", index=False)

    print("✅  Wrote:",
          args.out_root / "bench_metrics.csv",
          args.out_root / "bench_metrics_log.csv",
          (args.out_root / "bench_samples.csv" if not df_samples.empty else "(no samples)"),
          sep="\n")

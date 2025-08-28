#!/usr/bin/env python3
"""
src/preprocess.py
=================

Generate model‑ready datasets from raw Space‑Track TLE files.

For *each* requested “span” (look‑back in days) we create
    • a tidy CSV
    • X/Y windowed NPZ datasets (one per <window, horizon>)
    • a fitted StandardScaler (joblib)

Examples
--------
# full history (default) + 3 mo + 6 mo + 1 y, trig encoding
python -m src.preprocess data/raw/37746_tle_*.txt \
       --spans 0 90 180 365

# only 60‑day windows, deg encoding, just the last 180 days
python -m src.preprocess data/raw/37746_tle_*.txt \
       --spans 180 --windows 60 --encode deg
"""
from __future__ import annotations

import argparse, re
from datetime import datetime, timedelta
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

MU_EARTH   = 398_600.4418                      # km³ s⁻²
ANGLE_COLS = ["inclination", "raan", "arg_perigee", "mean_anomaly"]

# ───────────────────────── TLE helpers ──────────────────────────
def _epoch_to_datetime(ep:str)->datetime:
    yr=int(ep[:2]); yr+=2000 if yr<57 else 1900
    return datetime(yr,1,1)+timedelta(days=float(ep[2:])-1)

def _n_to_sma(n_rev_day:float)->float:
    n=n_rev_day*2*np.pi/86_400
    return (MU_EARTH**(1/3))/n**(2/3)

def parse_tle(txt:Path)->pd.DataFrame:
    pat=re.compile(r"^1 (\d{5})"); rows=[]
    L=txt.read_text().strip().splitlines()
    for i in range(0,len(L)-1,2):
        l1,l2=L[i],L[i+1]
        if not pat.match(l1): continue
        rows.append(dict(
            epoch=_epoch_to_datetime(l1[18:32]),
            inclination=float(l2[8:16]),
            raan=float(l2[17:25]),
            eccentricity=float(f"0.{l2[26:33]}"),
            arg_perigee=float(l2[34:42]),
            mean_anomaly=float(l2[43:51]),
            mean_motion=float(l2[52:63]),
        ))
        rows[-1]["semi_major_axis"]=_n_to_sma(rows[-1]["mean_motion"])
    df=pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    # .empty is a boolean property – do NOT call it
    if df.empty:
        raise ValueError(f"No valid TLE pairs found in {txt}")
    return df

# ─────────────────────── window builder ────────────────────────
def build_windows(df:pd.DataFrame,L:int,H:int)->tuple[np.ndarray,np.ndarray]:
    feats=df.drop(columns=["epoch"]).values.astype(np.float32)
    X,Y=[],[]
    for i in range(len(df)-L-H+1):
        X.append(feats[i:i+L]); Y.append(feats[i+L+H-1])
    return np.asarray(X),np.asarray(Y)

# ───────────────────────── main worker ─────────────────────────
def process_span(df:pd.DataFrame, span:int, *, raw_stem:str,
                 windows:list[int], horizons:list[int], enc:str, outdir:Path):
    """
    span == 0  → full history; otherwise last <span> days.
    """
    tag = "full" if span==0 else f"last{span}d"
    if span>0:
        cutoff=df["epoch"].max()-timedelta(days=span)
        df=df[df["epoch"]>=cutoff].reset_index(drop=True)
        if len(df)<max(windows)+max(horizons):
            print(f"⚠  {raw_stem}: span {span} d not enough rows – skipped.")
            return

    # ---------- feature engineering ----------
    df_proc = df.copy()
    if enc=="trig":
        df_proc[ANGLE_COLS]=np.deg2rad(df_proc[ANGLE_COLS])
        for c in ANGLE_COLS:
            df_proc[f"{c}_sin"]=np.sin(df_proc[c])
            df_proc[f"{c}_cos"]=np.cos(df_proc[c])
        df_proc.drop(columns=ANGLE_COLS,inplace=True)
    else:  # deg
        df_proc.rename(columns={c:f"{c}_deg" for c in ANGLE_COLS},inplace=True)

    # ---------- scaling ----------
    scaler=StandardScaler().fit(df_proc.drop(columns=["epoch"]))
    df_proc[df_proc.columns.difference(["epoch"])] = scaler.transform(
        df_proc.drop(columns=["epoch"])
    )

    # ---------- outputs ----------
    base=f"{raw_stem}_{tag}"
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/f"{base}.csv").write_text(df_proc.to_csv(index=False))
    joblib.dump(scaler,outdir/f"{base}_scaler.gz")

    for L in windows:
        for H in horizons:
            X,Y=build_windows(df_proc,L,H)
            if len(X)==0: continue
            np.savez_compressed(
                outdir/f"{base}_enc{enc}_w{L}_h{H}.npz", X=X, y=Y
            )
    print(f"✅  {raw_stem} [{tag}] – CSV, scaler, NPZs written.")

# ─────────────────────────── CLI ──────────────────────────────
def _cli():
    p=argparse.ArgumentParser(description="Prepare orbital datasets.")
    p.add_argument("files",type=Path,nargs="+",help="raw *.txt files")
    p.add_argument("--spans",type=int,nargs="+",default=[0],
                   help="Look‑back days. 0 = full history")
    p.add_argument("--windows",type=int,nargs="+",default=[30],
                   help="Input window lengths (days)")
    p.add_argument("--horizons",type=int,nargs="+",default=[1,3,7,30],
                   help="Prediction horizons (days)")
    p.add_argument("--encode",choices=["trig","deg"],default="trig")
    args=p.parse_args()

    outdir=Path("data/processed")
    for raw in args.files:
        for path in sorted(raw.parent.glob(raw.name)):
            df=parse_tle(path)
            for span in args.spans:
                process_span(df,span,
                             raw_stem=path.stem,
                             windows=args.windows,
                             horizons=args.horizons,
                             enc=args.encode,
                             outdir=outdir)

if __name__=="__main__":
    _cli()

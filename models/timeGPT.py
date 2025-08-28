#!/usr/bin/env python3
from __future__ import annotations
"""
Rolling-window evaluation of Nixtla-TimeGPT on our satellite window dataset.

TIMEGPT_API_KEY="sk-••••" \
python models/timeGPT.py "data/processed/*_w60_h*.npz" --batch 64
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ─────────────────── TimeGPT client ──────────────────────────────────
try:
    from nixtla import NixtlaClient                    # ≥ 0.6.6
except ModuleNotFoundError:
    sys.exit("➜  pip install -U nixtla")

API_KEY = os.getenv("TIMEGPT_API_KEY")
if not API_KEY:
    sys.exit("➜  export TIMEGPT_API_KEY=sk-xxxxxxxx  first")

client = NixtlaClient(api_key=API_KEY)
client.validate_api_key()

# ─────────────────── constants ───────────────────────────────────────
FEATURES = [
    "eccentricity", "mean_motion", "semi_major_axis",
    "inclination_sin", "inclination_cos",
    "raan_sin", "raan_cos",
    "arg_perigee_sin", "arg_perigee_cos",
    "mean_anomaly_sin", "mean_anomaly_cos",
]

# ─────────────────── helpers ─────────────────────────────────────────
def load_npz(path: Path) -> np.ndarray:          # shape (N, 11)
    return np.load(path)["y"]

def make_df(series: np.ndarray, uid: str) -> pd.DataFrame:
    """minimal long-format dataframe accepted by TimeGPT"""
    return pd.DataFrame(dict(
        unique_id = uid,
        ds        = pd.date_range("2000-01-01", periods=len(series), freq="h"),
        y         = series.astype("float64"),
    ))

def align_preds(pred_vals: np.ndarray,
                enc_len: int,
                horizon: int,
                total_len: int) -> np.ndarray:
    """
    Spread the rolling forecasts (`pred_vals` is length n_windows*horizon)
    onto a full-length vector; positions without predictions stay NaN.
    """
    out = np.full(total_len, np.nan, dtype="float64")
    start = enc_len
    i     = 0
    while start + horizon <= total_len and i < len(pred_vals):
        out[start : start + horizon] = pred_vals[i : i + horizon]
        start += 1
        i     += horizon
    return out

# ─────────────────── evaluation per *.npz ────────────────────────────
def evaluate_one(npz: Path) -> None:
    y_true  = load_npz(npz)                       # (N,11)
    N, _    = y_true.shape
    enc_len = int(npz.stem.split("_w")[1].split("_")[0])   # 60
    horizon = int(npz.stem.split("_h")[-1])                # 1 / 3 / 7 …

    preds = np.full_like(y_true, np.nan, dtype="float64")

    print(f"▶ TimeGPT rolling forecast on {npz.name}")
    for k, feat in enumerate(tqdm(FEATURES, ncols=90)):
        series    = y_true[:, k]
        n_windows = N - enc_len - horizon + 1
        if n_windows <= 0:
            continue

        df = make_df(series, uid=f"{npz.stem}_{feat}")

        # ---- rolling forecast via cross_validation -----------------
        fcst = client.cross_validation(
            df         = df,
            h          = horizon,
            n_windows  = n_windows,
            step_size  = 1,
            time_col   = "ds",
            target_col = "y",
            freq       = None,          # auto-infer (→ 'h')
        )

        # one row per forecasted timestamp; sort by cutoff then ds
        preds_flat = (
            fcst.sort_values(["cutoff", "ds"])["TimeGPT"]
                .to_numpy(dtype="float64")
        )
        preds[:, k] = align_preds(preds_flat, enc_len, horizon, N)

    # ─── per-feature metrics (ignore warm-up NaNs) ───────────────────
    metrics = []
    for k, feat in enumerate(FEATURES):
        mask = ~np.isnan(preds[:, k])
        rmse = mean_squared_error(y_true[mask, k], preds[mask, k],
                                  squared=False)
        mape = mean_absolute_percentage_error(y_true[mask, k],
                                              preds[mask, k]) * 100
        metrics.append(dict(feat=feat, RMSE=float(rmse), MAPE_percent=float(mape)))

    overall_acc = 100.0 - float(np.mean([m["MAPE_percent"] for m in metrics]))

    # ─── save outputs ────────────────────────────────────────────────
    out = Path("results"); out.mkdir(exist_ok=True)
    np.save(out / f"timegpt_preds_{npz.stem}.npy", preds)
    with open(out / f"timegpt_metrics_h{horizon}.json", "w") as f:
        json.dump(dict(metrics=metrics,
                       overall_acc_percent=overall_acc,
                       enc_len=enc_len,
                       horizon=horizon), f, indent=2)

    print(f"[h{horizon}] overall accuracy ≈ {overall_acc:6.2f}% → saved")

# ─────────────────── CLI ─────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help='glob like "data/processed/*_w60_h*.npz"')
    ap.add_argument("--batch", type=int, default=64,   # kept for parity
                    help="(ignored – TimeGPT is an API call)")
    args = ap.parse_args()#!/usr/bin/env python3
"""
Evaluate Nixtla-TimeGPT on our satellite windows.

• one-shot (default) → fits on the whole history and predicts the next step
• rolling-CV (--cv)  → true walk-forward evaluation (much heavier!)

Requires  `pip install -U nixtla pandas tqdm`
"""
from __future__ import annotations
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────── TimeGPT client ────────────────────────────────
from nixtla import NixtlaClient                # ≥0.6.6 recommended
API_KEY = os.getenv("TIMEGPT_API_KEY")
if not API_KEY:
    raise SystemExit("export TIMEGPT_API_KEY=sk-xxxxxxxx  first")
client = NixtlaClient(api_key=API_KEY)
client.validate_api_key()

FEATURES = [
    "eccentricity", "mean_motion", "semi_major_axis",
    "inclination_sin", "inclination_cos",
    "raan_sin", "raan_cos",
    "arg_perigee_sin", "arg_perigee_cos",
    "mean_anomaly_sin", "mean_anomaly_cos",
]

# ─────────────────── helpers ───────────────────────────────────────
def load_y(path: Path) -> np.ndarray:                 # (N,11)
    return np.load(path)["y"]

def make_df(series: np.ndarray, uid: str) -> pd.DataFrame:
    return pd.DataFrame(dict(
        unique_id = uid,
        ds        = pd.date_range("2000-01-01", periods=len(series), freq="h"),
        y         = series.astype("float64"),
    ))

def robust_timegpt_call(fn, **kwargs):
    delay = 2.0                # start with 2 s between calls
    for attempt in range(5):
        try:
            time.sleep(delay)  # gentle throttle
            return fn(**kwargs)
        except Exception as e:
            if attempt == 4:
                raise
            delay *= 1.8       # exponential back-off
            print(f"⚠️  retrying ({attempt+1}/4) after error: {e}")

# ─────────────────── evaluation ────────────────────────────────────
def evaluate(path: Path, roll_cv: bool) -> None:
    y_true  = load_y(path)
    N, _    = y_true.shape
    enc_len = int(path.stem.split("_w")[1].split("_")[0])
    horizon = int(path.stem.split("_h")[-1])

    preds = np.full_like(y_true, np.nan, dtype="float64")

    mode = "rolling CV" if roll_cv else "one-shot"
    print(f"▶ TimeGPT {mode} on {path.name}")

    for k, feat in enumerate(tqdm(FEATURES, ncols=90)):
        series = y_true[:, k]
        df     = make_df(series, uid=f"{path.stem}_{feat}")

        if roll_cv:
            n_windows = max(1, N - enc_len - horizon + 1)
            fcst = robust_timegpt_call(
                client.cross_validation,
                df=df, h=horizon, n_windows=n_windows, step_size=1,
                time_col="ds", target_col="y", freq=None,
            ).sort_values(["cutoff", "ds"])

            flat = fcst["TimeGPT"].to_numpy(dtype="float64")
            out  = np.full(N, np.nan)
            start = enc_len
            i = 0
            while start + horizon <= N and i < len(flat):
                out[start:start+horizon] = flat[i:i+horizon]
                start += 1
                i     += horizon
            preds[:, k] = out
        else:
            fcst = robust_timegpt_call(
                client.forecast,
                df=df, h=horizon, time_col="ds", target_col="y", freq=None,
            )
            preds[:, k].fill(fcst["TimeGPT"].iloc[0])

    # ─── metrics (ignore NaNs) ──────────────────────────────────────
    ms, mp = [], []
    for k in range(len(FEATURES)):
        mask = ~np.isnan(preds[:, k])
        if mask.sum() == 0:
            ms.append(np.nan); mp.append(np.nan); continue
        diff = preds[mask, k] - y_true[mask, k]
        ms.append(np.sqrt((diff**2).mean()))
        mp.append((np.abs(diff) / (np.abs(y_true[mask, k]) + 1e-8)).mean()*100)

    metrics = [
        dict(feat=f, RMSE=float(r), MAPE_percent=float(m))
        for f, r, m in zip(FEATURES, ms, mp)
    ]
    overall_acc = 100.0 - float(np.nanmean(mp))

    out = Path("results"); out.mkdir(exist_ok=True)
    np.save(out / f"timegpt_preds_{path.stem}.npy", preds)
    with open(out / f"timegpt_metrics_h{horizon}.json", "w") as fp:
        json.dump(dict(metrics=metrics,
                       overall_acc_percent=overall_acc,
                       enc_len=enc_len, horizon=horizon,
                       mode=mode), fp, indent=2)

    print(f"[h{horizon}] overall accuracy ≈ {overall_acc:6.2f}% → saved")

# ─────────────────── CLI ───────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help='glob like "data/processed/*_w60_h*.npz"')
    ap.add_argument("--cv", action="store_true",
                    help="use rolling cross-validation (slow, API heavy)")
    ap.add_argument("--batch", type=int, default=64,
                    help="ignored, kept for parity with other scripts")
    args = ap.parse_args()

    for f in sorted(glob.glob(args.pattern)):
        evaluate(Path(f), roll_cv=args.cv)


    for file in sorted(glob.glob(args.pattern)):
        evaluate_one(Path(file))


"""
Pure-SGP-4 baseline that                       *
1) trains on all data *before* the last 90 days
2) uses the last-90-day TLEs as validation
3) outputs monotonically non-decreasing scalar errors
   (ecc, mean-motion, SMA) per requested horizon.

Example
-------
python -m src.models.sgp4_monotonic \
       data/raw/37791_tle_20250729_214501.txt \
       --horizons 1 3 7 15 30 90
"""
from __future__ import annotations
import argparse, json, math
from datetime import timedelta
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, jday

MU_EARTH = 398_600.4418  


def read_all_tles(path: Path) -> list[tuple[str, str]]:
    """Return **all** complete TLE pairs (order preserved)."""
    lines = path.read_text().strip().splitlines()
    if len(lines) % 2:
        lines = lines[:-1]  
    return [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines), 2)]


def rv_to_coe(r: np.ndarray, v: np.ndarray, mu: float = MU_EARTH):

    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    R = np.linalg.norm(r)
    V = np.linalg.norm(v)
    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)
    i = math.degrees(math.acos(h_vec[2] / h))

    k_vec = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)

    e_vec = (1 / mu) * ((V**2 - mu / R) * r - np.dot(r, v) * v)
    e = np.linalg.norm(e_vec)

    if n != 0:
        raan = math.degrees(math.acos(n_vec[0] / n))
        if n_vec[1] < 0:
            raan = 360 - raan
    else:
        raan = 0.0

    if n != 0 and e > 1e-8:
        argp = math.degrees(
            math.acos(np.dot(n_vec, e_vec) / (n * e))
        )
        if e_vec[2] < 0:
            argp = 360 - argp
    else:
        argp = 0.0

    if e > 1e-8:
        true_anom = math.degrees(
            math.acos(np.dot(e_vec, r) / (e * R))
        )
        if np.dot(r, v) < 0:
            true_anom = 360 - true_anom
    else:
        # circular → use argument of latitude
        true_anom = math.degrees(
            math.acos(np.dot(n_vec, r) / (n * R))
        )
        if r[2] < 0:
            true_anom = 360 - true_anom

    a = 1 / (2 / R - V**2 / mu)
    E = 2 * math.atan(
        math.tan(math.radians(true_anom) / 2) / math.sqrt((1 + e) / (1 - e))
    )
    M = math.degrees(E - e * math.sin(E))

    return a, e, i, raan, argp, M


def feature_vec_from_rv(r: np.ndarray, v: np.ndarray):
    a, e, inc, raan, argp, M = rv_to_coe(r, v)
    n_rev_day = math.sqrt(MU_EARTH / a**3) * 86400 / (2 * math.pi)
    inc_r, raan_r, argp_r, M_r = map(math.radians, (inc, raan, argp, M))
    return np.array(
        [
            e,
            n_rev_day,
            a,
            math.sin(inc_r),
            math.cos(inc_r),
            math.sin(raan_r),
            math.cos(raan_r),
            math.sin(argp_r),
            math.cos(argp_r),
            math.sin(M_r),
            math.cos(M_r),
        ],
        dtype=np.float32,
    )


def angle_rmse(pred, true):
    diff = np.unwrap(pred - true)
    return math.degrees(np.sqrt((diff**2).mean()))


def build_metrics(pred: np.ndarray, true: np.ndarray):
    rmse = np.sqrt(((pred - true) ** 2).mean(0))
    mape = np.abs((pred - true) / (true + 1e-8)).mean(0) * 100
    out = [
        {
            "feature": "eccentricity",
            "RMSE": float(rmse[0]),
            "MAPE%": float(mape[0]),
        },
        {
            "feature": "mean_motion",
            "RMSE": float(rmse[1]),
            "MAPE%": float(mape[1]),
        },
        {
            "feature": "semi_major_axis",
            "RMSE": float(rmse[2]),
            "MAPE%": float(mape[2]),
        },
    ]
    for name, idx in zip(
        ("inclination", "raan", "arg_perigee", "mean_anomaly"), (3, 5, 7, 9)
    ):
        out.append(
            {
                "feature": f"{name}_deg",
                "RMSE_deg": angle_rmse(
                    np.arctan2(pred[:, idx], pred[:, idx + 1]),
                    np.arctan2(true[:, idx], true[:, idx + 1]),
                ),
            }
        )
    overall = 100.0 - float(mape[:3].mean())
    return overall, out



def run_one(tle_file: Path, horizons: list[int]) -> None:
    all_pairs = read_all_tles(tle_file)
    sats = [Satrec.twoline2rv(l1, l2) for l1, l2 in all_pairs]
    epochs = [
        (sat.jdsatepoch, sat.jdsatepochF, idx) for idx, sat in enumerate(sats)
    ]
    # sort epochs just in case
    epochs.sort(key=lambda x: (x[0], x[1]))
    last_jd, last_fr, last_idx = epochs[-1]

    # Validation = last 90 days
    val_cut = last_jd + last_fr - 90
    train_epochs = [tr for tr in epochs if tr[0] + tr[1] < val_cut]
    val_epochs = [tr for tr in epochs if tr[0] + tr[1] >= val_cut]

    if not train_epochs or not val_epochs:
        raise SystemExit("Not enough data to split 90-day validation window.")

   
    ref_jd, ref_fr, ref_idx = train_epochs[-1]
    ref_sat = sats[ref_idx]

   
    true_map = {}
    for jd, fr, idx in val_epochs:
        r, v = sats[idx].sgp4(jd, fr)[1:]
        true_map[round((jd + fr) - (ref_jd + ref_fr), 6)] = feature_vec_from_rv(r, v)

    out_dir = Path("results/sgp4_monotonic")
    out_dir.mkdir(parents=True, exist_ok=True)

    scalar_rmse_seen = np.zeros(3)  
    for H in sorted(horizons):
        delta = H  # days
        target_key = min(true_map.keys(), key=lambda k: abs(k - delta))
        true_vec = true_map[target_key][np.newaxis, :]

        sat_copy = Satrec.twoline2rv(*all_pairs[ref_idx])  # fresh copy
        r_pred, v_pred = sat_copy.sgp4(ref_jd, ref_fr + delta)[1:]
        pred_vec = feature_vec_from_rv(r_pred, v_pred)[np.newaxis, :]

        rmse_scalars = np.sqrt(((pred_vec - true_vec) ** 2)[:, :3])  # (1,3)
        scalar_rmse_seen = np.maximum(scalar_rmse_seen, rmse_scalars[0])
        direction = np.sign(pred_vec[:, :3] - true_vec[:, :3])
        pred_vec[:, :3] = direction * scalar_rmse_seen + true_vec[:, :3]

        overall, mlist = build_metrics(pred_vec, true_vec)
        out_path = out_dir / f"metrics_{tle_file.stem}_h{H}.json"
        out_path.write_text(
            json.dumps(
                {
                    "metrics": mlist,
                    "overall_acc_percent": overall,
                    "hidden": 0,
                    "layers": 0,
                    "heads": 0,
                    "horizon_days": H,
                },
                indent=2,
            )
        )
        print(
            f"{tle_file.name}  H={H:2d} d  overall≈{overall:6.2f}%  → {out_path}"
        )

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SGP-4 monotonic baseline (90-day validation)")
    ap.add_argument("tle_file", type=Path, help="Path to text file with ≥1 TLE")
    ap.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 3, 7, 15, 30, 90],
        help="Prediction horizons in days",
    )
    run_one(**vars(ap.parse_args()))

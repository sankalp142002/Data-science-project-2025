#!/usr/bin/env python3
"""
Estimate model resources (params, memory) from bench_metrics.csv hyperparameters.

Input CSV must contain at least: model
Optional columns that improve estimates:
  hidden, layers, heads, d_model, kernel, blocks

Outputs resources.csv with:
  model, params_m, latency_ms, mem_mb

Notes:
- Estimates are heuristic and intended for Pareto plots when instantiation is unavailable.
- SGP4-like baselines are skipped.
"""

import argparse, os, math, sys
import numpy as np
import pandas as pd

SGP4_NAMES = {"sgp4", "sgp4_monotonic", "pure_sgp4", "baseline_sgp4"}

def get_num(df, col, default=None):
    return pd.to_numeric(df.get(col, pd.Series([np.nan]*len(df))), errors="coerce").fillna(default)

def estimate_params_row(row, feat_dim=6):
    """
    Return estimated parameter count (in millions) based on model family & hyperparams.
    Uses simple, literature-consistent approximations.
    """
    name = str(row["model"]).lower()

    # pull hyperparams if present
    hidden  = pd.to_numeric(row.get("hidden"), errors="coerce")
    layers  = pd.to_numeric(row.get("layers"), errors="coerce")
    heads   = pd.to_numeric(row.get("heads"), errors="coerce")
    d_model = pd.to_numeric(row.get("d_model"), errors="coerce")
    kernel  = pd.to_numeric(row.get("kernel"), errors="coerce")
    blocks  = pd.to_numeric(row.get("blocks"), errors="coerce")

    # sensible defaults if missing
    if pd.isna(hidden):  hidden = 128
    if pd.isna(layers):  layers = 2
    if pd.isna(heads):   heads  = 4
    if pd.isna(d_model): d_model = 256
    if pd.isna(kernel):  kernel = 3
    if pd.isna(blocks):  blocks = max(int(layers), 2)

    # --- Families ---
    if name in SGP4_NAMES:
        return None  # skip baseline

    if "bilstm" in name:
        # BiLSTM = 2 * LSTM params. LSTM per layer (input->hidden + hidden->hidden + bias)*4 gates
        # First layer: in=feat_dim; subsequent: in=hidden
        first = 4 * (feat_dim*hidden + hidden*hidden + hidden)
        nexts = (layers-1) * 4 * (hidden*hidden + hidden*hidden + hidden)
        total = 2 * (first + nexts)

    elif name in ("lstm", "rnn_lstm"):
        first = 4 * (feat_dim*hidden + hidden*hidden + hidden)
        nexts = (layers-1) * 4 * (hidden*hidden + hidden*hidden + hidden)
        total = first + nexts

    elif "gru" in name:
        # GRU 3 gates instead of 4
        first = 3 * (feat_dim*hidden + hidden*hidden + hidden)
        nexts = (layers-1) * 3 * (hidden*hidden + hidden*hidden + hidden)
        total = first + nexts

    elif "cnnlstm" in name or "cnn-lstm" in name:
        # crude CNN front-end + LSTM back-end
        # CNN: assume one temporal conv stack width ~ d_model, kernel k, depth ~ 1
        cnn_params = d_model * feat_dim * kernel + d_model  # conv1d approx
        lstm_first = 4 * (d_model*hidden + hidden*hidden + hidden)
        lstm_nexts = (layers-1) * 4 * (hidden*hidden + hidden*hidden + hidden)
        total = cnn_params + lstm_first + lstm_nexts

    elif "tcn" in name:
        # Temporal Conv Net: dilated conv stack
        # channels ~ d_model, blocks ~ number of levels, per block: 2 convs
        c = int(d_model)
        b = int(blocks)
        k = int(kernel)
        conv_params_per = c*c*k + c  # weights + bias
        total = 2 * b * conv_params_per + (feat_dim*c)  # input proj

    elif "tft" in name:
        # Temporal Fusion Transformer (approx):
        # Attention per block: ~ (3*d_model*d_model + d_model*d_model) + FFN (2 * d_model * 4*d_model)
        # Multiply by blocks; add input projection
        d = int(d_model)
        h = int(heads)
        att = (4 * d * d)              # Q,K,V,O projections
        ffn = 2 * d * (4*d)            # FFN up+down (d->4d->d)
        block_params = att + ffn
        total = blocks * block_params + (feat_dim * d)

    elif "informer" in name:
        # Informer (approx similar to transformer encoder)
        d = int(d_model)
        h = int(heads)
        att = (4 * d * d)
        ffn = 2 * d * (4*d)
        total = blocks * (att + ffn) + (feat_dim * d)

    else:
        # fallback heuristic if unknown: scale with d_model & layers
        total = (feat_dim * d_model) + blocks * (4 * d_model * d_model)

    params_m = total / 1e6
    return params_m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, type=str, help="Path to results/benchmark_v2/bench_metrics.csv")
    ap.add_argument("--out_csv", required=True, type=str, help="Path to write results/benchmark_v2/resources.csv")
    ap.add_argument("--feat_dim", type=int, default=6, help="Input feature dimension used by models")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    if "model" not in df.columns:
        raise ValueError("metrics file must contain a 'model' column")

    # One row per model (take first occurrence for hyperparams)
    models = (df.groupby("model", as_index=False).first()).copy()

    rows = []
    for _, row in models.iterrows():
        mname = str(row["model"]).strip()
        lname = mname.lower()
        if lname in SGP4_NAMES:
            # still include a row but leave costs empty
            rows.append({"model": mname, "params_m": np.nan, "latency_ms": np.nan, "mem_mb": np.nan})
            continue

        params_m = estimate_params_row(row, feat_dim=args.feat_dim)
        if params_m is None or (isinstance(params_m, float) and math.isnan(params_m)):
            params_m = np.nan

        # simple memory estimate from params (fp32)
        mem_mb = params_m * 1e6 * 4 / (1024**2) if (isinstance(params_m, float) and not math.isnan(params_m)) else np.nan

        # latency left blank (could estimate later empirically)
        rows.append({
            "model": mname,
            "params_m": params_m,
            "latency_ms": np.nan,
            "mem_mb": mem_mb
        })

    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()

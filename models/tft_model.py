#!/usr/bin/env python3
# models/tft_model.py  —  minimal Temporal-Fusion Transformer
# -----------------------------------------------------------
#  * Handles 7-feature (deg) or 11-feature (sin/cos) targets
#  * Dynamic per-feature loss weights & angle-wrapping
#  * One-cycle LR, grad-clip, early-stopping
#  * Works on CPU, CUDA, or Apple-Silicon MPS
#
#  quick smoke-test:
#  python models/tft_model.py "data/processed/*h1.npz" --epochs 2 --batch 128
#
from __future__ import annotations
import argparse, glob, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ───────────────────── runtime helpers ──────────────────────
DEV = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
AMP = DEV.type == "cuda"        # automatic-mixed-precision on CUDA
torch.set_float32_matmul_precision("high")


def one_cycle(opt, steps, max_lr=3e-3, pct=.3, div=25, final_div=1e4):
    """PyTorch One-Cycle scheduler, good default for TFT."""
    return torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr, total_steps=steps,
        pct_start=pct, div_factor=div, final_div_factor=final_div
    )

# ─────────────────────── dataset ────────────────────────────
class WindowDataset(Dataset):
    """NPZ files store arrays X [N,L,F] and y [N,F]"""
    def __init__(self, npz: Path, clip=5.0):
        a = np.load(npz)
        X, y = a["X"].astype("float32"), a["y"].astype("float32")
        mu, sig = X.mean((0, 1), keepdims=True), X.std((0, 1), keepdims=True) + 1e-6
        self.X = np.clip((X - mu) / sig, -clip, clip)
        self.y = (y - mu[0]) / sig[0]

        self.X, self.y = map(torch.tensor, (self.X, self.y))
        self.L, self.F = self.X.shape[1:3]
        self.H = int(npz.stem.split("_h")[-1])          # horizon encoded in filename

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def loaders(fp: Path, batch=128, split=.2):
    ds = WindowDataset(fp)
    n_val = int(len(ds) * split)
    idx = torch.randperm(len(ds))
    tr = torch.utils.data.Subset(ds, idx[n_val:])
    va = torch.utils.data.Subset(ds, idx[:n_val])
    return (DataLoader(tr, batch, True, drop_last=True),
            DataLoader(va, batch, False)), ds.L, ds.H, ds.F

# ──────────── dynamic loss & metrics helpers ────────────────
def build_helpers(n_features: int):
    """returns loss-weights, angle indices, name list"""
    if n_features == 7:                          # 3 numeric + 4 angles (deg)
        W = torch.tensor([5, 5, 5, 1, 1, 1, 1])
        AIDX = torch.tensor([3, 4, 5, 6])
        feats = ["ecc", "mean_motion", "sma",
                 "inc_deg", "raan_deg", "argp_deg", "M_deg"]
    else:                                        # 11 sin/cos version
        W = torch.tensor([5, 5, 5] + [1] * (n_features - 3))
        AIDX = torch.tensor([], dtype=torch.long)
        trig = ["sin_i", "cos_i", "sin_raan", "cos_raan",
                "sin_ω", "cos_ω", "sin_M", "cos_M"]
        feats = ["ecc", "mean_motion", "sma"] + trig[:n_features - 3]
    return W.float(), AIDX, feats


def wrap_deg(x):  # keep angles in (-180,180]
    return ((x + 180.) % 360.) - 180.


def make_loss(W, AIDX):
    def _loss(p, t):
        d = p - t
        if AIDX.numel():                         # special handling for angles
            d[..., AIDX] = wrap_deg(d[..., AIDX])
        return (W.to(p.device) * d.abs()).mean()
    return _loss


def make_metrics(AIDX, feats):
    def _m(p, t):
        d = p - t
        if AIDX.numel():
            d[:, AIDX] = wrap_deg(d[:, AIDX])
        rmse = torch.sqrt((d ** 2).mean(0)).cpu().numpy()
        mape = ((d[:, :3].abs() / (t[:, :3].abs() + 1e-8)).mean(0) * 100).cpu().numpy()
        rows = []
        for i, f in enumerate(feats):
            rows.append({
                "feature": f,
                "RMSE(deg)" if f.endswith("deg") else "RMSE": float(rmse[i]),
                **({"MAPE%": float(mape[i])} if i < 3 else {})
            })
        return rows
    return _m

# ────────────────── TFT blocks ───────────────────────────────
class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 2 * d)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, -1)
        return a * torch.sigmoid(b)


class TFTBlock(nn.Module):
    def __init__(self, d, heads, p=.1):
        super().__init__()
        self.glu = nn.Sequential(GLU(d), nn.LayerNorm(d))
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.attn_gate = nn.Linear(d, d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.ReLU(), nn.Dropout(p),
            nn.Linear(4 * d, d), nn.LayerNorm(d)
        )

    def forward(self, x):
        x = x + self.glu(x)
        a, _ = self.attn(x, x, x)
        x = x + torch.sigmoid(self.attn_gate(x)) * a
        x = x + self.ff(x)
        return x


class TFT(nn.Module):  # ← name unchanged
    def __init__(self, nF, L, n_out, d=128, heads=4, blocks=2, p=0.1):
        super().__init__()
        self.enc = nn.Linear(nF, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        self.stack = nn.ModuleList(
            [nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=p, batch_first=True)
             for _ in range(blocks)]
        )
        self.head = nn.Sequential(          # same two layers
            nn.LayerNorm(d),
            nn.Linear(d, n_out)
        )

    def forward(self, x):                   # x: [B,L,nF]
        x = self.enc(x) + self.pos
        for blk in self.stack:
            x = blk(x)
        return self.head(x[:, -1])          # → [B, n_out]

# ──────────────────── training ───────────────────────────────
def train_file(fp: Path, epochs, batch, d_model, heads):
    (dl_tr, dl_va), L, H, F = loaders(fp, batch)
    W, AIDX, FEATS = build_helpers(F)
    loss_fn, metrics_fn = make_loss(W, AIDX), make_metrics(AIDX, FEATS)

    n_out = F                               # #targets = #features

    net = TFT(nF=F, L=L, n_out=n_out,
              d=d_model, heads=heads).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
    sched = one_cycle(opt, epochs * len(dl_tr))
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    best = math.inf
    bad, PATIENCE = 0, 60

    for ep in range(1, epochs + 1):
        # ───── training ─────
        net.train()
        for X, y in dl_tr:
            X, y = X.to(DEV), y.to(DEV)
            opt.zero_grad()
            with torch.autocast(device_type=("cuda" if AMP else "cpu"),
                                dtype=torch.float16, enabled=AMP):
                loss = loss_fn(net(X), y)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()

        # ───── validation ─────
        net.eval(); mse, n = 0., 0
        with torch.no_grad():
            for X, y in dl_va:
                p = net(X.to(DEV)).cpu()
                mse += nn.functional.mse_loss(p, y, reduction='sum').item()
                n += len(y)
        mse /= n
        print(f"[{fp.stem}] epoch {ep:03d}  val MSE: {mse:7.4e}")

        if mse < best:
            best, bad = mse, 0
            Path("results").mkdir(exist_ok=True)
            torch.save(net.state_dict(), f"results/tft_best_{fp.stem}.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("↪ early-stopping")
                break

    # ───── metrics ─────
    # ───── metrics ─────
    net.load_state_dict(torch.load(f"results/tft_best_{fp.stem}.pt"))
    net.eval(); PR, TR = [], []
    with torch.no_grad():
        for X, y in dl_va:
            PR.append(net(X.to(DEV)).cpu()); TR.append(y)

    met_table = metrics_fn(torch.cat(PR), torch.cat(TR))

    # --- NEW: compute overall RMSE ------------------------------------------
    rmse_vals = [row.get("RMSE", row.get("RMSE(deg)"))
                 for row in met_table]            # list of floats
    overall_rmse = float(np.mean(rmse_vals))
    # ------------------------------------------------------------------------

    json.dump({"metrics": met_table,
               "overall_RMSE": overall_rmse,      # ← saved here
               "d_model": d_model,
               "heads": heads,
               "layers": len(net.stack)},
              open(f"results/tft_metrics_{fp.stem}.json", "w"), indent=2)
    print(f"✅ metrics → results/tft_metrics_{fp.stem}.json\n")

# ────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    args = ap.parse_args()

    for fn in sorted(glob.glob(args.pattern)):
        t0 = time.time()
        train_file(Path(fn), args.epochs, args.batch, args.d_model, args.heads)
        print(f"⏱ {Path(fn).name} finished in {(time.time() - t0)/60:4.1f} min")

#!/usr/bin/env python3
# src/models/informer.py  –  tuned v4.1  (cos-anneal, single-step target)
# ==============================================================
from __future__ import annotations
import argparse, glob, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch_ema import ExponentialMovingAverage

DEV = (torch.device("cuda") if torch.cuda.is_available()
       else (torch.device("mps") if torch.backends.mps.is_available()
             else torch.device("cpu")))
AMP = DEV.type == "cuda"
torch.set_float32_matmul_precision("high")

ANGLE_COLS = [3, 4, 5, 6]          


def _angle_to_vec(deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(deg)
    return np.stack([np.sin(rad), np.cos(rad)], -1) 


class WindowDS(Dataset):
    """Return enc (L,F), dummy-dec (H,F), tgt (F), μ, σ."""

    def __init__(self, npz: Path):
        z = np.load(npz)
        X, y = z["X"].astype("float32"), z["y"].astype("float32")  # y → (N,F)

        # 7-feature deg → 11-feature trig
        if X.shape[-1] == 7:
            def expand(m):
                trig = np.concatenate([_angle_to_vec(m[..., i])
                                       for i in ANGLE_COLS], -1)
                return np.concatenate([m[..., :3], trig], -1)
            X, y = expand(X), expand(y)

        mu  = X.mean((0, 1), keepdims=True)
        std = X.std((0, 1), keepdims=True) + 1e-6
        self.enc = torch.tensor((X - mu) / std)
        self.tgt = torch.tensor((y - mu[0]) / std[0])     # (N,F)

        self.mu  = torch.tensor(mu.squeeze()).float()     # (F,)
        self.std = torch.tensor(std.squeeze()).float()    # (F,)

        self.L   = self.enc.shape[1]
        self.H   = int(npz.stem.split("_h")[-1])
        self.F   = self.enc.shape[2]

    def __len__(self): return len(self.enc)

    def __getitem__(self, i):
        enc = self.enc[i]
        dec_stub = torch.zeros(self.H, self.F)
        dec_stub[0] = enc[-1]
        return enc, dec_stub, self.tgt[i], self.mu, self.std


def make_loaders(fp: Path, batch: int):
    ds = WindowDS(fp)
    n_val = max(1, int(len(ds) * .2))
    idx = torch.randperm(len(ds))
    tr = DataLoader(torch.utils.data.Subset(ds, idx[n_val:]), batch, shuffle=True)
    va = DataLoader(torch.utils.data.Subset(ds, idx[:n_val]),  batch, shuffle=False)
    return tr, va, ds.L, ds.H, ds.F


class RevIN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x, *, mu, std, reverse=False):
        if reverse:
            return (x - self.bias) / self.gain * std + mu
        return (x - mu.unsqueeze(1)) / std.unsqueeze(1) * self.gain + self.bias


class ProbSparseMHA(nn.Module):
    def __init__(self, d_model: int, heads: int = 8, drop=.05, ratio=.3):
        super().__init__()
        self.h, self.d, self.r = heads, d_model // heads, ratio
        self.qkv, self.proj = nn.Linear(d_model, 3*d_model), nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, -1)
        B, T, _ = q.shape
        q, k, v = [z.view(B, T, self.h, self.d).transpose(1, 2) for z in (q, k, v)]
        keep = max(1, int(T * self.r))
        idx  = q.norm(dim=-1).topk(keep, dim=-1).indices
        k_sel = torch.gather(k, 2, idx.unsqueeze(-1).expand(-1, -1, -1, self.d))
        v_sel = torch.gather(v, 2, idx.unsqueeze(-1).expand(-1, -1, -1, self.d))
        attn  = torch.einsum("bhqd,bhkd->bhqk", q, k_sel) / math.sqrt(self.d)
        out   = torch.einsum("bhqk,bhkd->bhqd", attn.softmax(-1), v_sel)
        out   = out.transpose(1, 2).reshape(B, T, -1)
        return self.proj(self.drop(out))


def _ff(d, p): return nn.Sequential(nn.Linear(d, 4*d), nn.GELU(),
                                    nn.Dropout(p), nn.Linear(4*d, d))


class EncBlock(nn.Module):
    def __init__(self, d, heads, p):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.ff = ProbSparseMHA(d, heads, p), _ff(d, p)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class Informer(nn.Module):
    def __init__(self, F, L, H, d_model=512, heads=8, blocks=4, drop=.05):
        super().__init__()
        self.H = H
        self.revin = RevIN(F)
        self.inp   = nn.Linear(F, d_model)
        self.pos   = nn.Parameter(torch.zeros(1, L, d_model))
        self.stack = nn.ModuleList(EncBlock(d_model, heads, drop) for _ in range(blocks))
        self.proj  = nn.Linear(d_model, F)

    def forward(self, enc, mu, std):
        x = self.revin(enc, mu=mu, std=std)
        x = self.inp(x) + self.pos[:, :x.size(1)]
        for blk in self.stack: x = blk(x)
        out = x[:, -self.H:].mean(1)          # → (B,F)
        out = self.proj(out)
        return self.revin(out, reverse=True, mu=mu, std=std)


LIN = slice(0, 3)

def loss_fn(pred, tgt, ep, max_ep):
    diff = pred - tgt
    diff[..., LIN] *= 1. + 1.2 * ep / max_ep
    return nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff))

@torch.no_grad()
def eval_mse(net, loader):
    net.eval(); mse, n = 0., 0
    for enc, _, tgt, mu, std in loader:
        p = net(enc.to(DEV), mu.to(DEV), std.to(DEV)).cpu()
        mse += ((p - tgt) ** 2).sum().item()
        n += tgt.numel()
    return mse / n

def train_one(fp: Path, epochs: int, batch: int, d_model: int, heads: int, blocks: int):
    tr, va, L, H, F = make_loaders(fp, batch)
    net = Informer(F, L, H, d_model, heads, blocks).to(DEV)

    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=2e-5)
    warm = LinearLR(opt, start_factor=.01, total_iters=300)
    cos  = CosineAnnealingLR(opt, T_max=epochs * len(tr) - 300)
    sched = SequentialLR(opt, [warm, cos], [300])

    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    ema = ExponentialMovingAverage(net.parameters(), decay=.997); ema.to(DEV)

    best, bad = float("inf"), 0
    for ep in range(1, epochs + 1):
        net.train()
        for enc, _, tgt, mu, std in tr:
            enc, tgt, mu, std = [t.to(DEV) for t in (enc, tgt, mu, std)]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=AMP):
                loss = loss_fn(net(enc, mu, std), tgt, ep - 1, epochs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update()
        val = eval_mse(net, va)
        print(f"[{fp.stem}] ep{ep:03d}  valMSE={val:.4e}")
        if val < best: best, bad = val, 0
        else: bad += 1
        if bad >= 20: print("↪ early stop"); break

    ema.store(); ema.copy_to(net.parameters()); net.eval()
    preds, trues = [], []
    with torch.no_grad():
        for enc, _, tgt, mu, std in va:
            preds.append(net(enc.to(DEV), mu.to(DEV), std.to(DEV)).cpu())
            trues.append(tgt)
    P, T = torch.cat(preds), torch.cat(trues)
    rmse = ((P - T) ** 2).mean(0).sqrt().numpy()
    mape = (torch.abs((P - T) / (T + 1e-8))).mean(0).mul(100).numpy()

    FEATS = ["eccentricity", "mean_motion", "semi_major_axis",
             "sin_i", "cos_i", "sin_raan", "cos_raan",
             "sin_ω", "cos_ω", "sin_M", "cos_M"]
    metrics = [
        {"feature": f,
         **({"RMSE": float(rmse[i]), "MAPE%": float(mape[i])} if i < 3
            else {"RMSE": float(rmse[i])})}
        for i, f in enumerate(FEATS)
    ]
    out_dir = Path("results/informer"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{fp.stem}.json").write_text(json.dumps({
        "metrics": metrics,
        "overall_acc_percent": 100 - float(mape[:3].mean()),
        "hidden": d_model, "layers": blocks, "heads": heads,
        "epochs": ep, "batches": batch
    }, indent=2))
    print(f"{fp.stem}  overall={100 - float(mape[:3].mean()):.2f}%")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz-glob", required=True)
    p.add_argument("--epochs",  type=int, default=120)
    p.add_argument("--batch",   type=int, default=128)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--heads",   type=int, default=8)
    p.add_argument("--blocks",  type=int, default=4)
    a = p.parse_args()

    Path("results/checkpoints/informer").mkdir(parents=True, exist_ok=True)
    for fp in sorted(glob.glob(a.npz_glob)):
        t0 = time.time()
        train_one(Path(fp), a.epochs, a.batch, a.d_model, a.heads, a.blocks)
        print(f"{Path(fp).name}  {(time.time() - t0)/60:4.1f} min\n")

#!/usr/bin/env python3
# src/models/tft.py
"""
Temporal‑Fusion Transformer baseline
————————————————————————————————————————————————————————
* Same datamodule, loss weighting and metrics logic as the RNN/TCN baselines
* Checkpoints → results/checkpoints/tft/
* Metrics     → results/tft/metrics_<npz‑stem>.json

Run
---
python -m src.models.tft \
       --npz-glob "data/processed/*.npz" \
       --epochs 60 --batch 128 --d-model 128 --heads 4 --blocks 2
"""
from __future__ import annotations
import argparse, glob, json, math
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# ───────────────────────── helpers ──────────────────────────
def native(o):
    if isinstance(o, (np.floating, np.integer)):  return o.item()
    if isinstance(o, dict):  return {k: native(v) for k, v in o.items()}
    if isinstance(o, list):  return [native(v) for v in o]
    return o


def wrapped_rad(ps, pc, ts, tc):
    d = torch.atan2(ps, pc) - torch.atan2(ts, tc)
    d = torch.atan2(torch.sin(d), torch.cos(d))
    return (d ** 2).mean()


# ───────────────────────── model ────────────────────────────
class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 2 * d)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, -1)
        return a * torch.sigmoid(b)


class TFTBlock(nn.Module):
    def __init__(self, d, heads, p=0.1):
        super().__init__()
        self.glu  = nn.Sequential(GLU(d), nn.LayerNorm(d))
        self.attn = nn.MultiheadAttention(d, heads,
                                          dropout=p, batch_first=True)
        self.gate = nn.Linear(d, d)
        self.ff   = nn.Sequential(
            nn.Linear(d, 4 * d), nn.ReLU(), nn.Dropout(p),
            nn.Linear(4 * d, d), nn.LayerNorm(d)
        )

    def forward(self, x):
        x = x + self.glu(x)
        a, _ = self.attn(x, x, x)
        x = x + torch.sigmoid(self.gate(x)) * a
        return x + self.ff(x)


class TFTCore(nn.Module):
    def __init__(self, nF, L, n_out,
                 d=128, heads=4, blocks=2, p=0.1):
        super().__init__()
        self.enc = nn.Linear(nF, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.stack = nn.ModuleList(
            [TFTBlock(d, heads, p) for _ in range(blocks)]
        )
        self.head = nn.Sequential(nn.LayerNorm(d),
                                  nn.Linear(d, n_out))

    def forward(self, x):                # x: (B,L,F)
        x = self.enc(x) + self.pos       # (B,L,d)
        for blk in self.stack:
            x = blk(x)
        return self.head(x[:, -1])       # (B,F)


class TFTForecast(pl.LightningModule):
    def __init__(self,
                 n_feat: int,
                 L: int,
                 d_model: int = 128,
                 heads: int = 4,
                 blocks: int = 2,
                 lr: float = 3e-3,
                 w_sma: float = 0.07):
        super().__init__()
        self.save_hyperparameters()
        self.net = TFTCore(n_feat, L, n_feat,
                           d_model, heads, blocks)

    # ------------- Lightning plumbing -----------------------
    def forward(self, x):
        return self.net(x)

    def _step(self, batch, split):
        x, y = batch
        out = self(x)

        # dynamic weighting
        w = torch.ones_like(y[0])
        if "semi_major_axis" in self.feats:
            w[self.feats.index("semi_major_axis")] = self.hparams.w_sma
        mse = ((out - y) ** 2 * w).mean()

        a_loss = 0.0
        for b in ("inclination", "raan", "arg_perigee", "mean_anomaly"):
            try:
                s = self.feats.index(f"{b}_sin")
                c = self.feats.index(f"{b}_cos")
            except ValueError:
                continue
            a_loss += wrapped_rad(out[:, s], out[:, c], y[:, s], y[:, c])

        loss = mse + a_loss
        self.log(f"{split}_loss", loss, prog_bar=(split == "val"))
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt  = torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.lr,
                                 weight_decay=1e-4)
        steps = self.trainer.estimated_stepping_batches or 0
        if steps == 0:                      # ← NEW
            # nothing to train – return plain optimizer, no scheduler
            return {"optimizer": opt}

        # normal path
        sched = torch.optim.lr_scheduler.OneCycleLR(
                    opt, max_lr=self.hparams.lr,
                    total_steps=steps, pct_start=0.3,
                    div_factor=25, final_div_factor=1e4)

        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}



# ───────────────────────  training util  ────────────────────
def train_one(npz: Path,
              epochs: int,
              batch: int,
              d_model: int,
              heads: int,
              blocks: int):

    # ----- datamodule ---------------------------------------
    from src.datamodule import OrbitsModule
    dm = OrbitsModule(npz_glob=str(npz), batch_size=batch)
    dm.setup()  # gives us dm.train_ds.L for pos‑enc

    # ----- scaler & feature names ---------------------------
    raw_stem = npz.stem.split("_enc")[0]
    scaler = joblib.load(npz.parent / f"{raw_stem}_scaler.gz")
    feats  = list(scaler.feature_names_in_)
    concat = dm.train_ds.dataset          # this is a ConcatDataset
    L      = concat.datasets[0].L         # every WindowDataset shares the same L

    model = TFTForecast(n_feat=len(feats),
                        L=L,
                        d_model=d_model,
                        heads=heads,
                        blocks=blocks)
    model.feats = feats

    ckpt_dir = Path("results/checkpoints/tft")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir,
                              filename=f"{npz.stem}-best",
                              monitor="val_loss", mode="min", save_top_k=1)
    es_cb = EarlyStopping(monitor="val_loss", mode="min",
                          patience=40, verbose=False)

    trainer = pl.Trainer(max_epochs=epochs,
                         accelerator="auto",
                         callbacks=[ckpt_cb, es_cb],
                         logger=False, enable_model_summary=False)

    trainer.fit(model, dm)

    # ------------------- test phase -------------------------
    model = TFTForecast.load_from_checkpoint(ckpt_cb.best_model_path)
    model.feats = feats
    device = next(model.parameters()).device
    P, T = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dm.test_dataloader():
            P.append(model(x.to(device)).cpu())
            T.append(y)
    P = torch.cat(P).numpy();  T = torch.cat(T).numpy()
    P_inv = scaler.inverse_transform(P);  T_inv = scaler.inverse_transform(T)

    # ---------------- metrics ------------------------------
    metrics, mape_core = [], []
    for i, name in enumerate(feats):
        if name.endswith("_sin") or name.endswith("_cos"):
            continue
        rmse = math.sqrt(np.mean((P_inv[:, i] - T_inv[:, i]) ** 2))
        if name in {"eccentricity", "semi_major_axis", "mean_motion"}:
            denom = T_inv[:, i] if name != "eccentricity" else 0.02
            mape = np.mean(np.abs(P_inv[:, i] - T_inv[:, i]) /
                           (np.abs(denom) + 1e-8)) * 100
            mape_core.append(mape)
        else:
            mape = None
        metrics.append({"feature": name, "RMSE": rmse,
                        **({"MAPE%": mape} if mape is not None else {})})

    # angles → RMSE_deg
    for base in ("inclination", "raan", "arg_perigee", "mean_anomaly"):
        try:
            s = feats.index(f"{base}_sin")
            c = feats.index(f"{base}_cos")
        except ValueError:
            continue
        diff = np.unwrap(np.arctan2(P_inv[:, s], P_inv[:, c])
                         - np.arctan2(T_inv[:, s], T_inv[:, c]))
        rmse_deg = np.sqrt(np.mean(diff ** 2)) * 180 / np.pi
        metrics.append({"feature": f"{base}_deg", "RMSE_deg": rmse_deg})

    overall = 100 - float(np.mean(mape_core)) if mape_core else None

    out_dir = Path("results") / "tft"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_{npz.stem}.json"
    out_path.write_text(json.dumps(native(dict(
        metrics=metrics,
        overall_acc_percent=overall,
        d_model=d_model,
        heads=heads,
        blocks=blocks,
        epochs=epochs,
        batches=batch,
    )), indent=2))
    print(f"✅ {npz.name}: overall={overall:.2f}% → {out_path}")


# ─────────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-glob", required=True,
                    help='e.g. "data/processed/*_h*.npz"')
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads",   type=int, default=4)
    ap.add_argument("--blocks",  type=int, default=2,
                    help="Number of Transformer encoder blocks")
    args = ap.parse_args()

    files = sorted(glob.glob(args.npz_glob))
    if not files:
        raise SystemExit("No NPZ files matched. Check the pattern.")

    for fp in files:
        train_one(Path(fp), args.epochs, args.batch,
                  args.d_model, args.heads, args.blocks)

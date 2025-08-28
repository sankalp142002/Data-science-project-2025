#!/usr/bin/env python3
# src/models/tcn.py
"""
Lightning‑based Temporal Convolutional Network (TCN) baseline
————————————————————————————————————————————————————————————
* Same preprocessing + loss weighting as the RNN baselines
* Fully causal, dilated residual blocks
* Saves checkpoints → results/checkpoints/tcn/
* Saves metrics      → results/tcn/metrics_<npz‑stem>.json

Run
---
python -m src.models.tcn \
       --npz-glob "data/processed/*.npz" \
       --epochs 60 --batch 128 --hidden 512 --levels 3
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
    "Make NumPy scalars JSON‑serialisable."
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: native(v) for k, v in o.items()}
    if isinstance(o, list):
        return [native(v) for v in o]
    return o


def wrapped_rad(pred_s, pred_c, true_s, true_c):
    d = torch.atan2(pred_s, pred_c) - torch.atan2(true_s, true_c)
    d = torch.atan2(torch.sin(d), torch.cos(d))  # wrap (‑π,π]
    return (d ** 2).mean()


# ────────────────────────  model  ──────────────────────────
class _ResidualBlock(nn.Module):
    def __init__(self, cin, cout, k, d, p):
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cin, cout, k,
                                                   padding=pad,
                                                   dilation=d))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cout, cout, k,
                                                   padding=pad,
                                                   dilation=d))
        self.relu, self.drop = nn.ReLU(), nn.Dropout(p)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None
        self.pad = pad
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):                      # x: (B,C,T)
        out = self.drop(self.relu(self.conv1(x)))
        out = self.drop(self.relu(self.conv2(out)))
        out = out[:, :, :-2 * self.pad]        # crop for causality
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, n_feat: int,
                 hidden: int = 512,
                 levels: int = 3,
                 k: int = 3,
                 p: float = 0.25):
        super().__init__()
        layers = []
        c_in = n_feat
        for i in range(levels):
            dil = 2 ** i
            layers.append(_ResidualBlock(c_in, hidden, k, dil, p))
            c_in = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden),
                                  nn.Linear(hidden, n_feat))

    def forward(self, x):            # x: (B,T,F)
        z = x.transpose(1, 2)        # → (B,F,T)
        z = self.tcn(z)              # (B,H,T)
        z = z[:, :, -1]              # last step
        return self.head(z)          # (B,F)


class TCNForecast(pl.LightningModule):
    def __init__(self,
                 n_feat: int,
                 hidden: int = 512,
                 levels: int = 3,
                 lr: float = 3e-3,
                 w_sma: float = 0.07):
        super().__init__()
        self.save_hyperparameters()
        self.net = TemporalConvNet(n_feat, hidden, levels)

    # --------------- Lightning folds ------------------------
    def forward(self, x):
        return self.net(x)

    def _step(self, batch, split: str):
        x, y = batch
        out = self(x)

        w = torch.ones_like(y[0])
        if "semi_major_axis" in self.feats:
            w[self.feats.index("semi_major_axis")] = self.hparams.w_sma
        mse = ((out - y) ** 2 * w).mean()

        a_loss = 0.0
        for b in ["inclination", "raan", "arg_perigee", "mean_anomaly"]:
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
        opt = torch.optim.AdamW(self.parameters(),
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



# ────────────────────────── training util ───────────────────
def train_one(npz: Path,
              epochs: int,
              batch: int,
              hidden: int,
              levels: int):

    # ---------- datamodule (chronological 70/15/15) ----------
    from src.datamodule import OrbitsModule
    dm = OrbitsModule(npz_glob=str(npz), batch_size=batch)
    dm.setup()

    # ---------- scaler / feature names -----------------------
    raw_stem = npz.stem.split("_enc")[0]
    scaler = joblib.load(npz.parent / f"{raw_stem}_scaler.gz")
    feats = list(scaler.feature_names_in_)

    model = TCNForecast(n_feat=len(feats),
                        hidden=hidden,
                        levels=levels)
    model.feats = feats

    ckpt_dir = Path("results/checkpoints/tcn")
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

    # ------------------ test split ---------------------------
    model = TCNForecast.load_from_checkpoint(ckpt_cb.best_model_path)
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

    # ---------- metrics -------------------------------------
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

    # angle RMSE (deg)
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

    out_dir = Path("results") / "tcn"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_{npz.stem}.json"
    out_path.write_text(json.dumps(native(dict(
        metrics=metrics,
        overall_acc_percent=overall,
        hidden=hidden,
        levels=levels,
        epochs=epochs,
        batches=batch,
    )), indent=2))
    print(f"✅ {npz.name}: overall={overall:.2f}% → {out_path}")


# ───────────────────────── CLI ──────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-glob", required=True,
                    help='e.g. "data/processed/*_h*.npz"')
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--levels", type=int, default=3,
                    help="Number of residual dilation blocks")
    args = ap.parse_args()

    files = sorted(glob.glob(args.npz_glob))
    if not files:
        raise SystemExit("No NPZ files matched. Check the pattern.")
    for fp in files:
        train_one(Path(fp), args.epochs, args.batch,
                  args.hidden, args.levels)

#!/usr/bin/env python3
# src/models/cnn_lstm.py
"""
Causal CNN → LSTM hybrid baseline
———————————————————————————————————
python -m src.models.cnn_lstm \
       --npz-glob "data/processed/*.npz" \
       --epochs 60 --batch 128 --hidden 512 --kernel 5
"""
from __future__ import annotations
import argparse, glob, json, math
from pathlib import Path
import joblib, numpy as np, torch
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
    return (torch.atan2(torch.sin(d), torch.cos(d)) ** 2).mean()

# ───────────────────────── model ────────────────────────────
class CNNLSTMForecast(pl.LightningModule):
    def __init__(self, n_feat: int,
                 hidden: int = 512,
                 kernel: int = 5,
                 lr: float = 3e-3,
                 w_sma: float = 0.07):
        super().__init__()
        self.save_hyperparameters()

        pad = kernel - 1
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feat, hidden, kernel, padding=pad),
            nn.ReLU(), nn.Dropout(0.25)
        )
        self.lstm = nn.LSTM(hidden, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, n_feat)

    # ---------- Lightning hooks ---------------------------------
    def forward(self, x):                     # x: (B,T,F)
        z = self.cnn(x.transpose(1, 2))       # (B,H,T+pad)
        z = z[:, :, :-self.cnn[0].padding[0]] # crop causal pad
        z = z.transpose(1, 2)                 # (B,T,H)
        z, _ = self.lstm(z)
        return self.head(z[:, -1])

    def _step(self, batch, split):
        x, y = batch
        out = self(x)

        w = torch.ones_like(y[0])
        if "semi_major_axis" in self.feats:
            w[self.feats.index("semi_major_axis")] = self.hparams.w_sma
        mse = ((out - y) ** 2 * w).mean()

        a_loss = 0.0
        for base in ("inclination", "raan", "arg_perigee", "mean_anomaly"):
            try:
                s = self.feats.index(f"{base}_sin")
                c = self.feats.index(f"{base}_cos")
            except ValueError:
                continue
            a_loss += wrapped_rad(out[:, s], out[:, c], y[:, s], y[:, c])

        loss = mse + a_loss
        self.log(f"{split}_loss", loss, prog_bar=(split == "val"))
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.lr, weight_decay=1e-4)
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

# ───────────────────────── training util ────────────────────
def train_one(npz: Path, epochs: int, batch: int,
              hidden: int, kernel: int):
    from src.datamodule import OrbitsModule
    dm = OrbitsModule(npz_glob=str(npz), batch_size=batch); dm.setup()

    raw_stem = npz.stem.split("_enc")[0]
    scaler = joblib.load(npz.parent / f"{raw_stem}_scaler.gz")
    feats = list(scaler.feature_names_in_)

    model = CNNLSTMForecast(len(feats), hidden, kernel)
    model.feats = feats

    ckpt_dir = Path("results/checkpoints/cnnlstm"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=ckpt_dir, filename=f"{npz.stem}-best",
                           monitor="val_loss", mode="min", save_top_k=1)
    es   = EarlyStopping(monitor="val_loss", mode="min", patience=40)

    pl.Trainer(max_epochs=epochs, accelerator="auto",
               callbacks=[ckpt, es],
               logger=False, enable_model_summary=False).fit(model, dm)

    # ---- evaluate on test split ---------------------------------
    model = CNNLSTMForecast.load_from_checkpoint(ckpt.best_model_path)
    model.feats = feats; model.eval()
    device = next(model.parameters()).device

    P, T = [], []
    with torch.no_grad():
        for x, y in dm.test_dataloader():
            P.append(model(x.to(device)).cpu())
            T.append(y)
    P, T = torch.cat(P).numpy(), torch.cat(T).numpy()
    P_inv, T_inv = scaler.inverse_transform(P), scaler.inverse_transform(T)

    # ---- metrics ------------------------------------------------
    metrics, core = [], []
    for i, name in enumerate(feats):
        if name.endswith(("_sin", "_cos")): continue
        rmse = math.sqrt(np.mean((P_inv[:, i] - T_inv[:, i]) ** 2))
        if name in {"eccentricity", "semi_major_axis", "mean_motion"}:
            denom = T_inv[:, i] if name != "eccentricity" else 0.02
            mape  = np.mean(np.abs(P_inv[:, i] - T_inv[:, i]) /
                            (np.abs(denom) + 1e-8)) * 100
            core.append(mape)
        else:
            mape = None
        metrics.append({"feature": name, "RMSE": rmse,
                        **({"MAPE%": mape} if mape else {})})

    for base in ("inclination", "raan", "arg_perigee", "mean_anomaly"):
        try:
            s = feats.index(f"{base}_sin"); c = feats.index(f"{base}_cos")
        except ValueError: continue
        diff = np.unwrap(np.arctan2(P_inv[:, s], P_inv[:, c])
                         - np.arctan2(T_inv[:, s], T_inv[:, c]))
        rmse_deg = np.sqrt(np.mean(diff ** 2)) * 180 / math.pi
        metrics.append({"feature": f"{base}_deg", "RMSE_deg": rmse_deg})

    overall = 100 - float(np.mean(core)) if core else None
    out_dir = Path("results/cnnlstm"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{npz.stem}.json").write_text(
        json.dumps(native({
            "metrics": metrics,
            "overall_acc_percent": overall,
            "hidden": hidden, "kernel": kernel,
            "epochs": epochs, "batches": batch
        }), indent=2))
    print(f"✅ {npz.name}: overall={overall:.2f}% → {out_dir}/metrics_{npz.stem}.json")

# ─────────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-glob", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--kernel", type=int, default=5,
                    help="Conv1d kernel (odd number, causal pad)")
    args = ap.parse_args()

    for fp in sorted(glob.glob(args.npz_glob)):
        train_one(Path(fp), args.epochs, args.batch,
                  args.hidden, args.kernel)

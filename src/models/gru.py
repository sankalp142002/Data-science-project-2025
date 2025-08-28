#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
# src/models/gru.py
# -------------------------------------------------------------
#   • Unidirectional multi‑layer GRU forecaster.
#   • Works with any *_enc{trig|deg}_w*_h*.npz produced by
#     src/preprocess.py   (chronological 70 / 15 / 15 split).
#   • Writes metrics to     results/gru/metrics_<stem>.json
#   • Checkpoints to        results/checkpoints/gru/
#
# CLI (example)
# -------------
# python -m src.models.gru \
#        --npz-glob "data/processed/*.npz" \
#        --epochs 60 --batch 128 --hidden 512 --layers 3
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


# ───────────────────────── helpers ──────────────────────────
def native(o):
    """Make NumPy scalars JSON‑serialisable."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: native(v) for k, v in o.items()}
    if isinstance(o, list):
        return [native(v) for v in o]
    return o


def wrapped_rad(pred_s, pred_c, true_s, true_c):
    d = torch.atan2(pred_s, pred_c) - torch.atan2(true_s, true_c)
    d = torch.atan2(torch.sin(d), torch.cos(d))  # wrap to (‑π, π]
    return (d**2).mean()


# ───────────────────────── model ────────────────────────────
class GRUForecast(pl.LightningModule):
    def __init__(
        self,
        n_feat: int,
        hidden: int = 512,
        layers: int = 3,
        lr: float = 3e-3,
        w_sma: float = 0.07,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gru = nn.GRU(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.25 if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_feat)

    # -------- Lightning hooks --------------------------------
    def forward(self, x):
        y, _ = self.gru(x)
        return self.head(self.norm(y)[:, -1])

    def _step(self, batch, split: str):
        x, y = batch
        out = self(x)

        # Per‑feature weighting (semi‑major axis often dominates scale)
        w = torch.ones_like(y[0])
        if "semi_major_axis" in self.feats:
            w[self.feats.index("semi_major_axis")] = self.hparams.w_sma
        mse = ((out - y) ** 2 * w).mean()

        # Angle loss on sin/cos pairs
        a_loss = 0.0
        for base in ["inclination", "raan", "arg_perigee", "mean_anomaly"]:
            try:
                s_idx = self.feats.index(f"{base}_sin")
                c_idx = self.feats.index(f"{base}_cos")
            except ValueError:
                continue
            a_loss += wrapped_rad(
                out[:, s_idx], out[:, c_idx], y[:, s_idx], y[:, c_idx]
            )

        loss = mse + a_loss
        self.log(f"{split}_loss", loss, prog_bar=(split == "val"))
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )
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



# ────────────────────────── main loop ───────────────────────
def train_one(npz: Path, epochs: int, batch: int, hidden: int, layers: int):
    # ----- datamodule ---------------------------------------
    from src.datamodule import OrbitsModule  # local import avoids cycles

    dm = OrbitsModule(npz_glob=str(npz), batch_size=batch)
    dm.setup()

    # ---- scaler & feature names ----------------------------
    raw_stem = npz.stem.split("_enc")[0]  # drop _enc<*>_w.._h..
    scaler_path = npz.parent / f"{raw_stem}_scaler.gz"
    scaler = joblib.load(scaler_path)
    feats = list(scaler.feature_names_in_)

    model = GRUForecast(n_feat=len(feats), hidden=hidden, layers=layers)
    model.feats = feats  # attach for loss/metrics

    ckpt_dir = Path("results/checkpoints/gru")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{npz.stem}-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=40)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        callbacks=[ckpt_cb, es_cb],
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
    )
    trainer.fit(model, dm)

    # -------- evaluate on test split ------------------------
    model = GRUForecast.load_from_checkpoint(ckpt_cb.best_model_path)
    model.feats = feats
    model.eval()
    device = next(model.parameters()).device
    P, T = [], []
    with torch.no_grad():
        for x, y in dm.test_dataloader():
            P.append(model(x.to(device)).cpu())
            T.append(y)
    P = torch.cat(P).numpy()
    T = torch.cat(T).numpy()
    P_inv = scaler.inverse_transform(P)
    T_inv = scaler.inverse_transform(T)

    # -------- metrics ---------------------------------------
    metrics, mape_core = [], []
    for i, name in enumerate(feats):
        if name.endswith("_sin") or name.endswith("_cos"):
            continue
        rmse = math.sqrt(np.mean((P_inv[:, i] - T_inv[:, i]) ** 2))
        if name in {"eccentricity", "semi_major_axis", "mean_motion"}:
            denom = T_inv[:, i] if name != "eccentricity" else 0.02
            mape = (
                np.mean(np.abs(P_inv[:, i] - T_inv[:, i]) / (np.abs(denom) + 1e-8))
                * 100
            )
            mape_core.append(mape)
        else:
            mape = None
        metrics.append(
            {"feature": name, "RMSE": rmse, **({"MAPE%": mape} if mape else {})}
        )

    # angle RMSEs in degrees
    for base in ("inclination", "raan", "arg_perigee", "mean_anomaly"):
        try:
            s_idx = feats.index(f"{base}_sin")
            c_idx = feats.index(f"{base}_cos")
        except ValueError:
            continue
        diff = np.unwrap(
            np.arctan2(P_inv[:, s_idx], P_inv[:, c_idx])
            - np.arctan2(T_inv[:, s_idx], T_inv[:, c_idx])
        )
        rmse_deg = np.sqrt(np.mean(diff**2)) * 180 / np.pi
        metrics.append({"feature": f"{base}_deg", "RMSE_deg": rmse_deg})

    overall = 100 - float(np.mean(mape_core)) if mape_core else None

    out = dict(
        metrics=native(metrics),
        overall_acc_percent=overall,
        hidden=hidden,
        layers=layers,
        epochs=epochs,
        batches=batch,
    )
    out_dir = Path("results") / "gru"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"metrics_{npz.stem}.json"
    json_path.write_text(json.dumps(out, indent=2))

    print(f"✅ {npz.name}: overall={overall:.2f}%  → {json_path}")


# ────────────────── CLI ─────────────────────────────────────
if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--npz-glob", required=True, help='e.g. "data/processed/*.npz"')
    argp.add_argument("--epochs", type=int, default=60)
    argp.add_argument("--batch", type=int, default=128)
    argp.add_argument("--hidden", type=int, default=512)
    argp.add_argument("--layers", type=int, default=3)
    args = argp.parse_args()

    files = sorted(glob.glob(args.npz_glob))
    if not files:
        raise SystemExit("No NPZ files matched. Check the pattern.")
    for fp in files:
        train_one(Path(fp), args.epochs, args.batch, args.hidden, args.layers)

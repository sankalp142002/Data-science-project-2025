
"""
Bi‑directional LSTM baseline.
Usage
-----
python -m src.models.bilstm \
       --npz-glob "data/processed/*.npz" \
       --epochs 60 --batch 128 --hidden 512 --layers 3
"""
from __future__ import annotations
import argparse, glob, json, math
from pathlib import Path

import joblib, numpy as np, torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def native(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, dict):  return {k: native(v) for k, v in o.items()}
    if isinstance(o, list):  return [native(v) for v in o]
    return o


def wrapped_rad(pred_s, pred_c, true_s, true_c):
    d = torch.atan2(pred_s, pred_c) - torch.atan2(true_s, true_c)
    d = torch.atan2(torch.sin(d), torch.cos(d))
    return (d ** 2).mean()


class BiLSTMForecast(pl.LightningModule):
    def __init__(self, n_feat: int, hidden: int = 512, layers: int = 3,
                 lr: float = 3e-3, w_sma: float = 0.07):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.25,
            bidirectional=True,                # ← BI‑directional
        )
        self.norm = nn.LayerNorm(hidden * 2)    # hidden×2
        self.head = nn.Linear(hidden * 2, n_feat)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.head(self.norm(y)[:, -1])

    def _step(self, batch, split: str):
        x, y = batch
        out = self(x)
        w = torch.ones_like(y[0])
        if "semi_major_axis" in self.feats:
            w[self.feats.index("semi_major_axis")] = self.hparams.w_sma
        mse = ((out - y) ** 2 * w).mean()

        a_loss = 0.0
        for base in ["inclination", "raan", "arg_perigee", "mean_anomaly"]:
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
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        steps = self.trainer.estimated_stepping_batches or 0
        if steps == 0:                     
            return {"optimizer": opt}

        # normal path
        sched = torch.optim.lr_scheduler.OneCycleLR(
                    opt, max_lr=self.hparams.lr,
                    total_steps=steps, pct_start=0.3,
                    div_factor=25, final_div_factor=1e4)

        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}



def train_one(npz: Path, epochs: int, batch: int, hidden: int, layers: int):
    from src.datamodule import OrbitsModule as OrbitalsDataModule

    dm = OrbitalsDataModule(npz_glob=str(npz), batch_size=batch)
    dm.setup()

    raw_stem   = npz.stem.split("_enc")[0]
    scaler     = joblib.load(npz.parent / f"{raw_stem}_scaler.gz")
    feats      = list(scaler.feature_names_in_)

    model = BiLSTMForecast(n_feat=len(feats), hidden=hidden, layers=layers)
    model.feats = feats

    ckpt_dir = Path("results/checkpoints/bilstm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir,
                              filename=f"{npz.stem}-best",
                              monitor="val_loss", mode="min", save_top_k=1)
    es_cb   = EarlyStopping(monitor="val_loss", mode="min", patience=40)

    trainer = pl.Trainer(
        max_epochs=epochs, accelerator="auto",
        callbacks=[ckpt_cb, es_cb], logger=False,
        enable_checkpointing=True, enable_model_summary=False)

    trainer.fit(model, dm)


    model = BiLSTMForecast.load_from_checkpoint(ckpt_cb.best_model_path)
    model.feats = feats
    model.eval()
    device = next(model.parameters()).device

    P, T = [], []
    with torch.no_grad():
        for x, y in dm.test_dataloader():
            P.append(model(x.to(device)).cpu())
            T.append(y)
    P = torch.cat(P).numpy(); T = torch.cat(T).numpy()
    P_inv = scaler.inverse_transform(P); T_inv = scaler.inverse_transform(T)


    metrics, core = [], []
    for i, name in enumerate(feats):
        if name.endswith(("_sin", "_cos")):
            continue
        rmse = math.sqrt(np.mean((P_inv[:, i] - T_inv[:, i]) ** 2))
        if name in {"eccentricity", "semi_major_axis", "mean_motion"}:
            denom = T_inv[:, i] if name != "eccentricity" else 0.02
            mape = np.mean(np.abs(P_inv[:, i] - T_inv[:, i]) /
                           (np.abs(denom) + 1e-8)) * 100
            core.append(mape)
        else:
            mape = None
        metrics.append({"feature": name, "RMSE": rmse,
                        **({"MAPE%": mape} if mape is not None else {})})

    for base in ["inclination", "raan", "arg_perigee", "mean_anomaly"]:
        try:
            s = feats.index(f"{base}_sin"); c = feats.index(f"{base}_cos")
        except ValueError:
            continue
        diff = np.unwrap(np.arctan2(P_inv[:, s], P_inv[:, c])
                         - np.arctan2(T_inv[:, s], T_inv[:, c]))
        rmse_deg = np.sqrt(np.mean(diff ** 2)) * 180 / np.pi
        metrics.append({"feature": f"{base}_deg", "RMSE_deg": rmse_deg})

    overall = 100 - float(np.mean(core)) if core else None
    result = dict(metrics=native(metrics), overall_acc_percent=overall,
                  hidden=hidden, layers=layers, epochs=epochs, batches=batch)

    out_dir = Path("results/bilstm"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{npz.stem}.json").write_text(json.dumps(result, indent=2))
    print(f"{npz.name}: overall={overall:.2f}%  → {out_dir}/metrics_{npz.stem}.json")


# ────────────────── CLI ─────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-glob", required=True,
                    help='e.g. "data/processed/*_h*.npz"')
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    args = ap.parse_args()

    for fp in sorted(glob.glob(args.npz_glob)):
        train_one(Path(fp), args.epochs, args.batch,
                  args.hidden, args.layers)

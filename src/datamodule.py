# src/data/datamodule.py
# -----------------------------------------------------------
# Re‑usable Lightning‑friendly data‑loader for orbital window
# datasets (NPZ with arrays X [N,L,F] and y [N,F]).
# -----------------------------------------------------------
from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import pytorch_lightning as pl


class WindowDataset(Dataset):

    def __init__(self, npz_file: Path):
        self._npz = np.load(npz_file, allow_pickle=False, mmap_mode="r")
        self.X = self._npz["X"]  
        self.y = self._npz["y"]   

        self.N, self.L, self.F = self.X.shape
        self.H = int(npz_file.stem.split("_h")[-1])  # horizon days
        self.file = npz_file

    def __len__(self) -> int:              # noqa: Dunder
        return self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: Dunder
        # Cast only the returned slice to torch.Tensor to stay lazy in RAM.
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.from_numpy(self.y[idx]).float(),
        )


class OrbitsModule(pl.LightningDataModule):

    def __init__(
        self,
        npz_glob: str,
        batch_size: int = 128,
        num_workers: int = 0,
        splits: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        pin_memory: bool = True,
    ):
        super().__init__()
        self.npz_glob = npz_glob
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if not np.isclose(sum(splits), 1.0):
            raise ValueError("splits must sum to 1.0")
        self.tr_frac, self.va_frac, self.te_frac = splits

        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None
    def prepare_data(self) -> None:  
        """No downloading needed — files are already on disk."""
        pass

    def setup(self, stage: str | None = None) -> None:  # noqa: D401
        """Build datasets once per process."""
        if self.train_ds is not None:
            return  # already built

        files = sorted(glob.glob(self.npz_glob))
        if not files:
            raise FileNotFoundError(f"No NPZ matched pattern: {self.npz_glob}")

        datasets = [WindowDataset(Path(f)) for f in files]

        # Sanity‑check meta consistency
        Ls = {d.L for d in datasets}
        Hs = {d.H for d in datasets}
        Fs = {d.F for d in datasets}
        if len(Ls) > 1 or len(Hs) > 1 or len(Fs) > 1:
            raise ValueError(
                "All NPZ files fed to one experiment must share window/horizon/feature dims."
            )

        full_ds = ConcatDataset(datasets)
        N = len(full_ds)

        n_train = int(self.tr_frac * N)
        n_val = int(self.va_frac * N)
        n_test = N - n_train - n_val

        # Chronological order is preserved due to ConcatDataset order (no shuffling)
        indices = list(range(N))
        self.train_ds = Subset(full_ds, indices[:n_train])
        self.val_ds = Subset(full_ds, indices[n_train : n_train + n_val])
        self.test_ds = Subset(full_ds, indices[-n_test:])

    def train_dataloader(self) -> DataLoader:  # noqa: D401
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,         
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:  
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader: 
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

if __name__ == "__main__":
    import argparse
    from rich import print as rprint

    ap = argparse.ArgumentParser(description="Inspect orbital NPZ datasets")
    ap.add_argument("npz_glob", help="Glob pattern for *_w*_h*.npz files")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--splits", nargs=3, type=float, default=(0.70, 0.15, 0.15))
    args = ap.parse_args()

    dm = OrbitsModule(
        npz_glob=args.npz_glob,
        batch_size=args.batch,
        splits=tuple(args.splits),
    )
    dm.setup()

    rprint(
        {
            "train": len(dm.train_ds),
            "val": len(dm.val_ds),
            "test": len(dm.test_ds),
            "batches/train": len(dm.train_dataloader()),
        }
    )

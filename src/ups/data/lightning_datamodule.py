from __future__ import annotations

"""PyTorch Lightning DataModule for UPS."""

import copy
from typing import Any, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ups.data.latent_pairs import build_latent_pair_loader


class UPSDataModule(pl.LightningDataModule):
    """Wraps existing latent pair loaders for Lightning."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        # Preserve samplers created inside build_latent_pair_loader (task-balanced, DDP-aware)
        self.replace_sampler_ddp = False

    def _loader_with_split(self, split: str) -> DataLoader:
        cfg_copy: dict[str, Any] = copy.deepcopy(self.cfg)
        cfg_copy.setdefault("data", {})
        cfg_copy["data"]["split"] = split
        return build_latent_pair_loader(cfg_copy)

    def train_dataloader(self):
        if self.train_loader is None:
            self.train_loader = self._loader_with_split("train")
        return self.train_loader

    def val_dataloader(self):
        if self.val_loader is None:
            self.val_loader = self._loader_with_split("val")
        return self.val_loader

    def test_dataloader(self):
        if self.test_loader is None:
            self.test_loader = self._loader_with_split("test")
        return self.test_loader

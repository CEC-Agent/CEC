from __future__ import annotations
from functools import partial

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import cec.utils as U
from cec.data.dmlab.dataset import DMLabDataset, DTDMLabDataset, ATDMLabDataset
from cec.data.dmlab.process import (
    collate_fn as _collate_fn,
    collate_fn_AT as _collate_fn_AT,
)


class DMLabDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        ctx_len: int,
        batch_size: int,
        val_batch_size: int,
        dataloader_num_workers: int,
        task: str,
        train_levels: list[int],
        train_portion: float = 0.8,
        num_trajs: int | None = None,
        seed: int | None = None,
        n_episodes_per_level: int | list[int] | tuple[int, int] = 1,
        shuffle: bool = True,
        even_spacing: bool = False,
        adapted: bool = False,
        at_or_dt=None,
    ):
        super().__init__()
        self.path = path
        self.ctx_len = ctx_len
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size
        self._num_workers = dataloader_num_workers
        self._train_portion = train_portion
        self.task = task
        self.num_trajs = num_trajs
        self.seed = seed
        self.train_levels = train_levels
        self.n_episodes_per_level = n_episodes_per_level
        self.shuffle = shuffle
        self.even_spacing = even_spacing
        self.adapted = adapted

        self.collate_fn = partial(
            _collate_fn_AT if at_or_dt == "at" else _collate_fn,
            ctx_len=ctx_len,
        )
        if at_or_dt is None:
            self._ds_cls = DMLabDataset
        else:
            self._ds_cls = DTDMLabDataset if at_or_dt == "dt" else ATDMLabDataset

        self._dataset, self._train_set, self._val_set = None, None, None

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self._dataset = self._ds_cls(
                path=self.path,
                task=self.task,
                num_trajs=self.num_trajs,
                seed=self.seed,
                train_levels=self.train_levels,
                n_episodes_per_level=self.n_episodes_per_level,
                shuffle=self.shuffle,
                even_spacing=self.even_spacing,
                adapted=self.adapted,
            )
            self._train_set, self._val_set = U.sequential_split_dataset(
                self._dataset,
                split_portions=[self._train_portion, 1 - self._train_portion],
            )

    def train_dataloader(self):
        return DataLoader(
            self._train_set,
            batch_size=self._batch_size,
            collate_fn=self.collate_fn,
            num_workers=min(self._batch_size, self._num_workers),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_set,
            batch_size=self._val_batch_size,
            collate_fn=self.collate_fn,
            num_workers=min(self._batch_size, self._num_workers),
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        For test_step(), simply returns None N times.
        test_step() can have arbitrary logic
        """
        return DummyDataset(batch_size=1).get_dataloader()

    @property
    def max_traj_len(self):
        return self._dataset.max_traj_len


class DummyDataset(Dataset):
    """
    For test_step(), simply returns None N times.
    test_step() can have arbitrary logic
    """

    def __init__(self, batch_size, epoch_len=1):
        """
        Still set batch_size because pytorch_lightning tracks it
        """
        self.n = epoch_len
        self._batch_size = batch_size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return np.zeros((self._batch_size,), dtype=bool)

    def get_dataloader(self) -> DataLoader:
        """
        Our dataset directly returns batched tensors instead of single samples,
        so for DataLoader we don't need a real collate_fn and set batch_size=1
        """
        return DataLoader(
            self,
            batch_size=1,
            num_workers=72,
            pin_memory=True,
            shuffle=False,
            collate_fn=_singleton_collate_fn,
        )


def _singleton_collate_fn(tensor_list):
    """
    Our dataset directly returns batched tensors instead of single samples,
    so for DataLoader we don't need a real collate_fn.
    """
    assert len(tensor_list) == 1, "INTERNAL: collate_fn only allows a single item"
    return tensor_list[0]

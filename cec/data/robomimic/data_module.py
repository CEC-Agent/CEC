from __future__ import annotations
import os
from typing import Literal
from functools import partial

import json
import h5py
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from cec.data.robomimic.process import collate_fn as _collate_fn
from cec.data.robomimic.dataset import RoboMimicDataset


class RoboMimicDataModule(LightningDataModule):
    def __init__(
        self,
        task: Literal["lift", "can"],
        data_parent_dir: str,
        ctx_len: int,
        batch_size: int,
        val_batch_size: int,
        dataloader_num_workers: int,
        n_episodes_per_level: int | list[int] | tuple[int, int],
        seed: int | None = None,
        shuffle: bool = True,
        cache_in_mem: bool = False,
        precision: int = 32,
    ):
        super().__init__()
        data_path = os.path.join(data_parent_dir, task, "mh", "image.hdf5")
        assert os.path.exists(data_path), f"Data path {data_path} does not exist."
        self._data_path = data_path

        self.ctx_len = ctx_len
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size
        self._num_workers = dataloader_num_workers
        self.seed = seed
        self.n_episodes_per_level = n_episodes_per_level
        self.shuffle = shuffle
        self.cache_in_mem = cache_in_mem

        self.collate_fn = partial(
            _collate_fn,
            ctx_len=ctx_len,
            fp16=precision == 16,
        )

        self._train_set, self._val_set = None, None

        self._eval_env_meta = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self._train_set = RoboMimicDataset(
                path=self._data_path,
                partition="train",
                seed=self.seed,
                n_episodes_per_level=self.n_episodes_per_level,
                shuffle=self.shuffle,
                cache_in_mem=self.cache_in_mem,
            )
            self._val_set = RoboMimicDataset(
                path=self._data_path,
                partition="valid",
                seed=self.seed,
                n_episodes_per_level=self.n_episodes_per_level,
                shuffle=self.shuffle,
                cache_in_mem=self.cache_in_mem,
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
    def eval_env_meta(self):
        if self._eval_env_meta is None:
            hdf5_file = h5py.File(self._data_path, "r", swmr=True, libver="latest")
            self._eval_env_meta = json.loads(hdf5_file["data"].attrs["env_args"])
            hdf5_file.close()
        return self._eval_env_meta


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
            num_workers=0,
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

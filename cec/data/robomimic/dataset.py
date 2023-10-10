from __future__ import annotations

import os
from typing import Literal
from math import ceil

import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from cec.data.robomimic.process import prepare_sample


class RoboMimicDataset(Dataset):
    curricular = ["worse", "okay", "better"]
    modalities = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "agentview_image",
        "robot0_eye_in_hand_image",
    ]

    def __init__(
        self,
        path: str,
        partition: Literal["train", "valid"],
        seed: int | None = None,
        n_episodes_per_level: int | list[int] | tuple[int, int] = 1,
        shuffle: bool = True,
        cache_in_mem: bool = False,
    ):
        super().__init__()
        assert os.path.exists(path)
        self._path = path
        self._hdf5_file = None

        # construct demo pointer
        assert partition in {"train", "valid"}
        curriculum_demo_ids = {
            expertise: [
                elem.decode("utf-8")
                for elem in np.array(self.hdf5_file[f"mask/{expertise}_{partition}"][:])
            ]
            for expertise in self.curriculum
        }
        self._curriculum_demo_ids = {
            k: np.sort([int(v[len("demo_") :]) for v in vs])
            for k, vs in curriculum_demo_ids.items()
        }

        self._random_state = np.random.RandomState(seed)
        self._shuffle = shuffle

        # compute dataset size
        if isinstance(n_episodes_per_level, int):
            self._expected_n_episodes_per_level = n_episodes_per_level
        else:
            assert len(n_episodes_per_level) == 2
            self._expected_n_episodes_per_level = int(sum(n_episodes_per_level) / 2)
        n_episodes_per_sample = self._expected_n_episodes_per_level * len(
            self.curriculum
        )
        self._n_episodes_per_level = n_episodes_per_level
        self._dataset_size = ceil(
            sum(len(v) for v in self._curriculum_demo_ids.values())
            / n_episodes_per_sample
        )

        # handle caching if requested
        if cache_in_mem:
            all_demos = np.concatenate(list(self._curriculum_demo_ids.values()))
            self._hdf5_cache = self._load_dataset_in_memory(all_demos, self.hdf5_file)
        else:
            self._hdf5_cache = None

        self._close_and_delete_hdf5_handle()

    def _load_dataset_in_memory(self, demos, file):
        all_data = {}
        print("Load dataset into memory...")
        for demo in tqdm(demos):
            all_data[demo] = self._load_single_demo(demo, file, self.modalities)
        return all_data

    @staticmethod
    def _load_single_demo(demo, file, obs_keys):
        return {
            "obs": {
                k: file[f"data/demo_{demo}/obs/{k}"][()].astype("float32")
                for k in obs_keys
            },
            "action": file[f"data/demo_{demo}/actions"][()].astype("float32"),
        }

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self._path, "r", swmr=True, libver="latest")
        return self._hdf5_file

    def _close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None

    def __len__(self):
        return self._dataset_size

    def __del__(self):
        self._close_and_delete_hdf5_handle()

    def __getitem__(self, index):
        # concatenate n episodes from each level
        for i, l in enumerate(self.curriculum):
            n_episodes_this_level = (
                self._n_episodes_per_level
                if isinstance(self._n_episodes_per_level, int)
                else self._random_state.randint(
                    low=min(self._n_episodes_per_level),
                    high=max(self._n_episodes_per_level),
                )
            )
            start_idx = (
                self._random_state.randint(len(self._curriculum_demo_ids[l]))
                if self._shuffle
                else index * self._expected_n_episodes_per_level
            )
            end_idx = start_idx + n_episodes_this_level
            demo_ids = self._curriculum_demo_ids[l][start_idx:end_idx]
            for j, demo_id in enumerate(demo_ids):
                demo_data = (
                    self._hdf5_cache[demo_id]
                    if self._hdf5_cache is not None
                    else self._load_single_demo(
                        demo_id, self.hdf5_file, self.modalities
                    )
                )
                obs, action = demo_data["obs"], demo_data["action"]
                obs, action = prepare_sample(obs, action)
                if i == 0 and j == 0:
                    obs_all, action_all = obs, action
                else:
                    for k in obs.keys():
                        obs_all[k] = np.concatenate((obs_all[k], obs[k]), axis=0)
                    action_all = np.concatenate((action_all, action), axis=0)
        return obs_all, action_all

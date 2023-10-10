from __future__ import annotations

import os

import numpy as np
from torch.utils.data import Dataset

import cec.utils as U
from cec.data.dmlab.process import prepare_sample


class DMLabDataset(Dataset):
    def __init__(
        self,
        path: str,
        task: str,
        train_levels: list[int],
        num_trajs: int | None = None,
        seed: int | None = None,
        n_episodes_per_level: int | list[int] | tuple[int, int] = 1,
        shuffle: bool = True,
        even_spacing: bool = False,
        adapted: bool = False,
    ):
        assert os.path.exists(path)
        train_levels = sorted(train_levels)
        levels = [f"{task}_l{train_levels[0]}"]
        adapted = "_adapted" if adapted else ""
        if len(train_levels) > 1:
            levels += [f"{task}_l{room_num}{adapted}" for room_num in train_levels[1:]]
        self.modalities = ["partial_rgb", "action"]

        random_state = np.random.RandomState(seed)

        self.traj_paths = {level: None for level in levels}
        self._ptrs = {level: None for level in levels}
        # check if all levels are the same
        same_levels = True if len(set(levels)) == 1 else False
        for i, level in enumerate(levels):
            task_path = os.path.join(path, level, task)
            # get all folders in task_path, note that valid trajectories are in folders with names that are integers
            if not same_levels:
                folders = [f for f in sorted(os.listdir(task_path)) if f.isdigit()]
            else:
                folders = [
                    f
                    for f in sorted(os.listdir(task_path))
                    if f.isdigit() and int(f) % len(levels) == i
                ]
            if shuffle:
                random_state.shuffle(folders)
            if num_trajs is not None:
                assert num_trajs <= len(folders), f"{num_trajs} > {len(folders)}"
                if even_spacing:
                    # obtain trajectories with even spacing
                    folders = [
                        folders[i]
                        for i in np.linspace(0, len(folders) - 1, num_trajs, dtype=int)
                    ]
                else:
                    folders = folders[-num_trajs:]
            traj_paths = [os.path.join(task_path, f) for f in folders]
            self.traj_paths[level] = [traj_path for traj_path in traj_paths]
            self.n_demos_per_level = len(self.traj_paths[level])
            self._ptrs[level] = random_state.permutation(self.n_demos_per_level)
        self.levels = levels
        self.n_episodes_per_level = n_episodes_per_level

    def __len__(self):
        # assuming all levels have the same number of demos
        return self.n_demos_per_level * len(self.levels)

    def __getitem__(self, index):
        if isinstance(self.n_episodes_per_level, int):
            n_episodes_per_level = self.n_episodes_per_level
        else:
            assert len(self.n_episodes_per_level) == 2
            n_episodes_min, n_episodes_max = min(self.n_episodes_per_level), max(
                self.n_episodes_per_level
            )
            n_episodes_per_level = np.random.randint(
                low=n_episodes_min, high=n_episodes_max + 1
            )

        # concatenate n episodes from each level
        for i, l in enumerate(self.levels):
            shuffled_idx = self._ptrs[l][index % self.n_demos_per_level]
            if shuffled_idx + n_episodes_per_level > self.n_demos_per_level:
                shuffled_idx = np.random.randint(
                    low=0, high=(self.n_demos_per_level - n_episodes_per_level + 1)
                )
            for n in range(n_episodes_per_level):
                traj_path = self.traj_paths[l][shuffled_idx + n]
                traj = U.load_pickle(os.path.join(traj_path, "trajectory.pkl"))
                obs = {
                    modality: traj[modality]
                    for modality in self.modalities
                    if modality != "action"
                }
                action = traj["action"]
                # ensure action is long
                if action.dtype != np.int64:
                    action = action.astype(np.int64)
                obs, action = prepare_sample(obs, action)
                if i == 0 and n == 0:
                    obs_all = obs
                    action_all = action
                else:
                    for k in obs.keys():
                        obs_all[k] = np.concatenate((obs_all[k], obs[k]), axis=0)
                    action_all = np.concatenate((action_all, action), axis=0)
        return obs_all, action_all


class DTDMLabDataset(DMLabDataset):
    def __getitem__(self, index):
        level_idx = index // self.n_demos_per_level
        level_idx = list(self._ptrs.keys())[level_idx]

        # concatenate n episodes
        shuffled_idx = self._ptrs[level_idx][index % self.n_demos_per_level]
        traj = U.load_pickle(
            os.path.join(self.traj_paths[level_idx][shuffled_idx], "trajectory.pkl")
        )
        obs = {
            modality: traj[modality]
            for modality in self.modalities
            if modality != "action"
        }
        action = traj["action"]
        # ensure action is long
        if action.dtype != np.int64:
            action = action.astype(np.int64)
        obs, action = prepare_sample(obs, action)  # (L, ...)
        # construct return-to-go
        reward = traj["reward"]
        return_to_go = np.array(
            [np.sum(reward[t:]) for t in range(len(reward))]
        ).astype(np.float32)[..., None]
        obs["return_to_go"] = return_to_go
        return obs, action


class ATDMLabDataset(DMLabDataset):
    def __getitem__(self, index):
        if isinstance(self.n_episodes_per_level, int):
            n_episodes_to_sample = self.n_episodes_per_level
        else:
            assert len(self.n_episodes_per_level) == 2
            n_episodes_min, n_episodes_max = min(self.n_episodes_per_level), max(
                self.n_episodes_per_level
            )
            n_episodes_to_sample = np.random.randint(
                low=n_episodes_min, high=n_episodes_max + 1
            )

        level_idx = index // self.n_demos_per_level
        level_idx = list(self._ptrs.keys())[level_idx]

        # concatenate n episodes
        shuffled_idx = self._ptrs[level_idx][index % self.n_demos_per_level]
        if shuffled_idx + n_episodes_to_sample > self.n_demos_per_level:
            shuffled_idx = np.random.randint(
                low=0, high=(self.n_demos_per_level - n_episodes_to_sample + 1)
            )
        trajs = [
            U.load_pickle(
                os.path.join(
                    self.traj_paths[level_idx][shuffled_idx + n], "trajectory.pkl"
                )
            )
            for n in range(n_episodes_to_sample)
        ]
        # sort ascending by return
        trajs = sorted(trajs, key=lambda x: np.sum(x["reward"]))
        target_return_to_go = np.sum(
            trajs[-1]["reward"]
        )  # largest return-to-go is the target

        for i, traj in enumerate(trajs):
            obs = {
                modality: traj[modality]
                for modality in self.modalities
                if modality != "action"
            }
            action = traj["action"]
            # ensure action is long
            if action.dtype != np.int64:
                action = action.astype(np.int64)
            obs, action = prepare_sample(obs, action)  # (L, ...)
            # construct return-to-go
            reward = traj["reward"]
            return_to_go = np.array(
                [target_return_to_go - np.sum(reward[:t]) for t in range(len(reward))]
            ).astype(np.float32)[..., None]
            obs["return_to_go"] = return_to_go
            obs["reward"] = reward.astype(np.float32)[..., None]
            # construct the "d" token
            d_token = np.zeros_like(reward, dtype=bool)
            d_token[-1] = np.sum(reward) >= target_return_to_go
            obs["d_token"] = d_token
            if i == 0:
                obs_all = obs
                action_all = action
            else:
                obs_all = U.any_concat([obs_all, obs], dim=0)
                action_all = np.concatenate((action_all, action), axis=0)
        last_traj_T = len(action)
        return obs_all, action_all, last_traj_T

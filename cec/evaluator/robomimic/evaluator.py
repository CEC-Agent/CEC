from __future__ import annotations

from typing import Literal
from functools import partial

import gym
import torch
import numpy as np
from tqdm import tqdm
from robomimic.utils.env_utils import (
    create_env_from_metadata as _create_env_from_metadata,
)
import robomimic.utils.obs_utils as ObsUtils

import cec.utils as U
from cec.evaluator.base import BaseEvaluator
from cec.evaluator.shmem_vec_env import SubprocVectorEnv
from cec.evaluator.robomimic.utils import (
    horizons,
    obs_modality_specs,
    observation_spaces,
)


def create_env_from_metadata(*args, mode, task, n_episodes, episode_horizon, **kwargs):
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs[mode][task])
    return RoboMimicGymAutoReset(
        _create_env_from_metadata(*args, **kwargs),
        mode,
        task,
        n_episodes,
        episode_horizon,
    )


class ShmemCrossEpisodicVecEvaluator(BaseEvaluator):
    def __init__(
        self,
        *,
        env_meta: dict,
        mode: Literal["low_dim", "image"],
        n_eval_episodes: int = 50,
        vec_env_size: int = 1,
        deterministic_eval: bool = True,
    ):
        super().__init__()

        self._env_meta = env_meta
        self._mode = mode
        self._deterministic_eval = deterministic_eval
        assert vec_env_size >= 1
        self._vec_env_size = vec_env_size
        self._n_eval_episodes = n_eval_episodes
        self._vec_env = None
        self._results_per_eval = None

    @torch.no_grad()
    def start(self, policy):
        device = policy.device
        self._results_per_eval = []
        if self._vec_env is None:
            self._create_vec_envs()
        prev_action_input = policy.prev_action_input

        pbar = tqdm(
            total=horizons[self._env_meta["env_name"]] * self._n_eval_episodes,
            desc="Running vectorized cross-episodic evaluation...",
            leave=True,
        )

        ready_env_ids = np.arange(len(self._vec_env))

        first_mask = torch.ones(
            len(ready_env_ids),
            1,
            device=device,
            dtype=bool,
        )

        state = policy.transformer.initial_state(
            batchsize=len(ready_env_ids), device=device
        )

        vec_obs = self._vec_env.reset()
        prev_action = (
            np.zeros((len(ready_env_ids), 1, 7), dtype=np.float32)
            if prev_action_input
            else None
        )
        pbar.reset()

        while len(ready_env_ids) > 0:
            vec_obs = [obs for obs in vec_obs]
            vec_obs = [U.add_batch_dim(obs) for obs in vec_obs]  # list of (T = 1, ...)
            vec_obs = U.stack_sequence_fields(vec_obs)  # dict of (B, T = 1, ...)
            vec_obs = U.any_to_datadict(vec_obs)
            vec_obs = vec_obs.to_torch_tensor(device=device, dtypes=policy.dtype)
            prev_action = (
                U.any_to_torch_tensor(prev_action, device=device, dtype=policy.dtype)
                if prev_action_input
                else None
            )
            action, state = policy.act(
                obs=vec_obs,
                deterministic=self._deterministic_eval,
                first_mask=first_mask,
                state=state,
                acs=prev_action,
            )
            action = U.any_to_numpy(action)  # (B, T, ...)
            # the last timestep action is the one we want to execute
            action = U.any_slice(action, np.s_[:, -1])  # (B, ...)
            vec_obs, _, done, _ = self._vec_env.step(action, id=ready_env_ids)
            prev_action = (
                U.any_slice(action, np.s_[:, None]) if prev_action_input else None
            )
            pbar.update(1)
            if np.any(done):
                terminated_env_local_ids = np.where(done)[0]
                terminated_env_global_ids = ready_env_ids[terminated_env_local_ids]

                successes = self._vec_env.get_env_attr(
                    "successes", id=terminated_env_global_ids
                )  # list of (n_episodes,)
                self._results_per_eval.extend(successes)

                mask = np.ones_like(ready_env_ids, dtype=bool)
                mask[terminated_env_local_ids] = False
                ready_env_ids = ready_env_ids[mask]
                vec_obs = vec_obs[mask]
                prev_action = prev_action[mask] if prev_action_input else None

                state = U.any_slice(state, mask)
            first_mask = torch.zeros(
                len(ready_env_ids),
                1,
                device=device,
                dtype=bool,
            )

        self._results_per_eval = np.array(self._results_per_eval)

    def get_results(self) -> dict:
        results = {
            "avg_success": np.mean(self._results_per_eval),
            "std_success": np.std(self._results_per_eval),
        }
        return results

    def _create_vec_envs(self):
        assert self._vec_env is None, "Vec envs already created"

        # register obs specs
        ObsUtils.initialize_obs_utils_with_obs_specs(
            obs_modality_specs[self._mode][self._env_meta["env_name"]]
        )

        env_fns = [
            partial(
                create_env_from_metadata,
                env_meta=self._env_meta,
                env_name=self._env_meta["env_name"],
                render=False,
                render_offscreen=self._mode == "image",
                use_image_obs=self._mode == "image",
                mode=self._mode,
                task=self._env_meta["env_name"],
                n_episodes=self._n_eval_episodes,
                episode_horizon=horizons[self._env_meta["env_name"]],
            )
            for _ in range(self._vec_env_size)
        ]
        self._vec_env = SubprocVectorEnv(env_fns=env_fns)

    def close(self):
        if self._vec_env is not None:
            self._vec_env.close()
            self._vec_env = None


class RoboMimicGymAutoReset(gym.Wrapper):
    def __init__(self, env, mode, task, n_episodes: int, episode_horizon: int):
        super().__init__(env)
        self.observation_space = observation_spaces[mode][task]

        self._n_episodes = n_episodes
        self._episode_horizon = episode_horizon

        self._elapsed_steps, self._elapsed_episodes = None, None
        self._successes = None

    @property
    def successes(self):
        return self._successes

    def close(self):
        return self.env.env.close()

    def _filter_obs(self, obs):
        return {
            k: v for k, v in obs.items() if k in self.observation_space.spaces.keys()
        }

    def _is_done(self):
        return (
            self._elapsed_steps >= self._episode_horizon
            or self.env.is_success()["task"]
        )

    def reset(self, **kwargs):
        self._elapsed_steps, self._elapsed_episodes = 0, 0
        self._successes = []

        obs = self.env.reset(**kwargs)
        return self._filter_obs(obs)

    def step(self, action):
        obs, reward, _, info = self.env.step(action)
        done = False
        self._elapsed_steps += 1

        if self._is_done():
            self._elapsed_episodes += 1
            self._successes.append(self.env.is_success()["task"])
            if self._elapsed_episodes >= self._n_episodes:
                done = True
            else:
                obs = self.env.reset()
                self._elapsed_steps = 0
        return self._filter_obs(obs), reward, done, info

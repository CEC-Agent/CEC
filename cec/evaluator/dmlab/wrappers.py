from collections import deque

import gym
import numpy as np
import torch
from gym import spaces


class RGBImgObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.obs_shape = env.observation_space.spaces["obs"].shape
        self.observation_space.spaces["partial_rgb"] = spaces.Box(
            low=0,
            high=255,
            shape=(
                3,
                self.obs_shape[1],
                self.obs_shape[2],
            ),
            dtype="uint8",
        )

    @staticmethod
    def observation(obs):
        return {"partial_rgb": obs["obs"]}

    def step(self, action):
        results = self.env.step(action)
        if len(results) == 4:
            obs, reward, done, info = results
        elif len(results) == 5:
            obs, reward, terminated, truncated, info = results
            # terminate the episode if the goal is reached
            done = terminated or truncated or (reward > 0)
        return self.observation(obs), reward, done, info


class ChainEnvWrapper(gym.core.Wrapper):
    """A wrapper that chains multiple environments together."""

    def __init__(
        self, envs, n_episodes_per_level=1, action_repeat=1, in_context_episodes=None
    ):
        super().__init__(envs[0])
        self.envs = envs
        self.env_idx = 0
        self.env = self.envs[self.env_idx]
        self.n_episodes_per_level = n_episodes_per_level
        self.episode_idx = 0
        self.env_step = 0
        self.action_repeat = action_repeat
        self.in_context_episodes = in_context_episodes
        if isinstance(action_repeat, list):
            assert len(action_repeat) == len(envs)
            self.action_repeat_list = action_repeat
            self.action_repeat = self.action_repeat_list[0]
        self._eval_results = None

    def step(self, action):
        results = self.env.step(action)
        # check if we are on the last environment
        if self.env_idx == len(self.envs) - 1:
            self.env_step += 1
        if len(results) == 4:
            obs, reward, done, info = results
        elif len(results) == 5:
            obs, reward, terminated, truncated, info = results
            done = terminated or truncated or (reward > 0)
        is_last_timestep = False
        if done:
            if self.env_idx == len(self.envs) - 1:
                if "test_level_return" not in self._eval_results:
                    self._eval_results["test_level_return"] = []
                self._eval_results["test_level_return"].append(reward.cpu().item())
                if self.in_context_episodes is not None:
                    self.n_episodes_per_level = self.in_context_episodes

            # continue with the next episode in the same environment
            self.episode_idx += 1
            done = torch.tensor([False])
            if self.episode_idx == self.n_episodes_per_level:
                # finished all episodes in the current environment
                self.episode_idx = 0
                self.env_idx += 1
                if self.env_idx == len(self.envs):
                    # finished all environments
                    self.env_idx = 0
                    done = torch.tensor([True])
                self.env = self.envs[self.env_idx]
                if hasattr(self, "action_repeat_list"):
                    self.action_repeat = self.action_repeat_list[self.env_idx]
            obs, _ = self.env.reset()
            is_last_timestep = True
        # convert to numpy array
        if isinstance(obs, dict):
            obs = {k: np.array(v) for k, v in obs.items()}
        # extend info
        info[0]["env_idx"] = self.env_idx
        info[0]["episode_idx"] = self.episode_idx
        info[0]["is_eval_env"] = self.env_idx == len(self.envs) - 1
        info[0]["is_last_timestep"] = is_last_timestep
        return obs, reward, done, info

    def reset(self):
        self._eval_results = {}
        self.env_idx = 0
        self.episode_idx = 0
        if hasattr(self, "action_repeat_list"):
            self.action_repeat = self.action_repeat_list[0]
        obs, _ = self.env.reset()
        # convert to numpy array
        if isinstance(obs, dict):
            obs = {k: np.array(v) for k, v in obs.items()}
        return obs

    def render(self, mode="rgb_array"):
        return self.env.render(mode)

    def close(self):
        for env in self.envs:
            env.close()

    def seed(self, seed=None):
        for env in self.envs:
            env.seed(seed)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def eval_results(self):
        return self._eval_results


class AutoCurriculumWrapper(ChainEnvWrapper):
    def __init__(
        self,
        *,
        num_levels,
        consecutive_success: int,
        success_check_fn,
        envs,
        n_episodes_per_level,
        action_repeat=1,
    ):
        assert num_levels == len(envs)
        super().__init__(envs, n_episodes_per_level, action_repeat)
        self._success_check_fn = success_check_fn
        self._consecutive_success = consecutive_success
        self._num_levels = num_levels

        self._num_successes_all_levels = None
        self._latest_results = None

    def reset(self):
        self._num_successes_all_levels = np.zeros(self._num_levels, dtype=np.int32)
        self._latest_results = deque(
            [False] * self._consecutive_success, maxlen=self._consecutive_success
        )
        return super().reset()

    def step(self, action):
        results = self.env.step(action)
        # check if we are on the last environment
        if self.env_idx == len(self.envs) - 1:
            self.env_step += 1
        if len(results) == 4:
            obs, reward, done, info = results
        elif len(results) == 5:
            obs, reward, terminated, truncated, info = results
            done = terminated or truncated or (reward > 0)
        is_success = self._success_check_fn(reward)
        done = done or is_success
        is_last_timestep = False
        if done:
            self._latest_results.append(is_success)
            if self.env_idx == len(self.envs) - 1:
                if "test_level_return" not in self._eval_results:
                    self._eval_results["test_level_return"] = []
                self._eval_results["test_level_return"].append(reward.cpu().item())

            # continue with the next episode in the same environment
            self.episode_idx += 1
            done = torch.tensor([False])

            if is_success:
                self._num_successes_all_levels[self.env_idx] += 1

            if self.env_idx < (self._num_levels - 1) and all(self._latest_results):
                self._eval_results[f"L{self.env_idx + 1}_success_rate"] = (
                    self._num_successes_all_levels[self.env_idx] / self.episode_idx
                )
                self._eval_results[
                    f"L{self.env_idx + 1}_num_episodes"
                ] = self.episode_idx
                self._latest_results = deque(
                    [False] * self._consecutive_success,
                    maxlen=self._consecutive_success,
                )
                self.episode_idx = 0
                self.env_idx += 1
                self.env = self.envs[self.env_idx]
                if hasattr(self, "action_repeat_list"):
                    self.action_repeat = self.action_repeat_list[self.env_idx]
            elif self.episode_idx == self.n_episodes_per_level:
                self._eval_results[f"L{self.env_idx + 1}_success_rate"] = (
                    self._num_successes_all_levels[self.env_idx] / self.episode_idx
                )
                self._eval_results[
                    f"L{self.env_idx + 1}_num_episodes"
                ] = self.episode_idx
                # finished all episodes in the current environment
                self.episode_idx = 0
                self.env_idx += 1
                if self.env_idx == len(self.envs):
                    # finished all environments
                    self.env_idx = 0
                    done = torch.tensor([True])
                self.env = self.envs[self.env_idx]
                if hasattr(self, "action_repeat_list"):
                    self.action_repeat = self.action_repeat_list[self.env_idx]
            obs, _ = self.env.reset()
            is_last_timestep = True
        # convert to numpy array
        if isinstance(obs, dict):
            obs = {k: np.array(v) for k, v in obs.items()}
        # extend info
        info[0]["env_idx"] = self.env_idx
        info[0]["episode_idx"] = self.episode_idx
        info[0]["is_eval_env"] = self.env_idx == len(self.envs) - 1
        info[0]["is_last_timestep"] = is_last_timestep
        return obs, reward, done, info

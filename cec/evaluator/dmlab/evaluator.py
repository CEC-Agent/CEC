from __future__ import annotations

import os
from math import ceil
from functools import partial
from collections import deque

import torch
import numpy as np
from tqdm import tqdm
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sf_examples.dmlab.train_dmlab import parse_dmlab_args, register_dmlab_components

import cec.utils as U
from cec.evaluator.base import BaseEvaluator
from cec.evaluator.shmem_vec_env import ShmemVectorEnv
from cec.evaluator.dmlab.wrappers import (
    RGBImgObsWrapper,
    ChainEnvWrapper,
    AutoCurriculumWrapper,
)


ALL_TASKS = {
    "dmlab_goal": "explore_goal_locations_small",
    "dmlab_goal_large": "explore_goal_locations_large",
    "dmlab_obstructed": "explore_obstructed_goals_small",
    "dmlab_obstructed_large": "explore_obstructed_goals_large",
    "dmlab_watermaze": "rooms_watermaze",
    "dmlab_irreversible_path": "skymaze_irreversible_path_hard",
    "dmlab_irreversible_path_hard": "skymaze_irreversible_path_hard",
}


class DMLabEvaluator(BaseEvaluator):
    def __init__(
        self,
        *,
        task: str,
        vec_env_size: int | None,
        n_eval_per_level: int = 100,
        seed: int | None = None,
        deterministic: bool,
        difficulty_levels: int | list | None,
        n_episodes_per_level: int | None = 1,
        action_repeat: int | list = 1,
        in_context_episodes: int | None = None,
        use_auto_curriculum: bool = True,
        auto_curriculum_consecutive_success: int = 3,
        success_check_fn=lambda x: x > 0,
        auto_curriculum_n_episodes_per_level: int = 100,
    ):
        super().__init__()

        # Register DMLab and parse arguments
        register_dmlab_components()
        argv = [
            "--env",
            task,
            "--seed",
            str(seed),
            "--dmlab_extended_action_set",
            "True",
        ]
        self.cfg = parse_dmlab_args(argv=argv, evaluation=True)

        assert isinstance(task, str)
        task = [task]
        self.tasks = task
        vec_env_size = vec_env_size or os.cpu_count()
        self.vec_env_size = min(vec_env_size, len(task))
        self.n_vec_envs = ceil(len(task) / self.vec_env_size)
        self.n_eval_per_level = n_eval_per_level
        self.in_context_episodes = in_context_episodes

        self.modalities = ["partial_rgb", "action"]

        self._rng = np.random.default_rng(seed)

        self.vec_envs = None
        self._success = {ALL_TASKS[t]: [] for t in task}
        self._deterministic = deterministic
        self.difficulty_levels = difficulty_levels
        self.seed = seed
        self.n_episodes_per_level = n_episodes_per_level
        self.action_repeat = action_repeat

        self.use_auto_curriculum = use_auto_curriculum
        self.auto_curriculum_consecutive_success = auto_curriculum_consecutive_success
        self.success_check_fn = success_check_fn
        self.auto_curriculum_n_episodes_per_level = auto_curriculum_n_episodes_per_level

    def _create_vec_envs(self):
        def _env_fn(name, seed):
            if isinstance(self.difficulty_levels, list):
                envs = []
                for n in self.difficulty_levels:
                    if "irreversible" in name:
                        self.cfg.difficulty = n * 0.1
                    elif "watermaze" in name:
                        self.cfg.spawn_radius = n
                    else:
                        self.cfg.room_num = n
                    env = make_env_func_batched(
                        self.cfg,
                        env_config=AttrDict(
                            worker_index=0, vector_index=0, env_id=seed
                        ),
                        render_mode="rgb",
                    )
                    envs.append(RGBImgObsWrapper(env))
                if self.use_auto_curriculum:
                    return AutoCurriculumWrapper(
                        num_levels=len(self.difficulty_levels),
                        consecutive_success=self.auto_curriculum_consecutive_success,
                        success_check_fn=self.success_check_fn,
                        envs=envs,
                        n_episodes_per_level=self.auto_curriculum_n_episodes_per_level,
                        action_repeat=self.action_repeat,
                    )

                else:
                    return ChainEnvWrapper(
                        envs,
                        self.n_episodes_per_level,
                        action_repeat=self.action_repeat,
                        in_context_episodes=self.in_context_episodes,
                    )
            env = make_env_func_batched(
                self.cfg,
                env_config=AttrDict(worker_index=0, vector_index=0, env_id=seed),
                render_mode="rgb",
            )
            return ChainEnvWrapper(
                [RGBImgObsWrapper(env)],
                self.n_episodes_per_level,
                action_repeat=self.action_repeat,
                in_context_episodes=self.in_context_episodes,
            )

        vec_envs = []
        for i in range(self.n_vec_envs):
            tasks_this_vec_env = self.tasks[
                i * self.vec_env_size : (i + 1) * self.vec_env_size
            ]
            env_fns = [
                partial(_env_fn, level, self.seed + i)
                for i, level in enumerate(tasks_this_vec_env)
            ]
            vec_envs.append(ShmemVectorEnv(env_fns))
        return vec_envs

    def start(self, policy):
        self._success = {ALL_TASKS[t]: [] for t in self.tasks}
        if self.vec_envs is None:
            self.vec_envs = self._create_vec_envs()

        for i_th_env, vec_env in tqdm(
            enumerate(self.vec_envs), desc="Running each vector eval env...", leave=True
        ):
            for i_th_eval in tqdm(
                range(self.n_eval_per_level),
                desc=f"Running {self.n_eval_per_level} eval",
                leave=True,
            ):
                ready_env_ids = np.arange(len(vec_env))

                vec_obs = vec_env.reset()
                tasks = vec_env.get_env_attr("level_name", id=ready_env_ids)
                ep_len = 0

                vec_obs = list(vec_obs)
                vec_obs = [
                    {
                        k: v
                        for k, v in obs.items()
                        if k in self.modalities and k != "action"
                    }
                    for obs in vec_obs
                ]
                # stack
                vec_obs = U.stack_sequence_fields(vec_obs)
                # insert L dim
                vec_obs = U.any_slice(
                    vec_obs, np.s_[:, None, ...]
                )  # dict of (B, 1, ...)
                vec_obs = U.any_to_datadict(vec_obs)
                vec_obs = vec_obs.to_torch_tensor(device=policy.device)

                state = policy.transformer.initial_state(
                    batchsize=len(vec_env), device=policy.device
                )

                first_mask = torch.ones(
                    len(vec_env), 1, device=policy.device, dtype=bool
                )

                # set prev_action to be 0 for the first step
                prev_action = (
                    torch.zeros(
                        len(vec_env), 1, device=policy.device, dtype=torch.int64
                    )
                    if "action" in self.modalities
                    else None
                )  # (B, 1)

                while len(ready_env_ids) > 0:
                    actions, state = policy.act(
                        deterministic=self._deterministic,
                        obs=vec_obs,
                        state=state,
                        first_mask=first_mask,
                        acs=prev_action,
                    )
                    prev_action = actions
                    actions = U.any_to_numpy(actions)  # (B, 1, ...)
                    # remove the time dimension
                    actions = U.any_slice(actions, np.s_[:, 0])  # (B, ...)
                    vec_obs, vec_reward, vec_done, vec_info = vec_env.step(
                        action=actions, id=ready_env_ids
                    )
                    # action repeat
                    for _ in range(vec_env.get_env_attr("action_repeat")[0] - 1):
                        vec_obs, rew, done, vec_info = vec_env.step(
                            action=actions, id=ready_env_ids
                        )
                        vec_reward = vec_reward + rew
                        vec_done = np.logical_or(vec_done, done)
                    ep_len += 1

                    if np.any(vec_done):
                        terminated_env_local_ids = np.where(vec_done)[0]
                        terminated_env_global_ids = ready_env_ids[
                            terminated_env_local_ids
                        ]
                        terminated_tasks = vec_env.get_env_attr(
                            "level_name", id=terminated_env_global_ids
                        )
                        terminated_reward = vec_reward[terminated_env_local_ids]
                        for task, reward in zip(terminated_tasks, terminated_reward):
                            self._success[task].append(
                                reward > 0
                            )  # assuming the reward at the last step is the final reward

                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[terminated_env_local_ids] = False
                        ready_env_ids = ready_env_ids[mask]
                        vec_obs = vec_obs[mask]
                        # need to mask state as well
                        # state is a list of tuple[(B, ...), tuple[(B, ...), (B, ...)]]
                        state = [U.any_slice(s, mask) for s in state]
                        if prev_action is not None:
                            prev_action = prev_action[mask]
                    # break when all envs are done
                    if len(ready_env_ids) == 0:
                        break

                    # prepare for next step
                    vec_obs = list(vec_obs)  # list of obs obj
                    # filter modalities
                    vec_obs = [
                        {
                            k: v
                            for k, v in obs.items()
                            if k in self.modalities and k != "action"
                        }
                        for obs in vec_obs
                    ]
                    # stack
                    vec_obs = U.stack_sequence_fields(vec_obs)  # dict of (B, ...)
                    # insert L dim
                    vec_obs = U.any_slice(
                        vec_obs, np.s_[:, None, ...]
                    )  # dict of (B, 1, ...)
                    vec_obs = U.any_to_datadict(vec_obs)
                    vec_obs = vec_obs.to_torch_tensor(device=policy.device)

                    first_mask = torch.zeros(
                        len(ready_env_ids), 1, device=policy.device, dtype=bool
                    )

    def get_results(self) -> dict:
        return self._success


class DMLabDTEvaluator(DMLabEvaluator):
    target_return_to_go = {
        "dmlab_goal_large": 4.82,
        "dmlab_obstructed_large": 4.82,
        "dmlab_watermaze": 0.9869,
        "dmlab_irreversible_path": 5,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vec_env_size = kwargs["vec_env_size"]
        self.n_vec_envs = ceil(self.n_eval_per_level / self.vec_env_size)

    def _create_vec_envs(self):
        def _env_fn(level, seed):
            if isinstance(self.difficulty_levels, list):
                envs = []
                for n in self.difficulty_levels:
                    if "irreversible" in level:
                        self.cfg.difficulty = n * 0.1
                    elif "watermaze" in level:
                        self.cfg.spawn_radius = n
                    else:
                        self.cfg.room_num = n
                    env = make_env_func_batched(
                        self.cfg,
                        env_config=AttrDict(
                            worker_index=0, vector_index=0, env_id=seed
                        ),
                        render_mode="rgb",
                    )
                    envs.append(RGBImgObsWrapper(env))
                if self.use_auto_curriculum:
                    return AutoCurriculumWrapper(
                        num_levels=len(self.difficulty_levels),
                        consecutive_success=self.auto_curriculum_consecutive_success,
                        success_check_fn=self.success_check_fn,
                        envs=envs,
                        n_episodes_per_level=self.auto_curriculum_n_episodes_per_level,
                        action_repeat=self.action_repeat,
                    )

                else:
                    return ChainEnvWrapper(
                        envs,
                        self.n_episodes_per_level,
                        action_repeat=self.action_repeat,
                        in_context_episodes=self.in_context_episodes,
                    )
            env = make_env_func_batched(
                self.cfg,
                env_config=AttrDict(worker_index=0, vector_index=0, env_id=seed),
                render_mode="rgb",
            )
            return ChainEnvWrapper(
                [RGBImgObsWrapper(env)],
                self.n_episodes_per_level,
                action_repeat=self.action_repeat,
                in_context_episodes=self.in_context_episodes,
            )

        task = self.tasks[0]
        env_fns = [
            partial(_env_fn, task, self.seed + j) for j in range(self.vec_env_size)
        ]
        return ShmemVectorEnv(env_fns)

    def start(self, policy):
        assert len(self.tasks) == 1
        target_return_to_go_ = self.target_return_to_go[self.tasks[0]]

        if self.vec_envs is None:
            self.vec_envs = self._create_vec_envs()

        eval_success = []
        for i_th_eval in tqdm(
            range(self.n_vec_envs), desc="Running each vector eval env...", leave=True
        ):
            kv_cache = {i: None for i in range(self.vec_env_size)}
            ready_env_ids = np.arange(len(self.vec_envs))
            t = 0

            vec_obs = self.vec_envs.reset()  # np array of obs obj
            vec_obs = U.any_stack([obs["partial_rgb"] for obs in vec_obs], dim=0)[
                :, None, ...
            ]
            vec_obs = U.any_to_torch_tensor(vec_obs, device=policy.device)
            target_return_to_go = (
                torch.ones(
                    (self.vec_env_size, 1, 1), dtype=torch.float32, device=policy.device
                )
                * target_return_to_go_
            )
            prev_action = None

            while len(ready_env_ids) > 0:
                cache = (
                    U.any_stack([kv_cache[i] for i in ready_env_ids], dim=0)
                    if t > 0
                    else None
                )
                timestep = (torch.ones((1, 1), device=policy.device) * t).long()
                actions, new_caches = policy.act(
                    return_to_go=target_return_to_go,
                    state=vec_obs,
                    timestep=timestep,
                    prev_action=prev_action,
                    past_kv=cache,
                    deterministic=self._deterministic,
                )  # (B, 1), (B, nh, L, E)
                for idx, env_id in enumerate(ready_env_ids):
                    kv_cache[env_id] = U.any_slice(new_caches, np.s_[idx])  # (nh, L, E)
                prev_action = actions
                actions = U.any_to_numpy(actions)[:, 0]  # (B,)
                vec_obs, vec_reward, vec_done, vec_info = self.vec_envs.step(
                    action=actions, id=ready_env_ids
                )
                vec_obs = U.any_stack([obs["partial_rgb"] for obs in vec_obs], dim=0)[
                    :, None, ...
                ]
                vec_obs = U.any_to_torch_tensor(vec_obs, device=policy.device)

                if np.any(vec_done):
                    terminated_env_local_ids = np.where(vec_done)[0]
                    terminated_env_global_ids = ready_env_ids[terminated_env_local_ids]
                    terminated_reward = vec_reward[terminated_env_local_ids]
                    for i, reward in zip(terminated_env_global_ids, terminated_reward):
                        eval_success.append(reward > 0)

                    mask = np.ones_like(ready_env_ids, dtype=bool)
                    mask[terminated_env_local_ids] = False
                    ready_env_ids = ready_env_ids[mask]
                    target_return_to_go = target_return_to_go[mask]
                    vec_obs = vec_obs[mask]
                    prev_action = prev_action[mask]

                t += 1
                # break when all envs are done
                if len(ready_env_ids) == 0:
                    break
        eval_success = np.array(eval_success)
        self._eval_success = eval_success

    def get_results(self) -> dict:
        return self._eval_success


class DMLabATEvaluator(DMLabDTEvaluator):
    in_context_episodes_all_levels = {
        "dmlab_goal_large": 4 - 1,
        "dmlab_obstructed_large": 4 - 1,
        "dmlab_watermaze": 4 - 1,
        "dmlab_irreversible_path": 3 - 1,
    }

    def start(self, policy):
        cache_structure = None
        assert len(self.tasks) == 1
        target_return_to_go_ = self.target_return_to_go[self.tasks[0]]
        in_context_episodes = self.in_context_episodes_all_levels[self.tasks[0]]

        if self.vec_envs is None:
            self.vec_envs = self._create_vec_envs()

        episode_cache = {
            i: deque(maxlen=in_context_episodes) for i in range(self.vec_env_size)
        }
        eval_success = np.zeros(
            (self.vec_env_size, self.n_episodes_per_level), dtype=bool
        )

        for n_th_episode in tqdm(range(self.n_episodes_per_level), leave=True):
            kv_cache = {i: None for i in range(self.vec_env_size)}
            cum_reward = {i: 0 for i in range(self.vec_env_size)}
            ready_env_ids = np.arange(len(self.vec_envs))
            t = 0

            vec_obs = self.vec_envs.reset()  # np array of obs obj
            vec_obs = U.any_stack([obs["partial_rgb"] for obs in vec_obs], dim=0)[
                :, None, ...
            ]
            vec_obs = U.any_to_torch_tensor(vec_obs, device=policy.device)
            target_return_to_go = (
                torch.ones(
                    (self.vec_env_size, 1, 1), dtype=torch.float32, device=policy.device
                )
                * target_return_to_go_
            )
            prev_action, prev_reward, prev_dtoken = None, None, None

            while len(ready_env_ids) > 0:
                if t == 0 and n_th_episode == 0:
                    # literally the first step
                    cache = None
                    attention_mask = None
                elif t > 0 and n_th_episode == 0:
                    # in the first episode, but not the first step
                    cache = U.any_stack([kv_cache[i] for i in ready_env_ids], dim=0)
                    attention_mask = None
                elif t == 0 and n_th_episode > 0:
                    # first step in a new episode
                    cache = [
                        U.any_concat(list(episode_cache[i]), dim=1)
                        for i in ready_env_ids
                    ]
                    max_L = max(c[0][0].shape[1] for c in cache)
                    if cache_structure is None:
                        cache_structure = U.any_zeros_like(
                            U.any_slice(cache[0], np.s_[:, 0:1])
                        )  # (nh, 1, E)
                    # do left padding
                    cache = [
                        U.any_concat(
                            [cache_structure] * (max_L - c[0][0].shape[1]) + [c], dim=1
                        )
                        for c in cache
                    ]
                    attention_mask = [
                        torch.tensor(
                            [False] * (max_L - c[0][0].shape[1])
                            + [True] * c[0][0].shape[1],
                            dtype=torch.bool,
                            device=policy.device,
                        )
                        for c in cache
                    ]
                    cache = U.any_stack(cache, dim=0)  # (B, nh, L, E)
                    attention_mask = U.any_stack(attention_mask, dim=0)  # (B, L)
                else:
                    # not the first step in a new episode
                    past_episode_cache = [
                        U.any_concat(list(episode_cache[i]), dim=1)
                        for i in ready_env_ids
                    ]
                    max_L = max(c[0][0].shape[1] for c in past_episode_cache)
                    # do left padding
                    past_episode_cache = [
                        U.any_concat(
                            [cache_structure] * (max_L - c[0][0].shape[1]) + [c], dim=1
                        )
                        for c in past_episode_cache
                    ]
                    past_episode_attention_mask = [
                        torch.tensor(
                            [False] * (max_L - c[0][0].shape[1])
                            + [True] * c[0][0].shape[1],
                            dtype=torch.bool,
                            device=policy.device,
                        )
                        for c in past_episode_cache
                    ]
                    past_episode_cache = U.any_stack(
                        past_episode_cache, dim=0
                    )  # (B, nh, L, E)
                    past_episode_attention_mask = U.any_stack(
                        past_episode_attention_mask, dim=0
                    )  # (B, L)
                    current_episode_cache = U.any_stack(
                        [kv_cache[i] for i in ready_env_ids], dim=0
                    )
                    current_episode_attention_mask = torch.ones(
                        (len(ready_env_ids), current_episode_cache[0][0].shape[2]),
                        dtype=torch.bool,
                        device=policy.device,
                    )
                    cache = U.any_concat(
                        [past_episode_cache, current_episode_cache], dim=2
                    )  # (B, nh, L, E)
                    attention_mask = U.any_concat(
                        [past_episode_attention_mask, current_episode_attention_mask],
                        dim=1,
                    )  # (B, L)
                timestep = (torch.ones((1, 1), device=policy.device) * t).long()

                if attention_mask is not None:
                    attention_mask = U.any_concat(
                        [
                            attention_mask,
                            torch.ones(
                                (len(ready_env_ids), 2 if prev_action is None else 5),
                                dtype=torch.bool,
                                device=policy.device,
                            ),
                        ],
                        dim=1,
                    )
                actions, new_caches = policy.act(
                    return_to_go=target_return_to_go,
                    state=vec_obs,
                    timestep=timestep,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                    prev_dtoken=prev_dtoken,
                    past_kv=cache,
                    deterministic=self._deterministic,
                    attention_mask=attention_mask,
                )  # (B, 1), (B, nh, L, E)
                for idx, env_id in enumerate(ready_env_ids):
                    if n_th_episode == 0:
                        kv_cache[env_id] = U.any_slice(
                            new_caches, np.s_[idx]
                        )  # (nh, L, E)
                    else:
                        # only cache the current episode
                        kv_cache[env_id] = U.any_slice(
                            new_caches, np.s_[idx, :, -(2 + t * 5) :]
                        )
                prev_action = actions
                actions = U.any_to_numpy(actions)[:, 0]  # (B,)
                vec_obs, vec_reward, vec_done, vec_info = self.vec_envs.step(
                    action=actions, id=ready_env_ids
                )
                vec_obs = U.any_stack([obs["partial_rgb"] for obs in vec_obs], dim=0)[
                    :, None, ...
                ]
                vec_obs = U.any_to_torch_tensor(vec_obs, device=policy.device)
                prev_reward = U.any_to_torch_tensor(
                    vec_reward, device=policy.device
                ).unsqueeze(
                    1
                )  # (B, 1, 1)
                for i, reward in zip(ready_env_ids, vec_reward):
                    cum_reward[i] += reward
                prev_dtoken = (
                    torch.tensor(
                        [
                            cum_reward[i].item() >= target_return_to_go_
                            for i in ready_env_ids
                        ],
                        dtype=torch.bool,
                        device=policy.device,
                    )
                    .unsqueeze(1)
                    .long()
                )  # (B, 1)

                if np.any(vec_done):
                    terminated_env_local_ids = np.where(vec_done)[0]
                    terminated_env_global_ids = ready_env_ids[terminated_env_local_ids]
                    terminated_reward = vec_reward[terminated_env_local_ids]
                    for i, reward in zip(terminated_env_global_ids, terminated_reward):
                        eval_success[i, n_th_episode] = reward > 0

                    mask = np.ones_like(ready_env_ids, dtype=bool)
                    mask[terminated_env_local_ids] = False
                    ready_env_ids = ready_env_ids[mask]
                    target_return_to_go = target_return_to_go[mask]
                    vec_obs = vec_obs[mask]
                    prev_action = prev_action[mask]
                    prev_reward = prev_reward[mask]
                    prev_dtoken = prev_dtoken[mask]

                    for i in terminated_env_global_ids:
                        episode_cache[i].append(kv_cache[i])
                        kv_cache[i] = None

                t += 1
                # break when all envs are done
                if len(ready_env_ids) == 0:
                    break
        self._eval_success = eval_success

    def get_results(self) -> dict:
        return self._eval_success

    def _create_vec_envs(self):
        def _env_fn(level, seed):
            if isinstance(self.difficulty_levels, list):
                envs = []
                for n in self.difficulty_levels:
                    if "irreversible" in level:
                        self.cfg.difficulty = n * 0.1
                    elif "watermaze" in level:
                        self.cfg.spawn_radius = n
                    else:
                        self.cfg.room_num = n
                    env = make_env_func_batched(
                        self.cfg,
                        env_config=AttrDict(
                            worker_index=0, vector_index=0, env_id=seed
                        ),
                        render_mode="rgb",
                    )
                    envs.append(RGBImgObsWrapper(env))
                if self.use_auto_curriculum:
                    return AutoCurriculumWrapper(
                        num_levels=len(self.difficulty_levels),
                        consecutive_success=self.auto_curriculum_consecutive_success,
                        success_check_fn=self.success_check_fn,
                        envs=envs,
                        n_episodes_per_level=self.auto_curriculum_n_episodes_per_level,
                        action_repeat=self.action_repeat,
                    )

                else:
                    return ChainEnvWrapper(
                        envs,
                        self.n_episodes_per_level,
                        action_repeat=self.action_repeat,
                        in_context_episodes=self.in_context_episodes,
                    )
            env = make_env_func_batched(
                self.cfg,
                env_config=AttrDict(worker_index=0, vector_index=0, env_id=seed),
                render_mode="rgb",
            )
            return ChainEnvWrapper(
                [RGBImgObsWrapper(env)],
                self.n_episodes_per_level,
                action_repeat=self.action_repeat,
                in_context_episodes=self.in_context_episodes,
            )

        task = self.tasks[0]
        env_fns = [
            partial(_env_fn, task, self.seed + j) for j in range(self.vec_env_size)
        ]
        return ShmemVectorEnv(env_fns)

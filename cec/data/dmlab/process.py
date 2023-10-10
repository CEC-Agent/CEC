from __future__ import annotations
from math import ceil
from copy import deepcopy

import torch
import numpy as np
from einops import rearrange

import cec.utils as U


def prepare_sample(
    obs: dict[str, np.ndarray], action: dict[str, np.ndarray] | np.ndarray
):
    """
    Prepare trajectory data.
    """
    # discard terminal obs
    obs = U.any_slice(obs, np.s_[:-1])

    # images are in (L, H, W, C), transpose to (L, C, H, W)
    for k, v in obs.items():
        if v.ndim == 4 and v.shape[-1] == 3:
            obs[k] = rearrange(v, "L H W C -> L C H W")

    return obs, action


def collate_fn(
    samples_list: list[tuple[dict, dict | np.ndarray]],
    ctx_len: int = -1,
):
    """
    Collate a list of trajectories into a batch with shape (num_chunks, B, ctx_len, ...).

    samples_list: A list of tuple(obs, action), each with leading dimension L.
    """
    B = len(samples_list)
    _L_max = max([U.get_batch_size(obs) for obs, _ in samples_list])
    L_max = max([U.get_batch_size(action) for _, action in samples_list])

    # determine the max length after padding
    # it should be divisible by context_len and satisfy L_pad_max >= L_max
    if ctx_len != -1:
        assert ctx_len > 0
        L_pad_max = ceil(L_max / ctx_len) * ctx_len
    else:
        # when context_len is -1, context is simply the entire trajectory
        L_pad_max = L_max
        ctx_len = L_max

    # first pad each trajectory to L_pad_max in this batch
    # note that we slice instead of index to keep the first dim
    obs_structure = deepcopy(U.any_slice(samples_list[0][0], np.s_[0:1]))
    action_structure = deepcopy(U.any_slice(samples_list[0][1], np.s_[0:1]))
    padded_obs = U.any_stack(
        [
            U.any_concat(
                [sample[0]]
                + [U.any_zeros_like(obs_structure)]
                * (L_pad_max - U.get_batch_size(sample[0])),
                dim=0,
            )
            for sample in samples_list
        ],
        dim=0,
    )
    padded_action = U.any_stack(
        [
            U.any_concat(
                [sample[1]]
                + [U.any_zeros_like(action_structure)]
                * (L_pad_max - U.get_batch_size(sample[1])),
                dim=0,
            )
            for sample in samples_list
        ],
        dim=0,
    )

    # construct action_mask
    padded_action_mask = U.any_stack(
        [
            U.any_concat(
                [U.any_ones_like(action_structure)] * U.get_batch_size(sample[1])
                + [U.any_zeros_like(action_structure)]
                * (L_pad_max - U.get_batch_size(sample[1]))
            )
            for sample in samples_list
        ],
        dim=0,
    )

    # split obs and action into chunks
    n_chunks = L_pad_max // ctx_len if ctx_len != -1 else 1
    padded_obs = {
        k: U.any_stack(v, dim=0)
        for k, v in U.nested_np_split(padded_obs, n_chunks, axis=1).items()
    }
    if isinstance(padded_action, dict):
        padded_action = {
            k: U.any_stack(v, dim=0)
            for k, v in U.nested_np_split(padded_action, n_chunks, axis=1).items()
        }
    else:
        padded_action = U.any_stack(
            U.nested_np_split(padded_action, n_chunks, axis=1), dim=0
        )
    if isinstance(padded_action_mask, dict):
        padded_action_mask = {
            k: U.any_stack(v, dim=0)
            for k, v in U.nested_np_split(padded_action_mask, n_chunks, axis=1).items()
        }
    else:
        padded_action_mask = U.any_stack(
            U.nested_np_split(padded_action_mask, n_chunks, axis=1), dim=0
        )

    # convert to tensor
    padded_obs = U.any_to_datadict(padded_obs)
    padded_obs = padded_obs.to_torch_tensor()
    if isinstance(padded_action, dict):
        padded_action = U.any_to_datadict(padded_action)
        padded_action = padded_action.to_torch_tensor()
    else:
        padded_action = U.any_to_torch_tensor(padded_action)
    if isinstance(padded_action_mask, dict):
        padded_action_mask = U.any_to_datadict(padded_action_mask)
        padded_action_mask.map_structure(lambda x: x.astype(bool), inplace=True)
        padded_action_mask = padded_action_mask.to_torch_tensor()
    else:
        padded_action_mask = U.any_to_torch_tensor(padded_action_mask, dtype=torch.bool)

    return padded_obs, padded_action, padded_action_mask


def collate_fn_AT(
    samples_list: list[tuple[dict, dict | np.ndarray, int]],
    ctx_len: int = -1,
):
    """
    Collate a list of trajectories into a batch with shape (num_chunks, B, ctx_len, ...).

    samples_list: A list of tuple(obs, action), each with leading dimension L.
    """
    _L_max = max([U.get_batch_size(obs) for obs, _, _ in samples_list])
    L_max = max([U.get_batch_size(action) for _, action, _ in samples_list])

    # determine the max length after padding
    # it should be divisible by context_len and satisfy L_pad_max >= L_max
    if ctx_len != -1:
        assert ctx_len > 0
        L_pad_max = ceil(L_max / ctx_len) * ctx_len
    else:
        # when context_len is -1, context is simply the entire trajectory
        L_pad_max = L_max
        ctx_len = L_max

    # first pad each trajectory to L_pad_max in this batch
    # note that we slice instead of index to keep the first dim
    obs_structure = deepcopy(U.any_slice(samples_list[0][0], np.s_[0:1]))
    action_structure = deepcopy(U.any_slice(samples_list[0][1], np.s_[0:1]))
    padded_obs = U.any_stack(
        [
            U.any_concat(
                [sample[0]]
                + [U.any_zeros_like(obs_structure)]
                * (L_pad_max - U.get_batch_size(sample[0])),
                dim=0,
            )
            for sample in samples_list
        ],
        dim=0,
    )
    padded_action = U.any_stack(
        [
            U.any_concat(
                [sample[1]]
                + [U.any_zeros_like(action_structure)]
                * (L_pad_max - U.get_batch_size(sample[1])),
                dim=0,
            )
            for sample in samples_list
        ],
        dim=0,
    )

    # construct action_mask
    padded_action_mask = U.any_stack(
        [
            U.any_concat(
                [U.any_zeros_like(action_structure)]
                * (U.get_batch_size(sample[1]) - sample[2])
                + [U.any_ones_like(action_structure)] * sample[2]
                + [U.any_zeros_like(action_structure)]
                * (L_pad_max - U.get_batch_size(sample[1]))
            )
            for sample in samples_list
        ],
        dim=0,
    )

    # split obs and action into chunks
    n_chunks = L_pad_max // ctx_len if ctx_len != -1 else 1
    padded_obs = {
        k: U.any_stack(v, dim=0)
        for k, v in U.nested_np_split(padded_obs, n_chunks, axis=1).items()
    }
    if isinstance(padded_action, dict):
        padded_action = {
            k: U.any_stack(v, dim=0)
            for k, v in U.nested_np_split(padded_action, n_chunks, axis=1).items()
        }
    else:
        padded_action = U.any_stack(
            U.nested_np_split(padded_action, n_chunks, axis=1), dim=0
        )
    if isinstance(padded_action_mask, dict):
        padded_action_mask = {
            k: U.any_stack(v, dim=0)
            for k, v in U.nested_np_split(padded_action_mask, n_chunks, axis=1).items()
        }
    else:
        padded_action_mask = U.any_stack(
            U.nested_np_split(padded_action_mask, n_chunks, axis=1), dim=0
        )

    # convert to tensor
    padded_obs = U.any_to_datadict(padded_obs)
    padded_obs = padded_obs.to_torch_tensor()
    if isinstance(padded_action, dict):
        padded_action = U.any_to_datadict(padded_action)
        padded_action = padded_action.to_torch_tensor()
    else:
        padded_action = U.any_to_torch_tensor(padded_action)
    if isinstance(padded_action_mask, dict):
        padded_action_mask = U.any_to_datadict(padded_action_mask)
        padded_action_mask.map_structure(lambda x: x.astype(bool), inplace=True)
        padded_action_mask = padded_action_mask.to_torch_tensor()
    else:
        padded_action_mask = U.any_to_torch_tensor(padded_action_mask, dtype=torch.bool)

    return padded_obs, padded_action, padded_action_mask

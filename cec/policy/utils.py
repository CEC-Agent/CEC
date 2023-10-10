from __future__ import annotations

from typing import Callable, Literal

import gym

from cec.nn.distributions import CategoricalNet


__all__ = ["make_action_head"]


def make_action_head(
    *,
    action_space: gym.Space,
    input_dim: int,
    hidden_dim: int,
    hidden_depth: int,
    activation: str | Callable = "relu",
    norm_type: Literal["batchnorm", "layernorm"] | None = None,
    last_layer_gain: float | None = 0.01,
):
    assert isinstance(action_space, gym.Space)

    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
        return CategoricalNet(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
            last_layer_gain=last_layer_gain,
        )
    else:
        raise NotImplementedError(f"Unsupported action space: {action_space}")

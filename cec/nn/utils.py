from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding as _Embedding


class Embedding(_Embedding):
    @property
    def output_dim(self):
        return self.embedding_dim


def build_mlp(
    input_dim,
    *,
    hidden_dim: int,
    output_dim: int,
    hidden_depth: int = None,
    num_layers: int = None,
    activation: str | Callable = "relu",
    weight_init: str | Callable = "orthogonal",
    bias_init="zeros",
    norm_type: Literal["batchnorm", "layernorm"] | None = None,
    add_input_activation: bool | str | Callable = False,
    add_input_norm: bool = False,
    add_output_activation: bool | str | Callable = False,
    add_output_norm: bool = False,
) -> nn.Sequential:
    """
    In other popular RL implementations, tanh is typically used with orthogonal
    initialization, which may perform better than ReLU.

    Args:
        norm_type: None, "batchnorm", "layernorm", applied to intermediate layers
        add_input_activation: whether to add a nonlinearity to the input _before_
            the MLP computation. This is useful for processing a feature from a preceding
            image encoder, for example. Image encoder typically has a linear layer
            at the end, and we don't want the MLP to immediately stack another linear
            layer on the input features.
            - True to add the same activation as the rest of the MLP
            - str to add an activation of a different type.
        add_input_norm: see `add_input_activation`, whether to add a normalization layer
            to the input _before_ the MLP computation.
            values: True to add the `norm_type` to the input
        add_output_activation: whether to add a nonlinearity to the output _after_ the
            MLP computation.
            - True to add the same activation as the rest of the MLP
            - str to add an activation of a different type.
        add_output_norm: see `add_output_activation`, whether to add a normalization layer
            _after_ the MLP computation.
            values: True to add the `norm_type` to the input
    """
    assert (hidden_depth is None) != (num_layers is None), (
        "Either hidden_depth or num_layers must be specified, but not both. "
        "num_layers is defined as hidden_depth+1"
    )
    if hidden_depth is not None:
        assert hidden_depth >= 0
    if num_layers is not None:
        assert num_layers >= 1
    act_layer = get_activation(activation)

    weight_init = get_initializer(weight_init, activation)
    bias_init = get_initializer(bias_init, activation)

    if norm_type is not None:
        norm_type = norm_type.lower()

    if not norm_type:
        norm_type = nn.Identity
    elif norm_type == "batchnorm":
        norm_type = nn.BatchNorm1d
    elif norm_type == "layernorm":
        norm_type = nn.LayerNorm
    else:
        raise ValueError(f"Unsupported norm layer: {norm_type}")

    hidden_depth = num_layers - 1 if hidden_depth is None else hidden_depth
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), norm_type(hidden_dim), act_layer()]
        for i in range(hidden_depth - 1):
            mods += [
                nn.Linear(hidden_dim, hidden_dim),
                norm_type(hidden_dim),
                act_layer(),
            ]
        mods.append(nn.Linear(hidden_dim, output_dim))

    if add_input_norm:
        mods = [norm_type(input_dim)] + mods
    if add_input_activation:
        if add_input_activation is not True:
            act_layer = get_activation(add_input_activation)
        mods = [act_layer()] + mods
    if add_output_norm:
        mods.append(norm_type(output_dim))
    if add_output_activation:
        if add_output_activation is not True:
            act_layer = get_activation(add_output_activation)
        mods.append(act_layer())

    for mod in mods:
        if isinstance(mod, nn.Linear):
            weight_init(mod.weight)
            bias_init(mod.bias)

    return nn.Sequential(*mods)


def get_activation(activation: str | Callable | None) -> Callable:
    if not activation:
        return nn.Identity
    elif callable(activation):
        return activation
    ACT_LAYER = {
        "tanh": nn.Tanh,
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(inplace=True),
        "swish": lambda: nn.SiLU(inplace=True),  # SiLU is alias for Swish
        "sigmoid": nn.Sigmoid,
        "elu": lambda: nn.ELU(inplace=True),
        "gelu": nn.GELU,
    }
    activation = activation.lower()
    assert activation in ACT_LAYER, f"Supported activations: {ACT_LAYER.keys()}"
    return ACT_LAYER[activation]


def get_initializer(method: str | Callable, activation: str) -> Callable:
    if isinstance(method, str):
        assert hasattr(
            nn.init, f"{method}_"
        ), f"Initializer nn.init.{method}_ does not exist"
        if method == "orthogonal":
            try:
                gain = nn.init.calculate_gain(activation)
            except ValueError:
                gain = 1.0
            return lambda x: nn.init.orthogonal_(x, gain=gain)
        else:
            return getattr(nn.init, f"{method}_")
    else:
        assert callable(method)
        return method


class FanInInitReLULayer(nn.Module):
    """Implements a slightly modified init that correctly produces std 1 outputs given ReLU activation"""

    def __init__(
        self,
        inchan: int,
        outchan: int,
        *layer_args,
        layer_type: str = "conv",
        init_scale: int | float = 1,
        batch_norm: bool = False,
        batch_norm_kwargs: dict | None = None,
        group_norm_groups: int | None = None,
        layer_norm: bool = False,
        use_activation: bool = True,
        **layer_kwargs,
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if batch_norm:
            batch_norm_kwargs = batch_norm_kwargs or {}
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)

        layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear)[layer_type]
        self.layer = layer(
            inchan, outchan, bias=self.norm is None, *layer_args, **layer_kwargs
        )

        # Init Weights (Fan-In)
        self.layer.weight.data *= init_scale / self.layer.weight.norm(
            dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
        )
        # Init Bias
        if self.layer.bias is not None:
            self.layer.bias.data *= 0

        self.use_activation = use_activation

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x


class NormedLinear(nn.Linear):
    def __init__(self, *args, scale: int | float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.data *= scale / self.weight.norm(dim=1, p=2, keepdim=True)
        if kwargs.get("bias", True):
            self.bias.data *= 0


def sequential_fwd(layers, x, *args):
    for i, layer in enumerate(layers):
        x = layer(x, *args)
    return x


def split_heads(x: torch.Tensor, n_heads: int):
    """
    x: (b, t, e)
    """
    assert x.ndim == 3
    b, t, e = x.shape
    assert e % n_heads == 0
    q = e // n_heads
    x_bthq = x.reshape((b, t, n_heads, q))
    x_bhtq = torch.transpose(x_bthq, 1, 2)
    x_Btq = x_bhtq.reshape((b * n_heads, t, q))
    return x_Btq


def merge_heads(x: torch.Tensor, n_heads: int):
    """
    x: (b * n_heads, t, q)
    """
    assert x.ndim == 3
    B, t, q = x.shape
    assert B % n_heads == 0
    b = B // n_heads
    x_bhtq = x.reshape((b, n_heads, t, q))
    x_bthq = torch.transpose(x_bhtq, 1, 2)
    x_btq = x_bthq.reshape((b, t, n_heads * q))
    return x_btq


def _banded_repeat(x, t):
    """
    Repeats x with a shift.
    For example (ignoring the batch dimension):

    _banded_repeat([A B C D E], 4)
    =
    [D E 0 0 0]
    [C D E 0 0]
    [B C D E 0]
    [A B C D E]
    """
    b, T = x.shape
    x = torch.cat([x, x.new_zeros(b, t - 1)], dim=1)
    result = x.unfold(1, T, 1).flip(1)
    return result


def bandify(b_nd, t, T):
    """
    b_nd -> D_ntT, where
        "n" indexes over basis functions
        "d" indexes over time differences
        "t" indexes over output time
        "T" indexes over input time
        only t >= T is nonzero
    B_ntT[n, t, T] = b_nd[n, t - T]
    """
    nbasis, bandsize = b_nd.shape
    b_nd = b_nd[:, torch.arange(bandsize - 1, -1, -1)]
    if bandsize >= T:
        b_nT = b_nd[:, -T:]
    else:
        b_nT = torch.cat([b_nd.new_zeros(nbasis, T - bandsize), b_nd], dim=1)
    D_tnT = _banded_repeat(b_nT, t)
    return D_tnT

from __future__ import annotations

import math
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F

from cec.nn.utils import sequential_fwd, FanInInitReLULayer


class BasicBlock(nn.Module):
    """
    Residual basic block, as in ImpalaCNN. Preserves channel number and shape
    :param inchan: number of input channels
    :param init_scale: weight init scale multiplier
    """

    def __init__(
        self,
        inchan: int,
        init_scale: int | float = 1,
        init_norm_kwargs: dict | None = None,
    ):
        super().__init__()
        self.inchan = inchan
        s = math.sqrt(init_scale)
        init_norm_kwargs = init_norm_kwargs or {}
        self.conv0 = FanInInitReLULayer(
            self.inchan,
            self.inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            **init_norm_kwargs,
        )
        self.conv1 = FanInInitReLULayer(
            self.inchan,
            self.inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            **init_norm_kwargs,
        )

    def forward(self, x):
        x = x + self.conv1(self.conv0(x))
        return x


class DownStack(nn.Module):
    """
    Downsampling stack from Impala CNN.
    """

    def __init__(
        self,
        inchan: int,
        nblock: int,
        outchan: int,
        init_scale: float = 1,
        pool: bool = True,
        post_pool_groups: int | None = None,
        init_norm_kwargs: dict | None = None,
        first_conv_norm=False,
    ):
        super().__init__()
        init_norm_kwargs = init_norm_kwargs or {}
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        first_conv_init_kwargs = deepcopy(init_norm_kwargs)
        if not first_conv_norm:
            first_conv_init_kwargs["group_norm_groups"] = None
            first_conv_init_kwargs["batch_norm"] = False
        self.firstconv = FanInInitReLULayer(
            inchan,
            outchan,
            kernel_size=3,
            padding=1,
            **first_conv_init_kwargs,
        )
        self.post_pool_groups = post_pool_groups
        if post_pool_groups is not None:
            self.n = nn.GroupNorm(post_pool_groups, outchan)
        self.blocks = nn.ModuleList(
            [
                BasicBlock(
                    outchan,
                    init_scale=init_scale / math.sqrt(nblock),
                    init_norm_kwargs=init_norm_kwargs,
                )
                for _ in range(nblock)
            ]
        )

    def forward(self, x):
        x = self.firstconv(x)
        if self.pool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            if self.post_pool_groups is not None:
                x = self.n(x)
        x = sequential_fwd(self.blocks, x)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if self.pool:
            return self.outchan, (h + 1) // 2, (w + 1) // 2
        else:
            return self.outchan, h, w

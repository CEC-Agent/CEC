from __future__ import annotations

import math
import torch
import torch.nn as nn
from einops import rearrange

from cec.nn.utils import sequential_fwd, FanInInitReLULayer
from cec.nn.feature.image.convnet.blocks import DownStack


class ImpalaConvNet(nn.Module):
    def __init__(
        self,
        inshape: list[int, int, int],
        chans: list[int],
        outsize: int,
        nblock: int,
        init_norm_kwargs: dict | None = None,
        dense_init_norm_kwargs: dict | None = None,
        first_conv_norm=False,
        **kwargs,
    ):
        super().__init__()
        init_norm_kwargs = init_norm_kwargs or {}
        dense_init_norm_kwargs = dense_init_norm_kwargs or {}
        c, h, w = inshape
        curshape = (c, h, w)
        self.stacks = nn.ModuleList()
        for i, outchan in enumerate(chans):
            stack = DownStack(
                curshape[0],
                nblock=nblock,
                outchan=outchan,
                init_scale=math.sqrt(len(chans)),
                init_norm_kwargs=init_norm_kwargs,
                first_conv_norm=first_conv_norm if i == 0 else True,
                **kwargs,
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)

        self.dense = FanInInitReLULayer(
            math.prod(curshape),
            outsize,
            layer_type="linear",
            init_scale=1.4,
            **dense_init_norm_kwargs,
        )
        self.outsize = outsize

    def forward(self, x):
        b, t = x.shape[:2]
        x = x.reshape(b * t, *x.shape[-3:])
        # x = rearrange(x, "b h w c -> b c h w") # we don't need this for BabyAI
        x = sequential_fwd(self.stacks, x)
        x = x.reshape(b, t, *x.shape[1:])
        x = torch.flatten(x, start_dim=-3, end_dim=-1)
        x = self.dense(x)
        return x

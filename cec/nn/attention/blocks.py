from __future__ import annotations

import torch.nn as nn

from cec.nn.utils import FanInInitReLULayer
from cec.nn.attention.layers import CausalMaskedAttention, CausalMaskedCrossAttention


class TransformerBlocks(nn.Module):
    def __init__(
        self,
        hidden_size,
        timesteps,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        attention_heads=8,
        attention_memory_size=2048,
        n_block=2,
        subsample_every=1,
    ):
        super().__init__()
        init_scale = n_block**-0.5
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    timesteps=timesteps,
                    init_scale=init_scale,
                    use_pointwise_layer=use_pointwise_layer,
                    pointwise_ratio=pointwise_ratio,
                    pointwise_use_activation=pointwise_use_activation,
                    attention_heads=attention_heads,
                    attention_memory_size=attention_memory_size,
                    subsample_every=subsample_every,
                )
                for _ in range(n_block)
            ]
        )

    def forward(self, x, first, state, prompt=None):
        state_out = []
        assert len(state) == len(
            self.blocks
        ), f"Length of state {len(state)} did not match length of blocks {len(self.blocks)}"
        for block, _s_in in zip(self.blocks, state):
            x, _s_o = block(x, first, _s_in, prompt)
            state_out.append(_s_o)
        return x, state_out

    def initial_state(self, batchsize, device=None):
        return [b.r.initial_state(batchsize, device) for b in self.blocks]


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        timesteps,
        init_scale: int | float = 1,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        attention_heads=8,
        attention_memory_size=2048,
        subsample_every=1,
    ):
        super().__init__()
        self.use_pointwise_layer = use_pointwise_layer
        if use_pointwise_layer:
            self.mlp0 = FanInInitReLULayer(
                hidden_size,
                hidden_size * pointwise_ratio,
                init_scale=1,
                layer_type="linear",
                layer_norm=True,
            )
            self.mlp1 = FanInInitReLULayer(
                hidden_size * pointwise_ratio,
                hidden_size,
                init_scale=init_scale * 2**-0.5,
                layer_type="linear",
                use_activation=pointwise_use_activation,
            )

        self.pre_r_ln = nn.LayerNorm(hidden_size)
        self.r = CausalMaskedAttention(
            input_size=hidden_size,
            timesteps=timesteps,
            memory_size=attention_memory_size,
            n_heads=attention_heads,
            init_scale=init_scale * 2**-0.5,
            norm="layer",
            use_muP_factor=True,
        )
        self.subsample_every = subsample_every

    def forward(self, x, first, state, prompt=None, subsample_every=None):
        x = self.pre_r_ln(x)
        x, state_out = self.r(x, first, state, subsample_every=self.subsample_every)
        if self.use_pointwise_layer:
            # Residual MLP
            residual = x
            x = self.mlp1(self.mlp0(x))
            x = x + residual
        return x, state_out


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        timesteps,
        init_scale: int | float = 1,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        attention_heads=8,
        attention_memory_size=2048,
    ):
        super().__init__()
        self.use_pointwise_layer = use_pointwise_layer
        if use_pointwise_layer:
            self.mlp0 = FanInInitReLULayer(
                hidden_size,
                hidden_size * pointwise_ratio,
                init_scale=1,
                layer_type="linear",
                layer_norm=True,
            )
            self.mlp1 = FanInInitReLULayer(
                hidden_size * pointwise_ratio,
                hidden_size,
                init_scale=init_scale * 2**-0.5,
                layer_type="linear",
                use_activation=pointwise_use_activation,
            )

        self.pre_r_ln_x = nn.LayerNorm(hidden_size)
        self.pre_r_ln_y = nn.LayerNorm(hidden_size)
        self.r = CausalMaskedCrossAttention(
            input_size=hidden_size,
            timesteps=timesteps,
            memory_size=attention_memory_size,
            n_heads=attention_heads,
            init_scale=init_scale * 2**-0.5,
            norm="layer",
            use_muP_factor=True,
        )
        self.state = None
        self.output_dim = hidden_size

    def forward(self, x, first, state, y=None):
        if self.state is None or self.state[0].size(0) != x.size(0) or first[0][0] == 1:
            # if first[0][0] == 1:
            self.state = self.r.initial_state(batchsize=x.size(0), device=x.device)
        x = self.pre_r_ln_x(x)
        y = self.pre_r_ln_y(y)
        x, state_out = self.r(x, y, first, state or self.state)
        if self.use_pointwise_layer:
            # Residual MLP
            residual = x
            x = self.mlp1(self.mlp0(x))
            x = x + residual
        # Update internal state
        self.state = state_out
        return x, state_out

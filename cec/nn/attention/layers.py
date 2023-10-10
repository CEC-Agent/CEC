from __future__ import annotations

import math
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

import torch
import torch.nn as nn

from cec.nn.attention.op import attention, get_mask
from cec.nn.tree_utils import tree_map
from cec.nn.utils import NormedLinear, split_heads, merge_heads, bandify


Q_SCALE = 0.1
K_SCALE = 0.2
V_SCALE = 1.0
PROJ_SCALE = 1.0
MLP0_SCALE = 1.0
MLP1_SCALE = 1.0
R_SCALE = 0.1
B_SCALE = 0.2


class AttentionLayerBase(nn.Module):
    def __init__(
        self,
        *,
        scale: int | float,
        x_size: int,
        c_size: int,
        qk_size: int,
        v_size: int,
        n_heads: int,
        rel_attn=False,
        n_basis: int | None = None,
        max_len: int,
    ):
        super().__init__()
        self.x_size = x_size
        self.c_size = c_size
        s = math.sqrt(scale)
        self.q_layer = NormedLinear(x_size, qk_size, scale=Q_SCALE)
        self.k_layer = NormedLinear(c_size, qk_size, scale=K_SCALE, bias=False)
        self.v_layer = NormedLinear(c_size, v_size, scale=V_SCALE * s, bias=False)
        self.proj_layer = NormedLinear(v_size, x_size, scale=PROJ_SCALE * s)
        self.rel_attn = rel_attn
        self.max_len = max_len
        if rel_attn:
            n_basis = n_basis or 10
            self.r_layer = NormedLinear(x_size, n_basis * n_heads, scale=R_SCALE)
            self.b_nd = nn.Parameter(torch.randn(n_basis, max_len) * B_SCALE)

    def relattn_logits(self, X_bte, T):
        R_btn = self.r_layer(X_bte)  # (batch, t, n_basis * n_heads)
        R_btn = split_heads(R_btn, self.n_heads)  # (batch * n_heads, t, n_basis)
        t = R_btn.shape[1]
        D_ntT = bandify(self.b_nd, t, T)  # (n_basis, t, T)
        extra_btT = torch.einsum("btn,ntp->btp", R_btn, D_ntT)
        return extra_btT


class SelfAttentionLayer(AttentionLayerBase):
    def __init__(
        self,
        x_size: int,
        max_len: int,
        n_heads: int,
        scale: int | float,
        norm: str,
        cache_keep_len: int | None = None,
        rel_attn: bool = False,
        use_muP_factor: bool = False,
        n_basis: int | None = None,
    ):
        super().__init__(
            scale=scale,
            x_size=x_size,
            c_size=x_size,
            qk_size=x_size,
            v_size=x_size,
            n_heads=n_heads,
            rel_attn=rel_attn,
            n_basis=n_basis,
            max_len=max_len,
        )
        if norm == "none":
            self.ln_x = lambda x: x
        elif norm == "layer":
            self.ln_x = nn.LayerNorm(x_size)
        cache_keep_len = cache_keep_len or max_len
        self.cache_keep_len = cache_keep_len
        self.use_muP_factor = use_muP_factor
        self.n_heads = n_heads

    def residual(self, X_bte, state, mask: torch.Tensor | None = None):
        X_bte = self.ln_x(X_bte)
        Q_bte = self.q_layer(X_bte)
        K_bte = self.k_layer(X_bte)
        V_bte = self.v_layer(X_bte)
        if state:
            state, K_bte, V_bte = self.update_state(state, K_bte, V_bte)
        Q_bte = split_heads(Q_bte, self.n_heads)
        K_bte = split_heads(K_bte, self.n_heads)
        V_bte = split_heads(V_bte, self.n_heads)
        extra_btT = (
            self.relattn_logits(X_bte, K_bte.shape[1]) if self.rel_attn else None
        )
        A_bte = attention(
            Q_bte,
            K_bte,
            V_bte,
            mask=mask,
            extra_btT=extra_btT,
            use_muP_factor=self.use_muP_factor,
        )
        A_bte = merge_heads(A_bte, self.n_heads)
        Aproj_bte = self.proj_layer(A_bte)
        return Aproj_bte, state

    def forward(self, X_bte, state, mask: torch.Tensor | None = None):
        R_bte, state = self.residual(X_bte, state, mask)
        return X_bte + R_bte, state

    def stateless_forward(self, X_bte, mask: torch.Tensor | None = None):
        out_bte, _state = self.forward(X_bte, None, mask)
        return out_bte

    def update_state(self, state, K_bte, V_bte):
        def append(prev, new):
            """
            Given `prev` keys from cache, and `new` keys,
            returns (cache, full), where
            - cache goes into the output state, length chosen so that on the
                next timestep, there are enough cached timesteps to get the full
                context of lenth self.maxlen.
            - full is used for the current forward pass, with length chosen so
                that the first timestep new[:, 0] gets to see a context of
                self.maxlen.
            """
            tprev = prev.shape[1]
            startfull = max(tprev - self.cache_keep_len, 0)
            full = torch.cat([prev[:, startfull:], new], dim=1)
            outstate = full[:, max(full.shape[1] - (self.cache_keep_len), 0) :]
            # To see that the preceding slicing is correct, consider the case
            # that maxlen==1. Then `full` only consists of `new`, and
            # `outstate` is empty
            return outstate, full

        instate_K, instate_V = state
        outstate_K, K_bte = append(instate_K, K_bte)
        outstate_V, V_bte = append(instate_V, V_bte)
        assert outstate_K.shape[-2] <= self.cache_keep_len
        return (outstate_K, outstate_V), K_bte, V_bte

    def initial_state(self, batchsize, initial_T=0):
        return (
            torch.zeros((batchsize, initial_T, self.x_size), dtype=torch.float16),
            torch.zeros((batchsize, initial_T, self.x_size), dtype=torch.float16),
        )


class CausalMaskedAttention(nn.Module):
    """
    Transformer self-attention layer that removes frames from previous episodes from the hidden state under certain constraints.

    The constraints are:
    - The "first" flag can only be true for the first timestep of each batch. An assert will fire if other timesteps have first = True.

    input_size: The dimension of the input (which also happens to be the size of the output)
    memory_size: The number of frames to keep in the inner state. Note that when attending, we will be able to attend
                 to both the frames in the inner state (which presumably won't have gradients anymore) and the frames
                 in the batch. "mask" for some additional considerations on this.
    n_heads: The number of attention heads to use. Note that we will split the input into this number of heads, so
           input_size needs to be divisible by heads.
    timesteps: number of timesteps with which we'll be taking gradient


    Apply a normal causal mask but solves the following minor problem:
        if you have a state of length 128 and a batch of 128 frames, then the first frame of your batch will be able to
        attend to 128 previous frames, but the last one will be able to attend to 255 previous frames. In this example,
        "clipped_causal" will make it so that the last frame can only attend to 128 previous frames, so that there is no
        bias coming from the position in the batch. None simply allows you to attend to any frame in the state + batch,
        which means you can also attend to future frames.
    """

    def __init__(
        self,
        input_size: int,
        memory_size: int,
        n_heads: int,
        timesteps: int,
        init_scale=1,
        norm="layer",
        use_muP_factor=False,
    ):
        super().__init__()

        assert memory_size >= 0

        self.max_len = memory_size - timesteps
        self.n_heads = n_heads

        self.orc_block = SelfAttentionLayer(
            x_size=input_size,
            max_len=self.max_len,
            n_heads=n_heads,
            scale=init_scale,
            norm=norm,
            cache_keep_len=self.max_len,
            rel_attn=True,
            use_muP_factor=use_muP_factor,
        )

        self.stateless = False

    def initial_state(self, batchsize: int, device=None):
        """Return the initial state mask (None) and the initial state of the transformer (zerod out keys and queries)"""
        state = self.orc_block.initial_state(batchsize, initial_T=self.max_len)
        state_mask = None
        if device is not None:
            state = tree_map(lambda x: x.to(device), state)
        return state_mask, state

    def forward(self, input_bte, first_bt, state, subsample_every):
        state_mask, xf_state = state
        t = first_bt.shape[1]
        new_mask, state_mask = get_mask(
            first_b11=first_bt[:, [[0]]],
            state_mask=state_mask,
            t=t,
            T=t + self.max_len,
            maxlen=self.max_len,
            heads=self.n_heads,
            device=input_bte.device,
            subsample_every=subsample_every,
            stateless=self.stateless,
        )
        output, xf_state = self.orc_block(input_bte, xf_state, new_mask)

        return output, (state_mask, xf_state)


# Gated Cross Attention. Adapted from https://github.com/lucidrains/flamingo-pytorch


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class MaskedCrossAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, media):
        b, t, m = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = torch.einsum("... i d, ... j d -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.output_dim = dim
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x,  # q
        media,  # kv, encoded by perceiver resample - (batch, time, latents, dim)
        first_mask=None,
    ):
        x = self.attn(x, media) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x


# Gated cross attention based on CausalMaskedAttention


class CrossAttentionLayer(AttentionLayerBase):
    def __init__(
        self,
        x_size: int,
        max_len: int,
        n_heads: int,
        scale: int | float,
        norm: str,
        cache_keep_len: int | None = None,
        rel_attn: bool = False,
        use_muP_factor: bool = False,
        n_basis: int | None = None,
    ):
        super().__init__(
            scale=scale,
            x_size=x_size,
            c_size=x_size,
            qk_size=x_size,
            v_size=x_size,
            n_heads=n_heads,
            rel_attn=rel_attn,
            n_basis=n_basis,
            max_len=max_len,
        )
        if norm == "none":
            self.ln_x = lambda x: x
            self.ln_y = lambda x: x
        elif norm == "layer":
            self.ln_x = nn.LayerNorm(x_size)
            self.ln_y = nn.LayerNorm(x_size)  # assume x_size == y_size
        cache_keep_len = cache_keep_len or max_len
        self.cache_keep_len = cache_keep_len
        self.use_muP_factor = use_muP_factor
        self.n_heads = n_heads
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def residual(self, X_bte, Y_bte, state, mask: torch.Tensor | None = None):
        X_bte = self.ln_x(X_bte)
        Y_bte = self.ln_y(Y_bte)
        Q_bte = self.q_layer(X_bte)
        K_bte = self.k_layer(Y_bte)
        V_bte = self.v_layer(Y_bte)
        if state:
            state, K_bte, V_bte = self.update_state(state, K_bte, V_bte)
        Q_bte = split_heads(Q_bte, self.n_heads)
        K_bte = split_heads(K_bte, self.n_heads)
        V_bte = split_heads(V_bte, self.n_heads)
        extra_btT = (
            self.relattn_logits(X_bte, K_bte.shape[1]) if self.rel_attn else None
        )  # or use Y_bte for extra_btT?
        A_bte = attention(
            Q_bte,
            K_bte,
            V_bte,
            mask=mask,
            extra_btT=extra_btT,
            use_muP_factor=self.use_muP_factor,
        )
        A_bte = merge_heads(A_bte, self.n_heads)
        return A_bte, state

    def forward(self, X_bte, Y_bte, state, mask: torch.Tensor | None = None):
        A_bte, state = self.residual(X_bte, Y_bte, state, mask)
        # add tanh gating
        A_bte = A_bte * self.attn_gate.tanh() + X_bte
        # FF
        Aproj_bte = self.proj_layer(A_bte)
        # add tanh gating
        R_bte = Aproj_bte * self.ff_gate.tanh() + A_bte
        # R_bte = Aproj_bte + X_bte # orginal, no gating
        return R_bte, state

    def stateless_forward(self, X_bte, Y_bte, mask: torch.Tensor | None = None):
        out_bte, _state = self.forward(X_bte, Y_bte, None, mask)
        return out_bte

    def update_state(self, state, K_bte, V_bte):
        def append(prev, new):
            """
            Given `prev` keys from cache, and `new` keys,
            returns (cache, full), where
            - cache goes into the output state, length chosen so that on the
                next timestep, there are enough cached timesteps to get the full
                context of lenth self.maxlen.
            - full is used for the current forward pass, with length chosen so
                that the first timestep new[:, 0] gets to see a context of
                self.maxlen.
            """
            tprev = prev.shape[1]
            startfull = max(tprev - self.cache_keep_len, 0)
            full = torch.cat([prev[:, startfull:], new], dim=1)
            outstate = full[:, max(full.shape[1] - (self.cache_keep_len), 0) :]
            # To see that the preceding slicing is correct, consider the case
            # that maxlen==1. Then `full` only consists of `new`, and
            # `outstate` is empty
            return outstate, full

        instate_K, instate_V = state
        outstate_K, K_bte = append(instate_K, K_bte)
        outstate_V, V_bte = append(instate_V, V_bte)
        assert outstate_K.shape[-2] <= self.cache_keep_len
        return (outstate_K, outstate_V), K_bte, V_bte

    def initial_state(self, batchsize, initial_T=0):
        return (
            torch.zeros((batchsize, initial_T, self.x_size), dtype=torch.float16),
            torch.zeros((batchsize, initial_T, self.x_size), dtype=torch.float16),
        )


class CausalMaskedCrossAttention(nn.Module):
    """
    Transformer self-attention layer that removes frames from previous episodes from the hidden state under certain constraints.

    The constraints are:
    - The "first" flag can only be true for the first timestep of each batch. An assert will fire if other timesteps have first = True.

    input_size: The dimension of the input (which also happens to be the size of the output)
    memory_size: The number of frames to keep in the inner state. Note that when attending, we will be able to attend
                 to both the frames in the inner state (which presumably won't have gradients anymore) and the frames
                 in the batch. "mask" for some additional considerations on this.
    n_heads: The number of attention heads to use. Note that we will split the input into this number of heads, so
           input_size needs to be divisible by heads.
    timesteps: number of timesteps with which we'll be taking gradient


    Apply a normal causal mask but solves the following minor problem:
        if you have a state of length 128 and a batch of 128 frames, then the first frame of your batch will be able to
        attend to 128 previous frames, but the last one will be able to attend to 255 previous frames. In this example,
        "clipped_causal" will make it so that the last frame can only attend to 128 previous frames, so that there is no
        bias coming from the position in the batch. None simply allows you to attend to any frame in the state + batch,
        which means you can also attend to future frames.
    """

    def __init__(
        self,
        input_size: int,
        memory_size: int,
        n_heads: int,
        timesteps: int,
        init_scale=1,
        # norm="none",
        norm="layer",
        use_muP_factor=False,
    ):
        super().__init__()

        assert memory_size >= 0

        self.max_len = memory_size - timesteps
        self.n_heads = n_heads

        self.orc_block = CrossAttentionLayer(
            x_size=input_size,
            max_len=self.max_len,
            n_heads=n_heads,
            scale=init_scale,
            norm=norm,
            cache_keep_len=self.max_len,
            rel_attn=True,
            use_muP_factor=use_muP_factor,
        )

    def initial_state(self, batchsize: int, device=None):
        """Return the initial state mask (None) and the initial state of the transformer (zerod out keys and queries)"""
        state = self.orc_block.initial_state(batchsize, initial_T=self.max_len)
        state_mask = None
        if device is not None:
            state = tree_map(lambda x: x.to(device), state)
        return state_mask, state

    def forward(self, X_bte, Y_bte, first_bt, state):
        state_mask, xf_state = state
        t = first_bt.shape[1]
        new_mask, state_mask = get_mask(
            first_b11=first_bt[:, [[0]]],
            state_mask=state_mask,
            t=t,
            T=t + self.max_len,
            maxlen=self.max_len,
            heads=self.n_heads,
            device=X_bte.device,
        )
        output, xf_state = self.orc_block(X_bte, Y_bte, xf_state, new_mask)

        return output, (state_mask, xf_state)

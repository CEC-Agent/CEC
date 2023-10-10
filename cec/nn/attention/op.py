from __future__ import annotations

import math
import functools

import torch


def attention(
    Q_bte: torch.Tensor,
    K_bTe: torch.Tensor,
    V_bTe: torch.Tensor,
    mask: torch.Tensor | None = None,
    extra_btT: torch.Tensor | None = None,
    use_muP_factor: bool = False,
):
    """
    performs softmax(Q*K)*V operation

    t : output (write) time axis, possibly size=1 for just the last timestep
    T : input (read) time axis
    t < T is OK

    mask: (b*h, t, T)
    """
    assert Q_bte.dtype == K_bTe.dtype == V_bTe.dtype
    assert K_bTe.shape == V_bTe.shape
    assert Q_bte.ndim == K_bTe.ndim == V_bTe.ndim == 3
    assert K_bTe.shape == V_bTe.shape
    assert Q_bte.shape[0] == K_bTe.shape[0]
    assert Q_bte.shape[2] == K_bTe.shape[2]

    e = Q_bte.shape[2]
    if mask is not None:
        assert (
            mask.ndim == 3
            and mask.shape[:2] == Q_bte.shape[:2]
            and mask.shape[2] == K_bTe.shape[1]
        ), "mask shape should be (b, t, T)"
        bias = torch.zeros_like(mask).float()
        bias = bias.masked_fill(~mask, -float("inf"))
    else:
        bias = Q_bte.new_zeros(())
    if extra_btT is not None:
        bias = bias + extra_btT

    logit_btT = torch.baddbmm(
        bias,
        Q_bte.float(),
        K_bTe.float().transpose(-1, -2),
        alpha=(1 / e) if use_muP_factor else (1 / math.sqrt(e)),
    )  # (b, t, T)
    W_btT = torch.softmax(logit_btT, dim=2)  # (b, t, T)
    A_bte = W_btT @ V_bTe  # (b, t, T) x (b, T, e) -> (b, t, e)
    return A_bte


@functools.lru_cache()
def get_band_diagonal_mask(
    t: int, T: int, maxlen: int, batchsize: int, device: torch.device
) -> torch.Tensor:
    """Returns a band diagonal mask which is causal (upper triangle is masked)
    and such that any frame can only view up to maxlen total past frames
    including the current frame.

    Example Masks: Here 0 means that frame is masked and we mask it by adding a huge number to the attention logits (see orc.xf)
        t = 3, T = 3, maxlen = 3
          T
        t 1 0 0 |  mask out T > t
          1 1 0 |
          1 1 1 |
        t = 3, T = 6, maxlen = 3
        t 0 1 1 1 0 0 |  mask out T > t
          0 0 1 1 1 0 |
          0 0 0 1 1 1 |

    Args:
        t: number of rows (presumably number of frames recieving gradient)
        T: number of cols (presumably t + past context that isn't being gradient updated)
        maxlen: maximum number of frames (including current frame) any frame can attend to
        batchsize: number of masks to return
        device: torch device to place mask on

    Returns:
        Boolean mask of shape (batchsize, t, T)
    """
    m = torch.ones(t, T, dtype=bool)
    m.tril_(T - t)  # Mask out upper triangle
    if maxlen is not None and maxlen < T:  # Mask out lower triangle
        m.triu_(T - t - maxlen + 1)
    m_btT = m[None].repeat_interleave(batchsize, dim=0)
    m_btT = m_btT.to(device=device)
    return m_btT


@functools.lru_cache()
def get_subsampled_band_diagonal_mask(
    t: int,
    T: int,
    maxlen: int,
    batchsize: int,
    device: torch.device,
    subsample_every: int,
) -> torch.Tensor:
    m = torch.zeros(t, T, dtype=bool)
    # mask out the triangle, with stride of subsample_every
    for i in range(t):
        m[i, i::subsample_every] = True
    # flip the bottom right to the top left
    m = m.flip(0).flip(1)
    m_btT = m[None].repeat_interleave(batchsize, dim=0)
    m_btT = m_btT.to(device=device)
    return m_btT


def get_mask(
    first_b11: torch.Tensor,
    state_mask: torch.Tensor,
    t: int,
    T: int,
    maxlen: int,
    heads: int,
    device,
    subsample_every: int = 1,
    stateless: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns a band diagonal mask that respects masking past states (columns 0:T-t inclusive)
        if first_b11 is True. See get_band_diagonal_mask for how the base mask is computed.
        This function takes that mask and first zeros out any past context if first_b11 is True.

        Say our context is in chunks of length t (so here T = 4t). We see that in the second batch we recieved first=True
        context     t t t t
        first       F T F F
        Now, given this the mask should mask out anything prior to T < t; however since we don't have access to the past first_b11's
        we need to keep a state of the mask at those past timesteps. This is what state_mask is.

        In particular state_mask is a [b, t, T - t] mask matrix that contains the mask for the past T - t frames.

    Args: (See get_band_diagonal_mask for remaining args)
        first_b11: boolean tensor with shape [batchsize, 1, 1] indicating if the first timestep for each batch element had first=True
        state_mask: mask tensor of shape [b, t, T - t]
        t: number of mask rows (presumably number of frames for which we take gradient)
        T: number of mask columns (t + the number of past frames we keep in context)
        maxlen: actual context length
        heads: number of attention heads
        device: torch device

    Returns:
        m_btT: Boolean mask of shape (batchsize * heads, t, T)
        state_mask: updated state_mask
    """
    b = first_b11.shape[0]

    if state_mask is None:
        state_mask = torch.zeros((b, 1, T - t), dtype=bool, device=device)

    if subsample_every == 1:
        m_btT = get_band_diagonal_mask(t, T, maxlen, b, device).clone()
    else:
        # subsample the context so that the agent can look further back
        m_btT = get_subsampled_band_diagonal_mask(
            t, T, maxlen, b, device, subsample_every=subsample_every
        ).clone()  # Should be shape B, t, T
    if stateless:
        m_btT = m_btT[:, :, -t:]
        m_bhtT = m_btT[:, None].repeat_interleave(heads, dim=1)
        m_btT = m_bhtT.reshape((b * heads), t, t)
        state_mask = None
    else:
        not_first = ~first_b11.to(device=device)
        m_btT[:, :, :-t] &= not_first  # Zero out anything in the past if first is true
        m_btT[:, :, :-t] &= state_mask
        m_bhtT = m_btT[:, None].repeat_interleave(heads, dim=1)
        m_btT = m_bhtT.reshape((b * heads), t, T)

        # Update state_mask such that it reflects the most recent first
        state_mask = torch.cat(
            [
                state_mask[:, :, t:] & not_first,
                torch.ones((b, 1, min(t, T - t)), dtype=bool, device=device),
            ],
            dim=-1,
        )

    return m_btT, state_mask

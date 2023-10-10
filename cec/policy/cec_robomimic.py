from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

import cec.utils as U
from cec.policy.base import BasePolicy
from cec.nn.utils import build_mlp
from cec.nn.attention import TransformerBlocks
from cec.nn.feature.base import DictFeatureExtractor
from cec.nn.distributions import GMMHead


class CECRoboMimicPolicy(LightningModule, BasePolicy):
    def __init__(
        self,
        feature_encoders: dict[str, nn.Module],
        robosuite_action_size: int | None,
        hidden_size: int,
        timesteps: int,
        use_pointwise_layer: bool,
        pointwise_ratio: int,
        pointwise_use_activation: bool,
        attention_heads: int,
        attention_memory_size: int,
        n_transformer_blocks: int,
        pi_head_kwargs,
        subsample_every: int,
        low_noise_eval: bool = True,
    ):
        super().__init__()
        self.feature_encoders = DictFeatureExtractor(feature_encoders)
        if self.feature_encoders.output_dim != hidden_size:
            self.feature_linear = build_mlp(
                input_dim=self.feature_encoders.output_dim,
                hidden_dim=hidden_size,
                output_dim=hidden_size,
                hidden_depth=0,
                add_output_activation=False,
            )
        else:
            self.feature_linear = nn.Identity()

        self.transformer = TransformerBlocks(
            hidden_size=hidden_size,
            timesteps=timesteps,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_transformer_blocks,
            subsample_every=subsample_every,
        )
        self.post_transformer_layer = build_mlp(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            hidden_depth=0,
            norm_type="layernorm",
            add_input_norm=True,
            add_output_activation=True,
        )
        self.final_ln = nn.LayerNorm(hidden_size)

        self.pi_head = GMMHead(
            input_dim=hidden_size,
            num_gaussians=5,
            action_dim=robosuite_action_size,
            low_noise_eval=low_noise_eval,
            **pi_head_kwargs,
        )
        self._robosuite_action_size = robosuite_action_size
        self.ctx_len = timesteps

    @staticmethod
    def _prepare_input(obs, state, first_mask):
        return obs, state, first_mask

    def forward(self, *args, **kwargs):
        obs, state, first_mask = self._prepare_input(*args, **kwargs)

        x = self.feature_encoders(obs, first_mask=first_mask)
        x = self.feature_linear(x)

        x, state_out = self.transformer(x=x, first=first_mask, state=state)
        x = F.relu(x, inplace=False)
        x = self.post_transformer_layer(x)
        x = self.final_ln(x)

        pi = self.pi_head(x)
        return pi, state_out

    def act(self, *args, deterministic: bool, **kwargs):
        with torch.no_grad():
            pi, state_out = self.forward(*args, **kwargs)

        if deterministic:
            action = pi.mode()
        else:
            action = pi.sample()

        return action, state_out

    def training_step(self, batch, batch_idx):
        obs_chunks, action_chunks, action_mask_chunks = batch
        # obs, action, and action_mask are (n_chunks, B, ctx_len, ...)
        # we loop over n_chunks
        obs_chunks = U.unstack_sequence_fields(
            obs_chunks, batch_size=U.get_batch_size(obs_chunks, strict=True)
        )
        assert len(obs_chunks) == U.get_batch_size(
            action_chunks, strict=True
        ), "INTERNAL"
        losses, accuracies = [], []
        for i, (obs, action, action_mask) in enumerate(
            zip(obs_chunks, action_chunks, action_mask_chunks)
        ):
            T = U.get_batch_size(U.any_slice(obs, np.s_[0]))
            B = U.get_batch_size(obs)
            if i == 0:
                state = self.transformer.initial_state(batchsize=B, device=self.device)
                first_mask = U.any_concat(
                    [
                        torch.ones((B, 1), dtype=bool),
                        torch.zeros((B, T - 1), dtype=bool),
                    ],
                    dim=1,
                )
            else:
                first_mask = torch.zeros((B, T), dtype=bool)
            first_mask.type_as(action_mask)
            pi, state = self.forward(obs, state, first_mask)
            raw_loss = pi.imitation_loss(actions=action, reduction="none")
            raw_loss = raw_loss.reshape(action_mask.shape)
            # reduce the loss according to the action mask
            # "True" indicates should calculate the loss
            loss = raw_loss * action_mask
            loss = loss.sum() / action_mask.sum()
            losses.append(loss)
            accuracy = pi.imitation_accuracy(
                actions=action, reduction="mean", mask=action_mask
            )
            accuracies.append(accuracy)
        losses = torch.mean(torch.stack(losses))
        accuracies = sum(accuracies) / len(accuracies)
        # calculate real batch size
        assert action_mask_chunks.ndim == 3  # (n_chunks, B, T)
        real_batch_size = action_mask_chunks.sum()
        return losses, {"acc": accuracies, "real_batch_size": real_batch_size}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self.training_step(batch, batch_idx)

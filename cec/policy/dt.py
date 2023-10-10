import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    DecisionTransformerPreTrainedModel,
    DecisionTransformerGPT2Model,
)
from pytorch_lightning import LightningModule

import cec.utils as U
from cec.policy.base import BasePolicy


class DTPolicy(LightningModule, BasePolicy):
    def __init__(self, config, img_encoder):
        super().__init__()
        self.dt = DT(config, img_encoder)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask):
        action_preds = self.dt(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )
        return action_preds

    def training_step(self, batch, batch_idx):
        """
        (n_chunks, B, L, ....)
        """
        obs_dict, actions, pad_masks = batch
        # should be only one chunk
        assert (
            U.get_batch_size(obs_dict, strict=True)
            == actions.shape[0]
            == pad_masks.shape[0]
            == 1
        )
        obs_dict = U.any_slice(obs_dict, np.s_[0])
        actions = actions[0]
        pad_masks = pad_masks[0]
        states = obs_dict["partial_rgb"]
        returns_to_go = obs_dict["return_to_go"]
        B, L = states.shape[:2]
        timesteps = (
            torch.arange(L, device=self.device).unsqueeze(0).long()
        )  # (1, L), broadcastable
        action_preds = self.forward(
            states, actions, returns_to_go, timesteps, pad_masks
        )  # (B, L, A), logits
        loss = F.cross_entropy(
            action_preds.reshape((B * L, action_preds.shape[-1])),
            actions.reshape(-1),
            reduction="none",
        ).reshape(
            (B, L)
        )  # (B, L)
        loss = (loss * pad_masks).sum() / pad_masks.sum()
        acc = (action_preds.argmax(-1) == actions).sum() / pad_masks.sum()
        real_batch_size = pad_masks.sum()
        return loss, {"loss": loss, "acc": acc, "real_batch_size": real_batch_size}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self.training_step(batch, batch_idx)

    def act(self, *args, deterministic: bool, **kwargs):
        with torch.no_grad():
            logits, past_kv = self.dt.act(*args, **kwargs)  # (B, 1)

        if deterministic:
            return torch.argmax(logits, dim=2), past_kv
        else:
            return torch.distributions.Categorical(logits=logits).sample(), past_kv


class DT(DecisionTransformerPreTrainedModel):
    """

    The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL
    setting. Refer to the paper for more details: https://arxiv.org/abs/2106.01345

    """

    def __init__(self, config, img_encoder):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.img_encoder = img_encoder
        self.embed_action = torch.nn.Embedding(config.act_dim, config.hidden_size)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        self.predict_action = nn.Sequential(
            *([nn.Linear(config.hidden_size, config.act_dim)])
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        states=None,
        actions=None,
        rewards=None,
        returns_to_go=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=self.device
            )

        # embed each modality with a different head
        state_embeddings = self.img_encoder(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )
        device = stacked_inputs.device
        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(
                stacked_attention_mask.shape, device=device, dtype=torch.long
            ),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        action_preds = self.predict_action(x[:, 1])  # predict next action given state
        return action_preds

    @torch.no_grad()
    def act(
        self,
        return_to_go,
        state,
        timestep,
        prev_action=None,
        past_kv=None,
    ):
        input_seq = []
        return_to_go_embedding = self.embed_return(return_to_go)
        state_embedding = self.img_encoder(state)
        time_embedding = self.embed_timestep(timestep)
        return_to_go_embedding = return_to_go_embedding + time_embedding
        state_embedding = state_embedding + time_embedding
        if prev_action is not None:
            prev_action_embedding = self.embed_action(prev_action)
            prev_time_embedding = self.embed_timestep(timestep - 1)
            prev_action_embedding = prev_action_embedding + prev_time_embedding
            input_seq.append(prev_action_embedding)
        input_seq.append(return_to_go_embedding)
        input_seq.append(state_embedding)
        input_seq = U.any_concat(input_seq, dim=1)  # (B, L, E)

        B, L = input_seq.shape[:2]
        input_seq = self.embed_ln(input_seq)
        output = self.encoder(
            inputs_embeds=input_seq,
            position_ids=torch.zeros((B, L), device=self.device, dtype=torch.long),
            use_cache=True,
            past_key_values=past_kv,
        )
        last_hidden_state, past_key_values = output[:2]
        last_hidden_state = last_hidden_state[:, -1:]  # (B, 1, E)
        action_preds = self.predict_action(last_hidden_state)  # (B, 1, A)
        return action_preds, past_key_values

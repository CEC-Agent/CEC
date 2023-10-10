from __future__ import annotations

import torch
from pytorch_lightning import LightningModule

from cec.module.utils import CosineScheduleFunction


class DMLabImitationModule(LightningModule):
    def __init__(
        self,
        *,
        policy,
        lr: float,
        lr_warmup_steps: int,
        lr_cosine_steps: int,
        lr_cosine_min: float,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.policy = policy

        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # calculate cosine scheduler based on gradient steps
        # so we don't need to know number of batches apriori
        scheduler_kwargs = dict(
            base_value=1.0,  # anneal from the original LR value
            final_value=self.lr_cosine_min / self.lr,
            epochs=self.lr_cosine_steps,
            warmup_start_value=self.lr_cosine_min / self.lr,
            warmup_epochs=self.lr_warmup_steps,
            steps_per_epoch=1,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=CosineScheduleFunction(**scheduler_kwargs),
        )
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step"}],
        )

    def imitation_training_step(
        self, batch, batch_idx
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.policy.training_step(batch, batch_idx)

    def imitation_validation_step(
        self, batch, batch_idx
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.policy.validation_step(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        loss, _log_dict = self.imitation_training_step(batch, batch_idx)
        real_bs = (
            _log_dict.pop("real_batch_size") if "real_batch_size" in _log_dict else None
        )
        log_dict = {f"train/{k}": v for k, v in _log_dict.items()}
        log_dict["train/loss"] = loss
        # note that real batch size is the number of valid transitions
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, batch_size=real_bs
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _log_dict = self.imitation_validation_step(batch, batch_idx)
        real_bs = (
            _log_dict.pop("real_batch_size") if "real_batch_size" in _log_dict else None
        )
        log_dict = {f"val/{k}": v for k, v in _log_dict.items()}
        log_dict["val/loss"] = loss
        # note that real batch size is the number of valid transitions
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, batch_size=real_bs
        )
        return loss

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from cec import utils as U


class BaseFeatureExtractor(ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError


class DictFeatureExtractor(BaseFeatureExtractor, nn.ModuleDict):
    def forward(self, obs: dict[str, torch.Tensor], **kwargs):
        outs = {}
        for f_name, f_extractor in self.items():
            e_kwargs = {
                kwarg_name: kwarg[f_name]
                for kwarg_name, kwarg in kwargs.items()
                if kwarg is not None and f_name in kwarg
            }
            outs[f_name] = f_extractor(obs[f_name], **e_kwargs)
        outs = U.any_concat([outs[k] for k in sorted(outs.keys())], dim=-1)
        return outs

    @property
    def output_dim(self):
        return sum(fe.output_dim for fe in self.values())

import torch.nn as nn

from cec.nn.feature.base import BaseFeatureExtractor


class DummyIdentity(nn.Identity, BaseFeatureExtractor):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    @property
    def output_dim(self):
        return self.input_dim

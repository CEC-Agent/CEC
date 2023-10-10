import torch.nn as nn
import torch.nn.functional as F

from cec.nn.feature.base import BaseFeatureExtractor


class ActionOneHotEmb(nn.Module, BaseFeatureExtractor):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 7,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.l = nn.Linear(num_classes, embed_dim)
        self.num_classes = num_classes

    @property
    def output_dim(self):
        return self.embed_dim

    def forward(self, x):
        if x is None:
            return
        emb = F.one_hot(x.long(), num_classes=self.num_classes)
        return self.l(emb.half())

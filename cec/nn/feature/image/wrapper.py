from __future__ import annotations

import torch.nn as nn

from cec.nn.utils import FanInInitReLULayer
from cec.nn.feature.image.convnet import ImpalaConvNet


class ImpalaImgEncoder(nn.Module):
    """ImpalaCNN followed by a linear layer.

    cnn_outsize: impala output dimension
    output_size: output size of the linear layer.
    dense_init_norm_kwargs: kwargs for linear FanInInitReLULayer
    init_norm_kwargs: kwargs for 2d and 3d conv FanInInitReLULayer
    """

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: dict | None = None,
        init_norm_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__()

        dense_init_norm_kwargs = dense_init_norm_kwargs or {}
        init_norm_kwargs = init_norm_kwargs or {}

        self.cnn = ImpalaConvNet(
            outsize=cnn_outsize,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **kwargs,
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

    def forward(self, img):
        return self.linear(self.cnn(img))

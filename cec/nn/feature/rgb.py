from __future__ import annotations

from copy import deepcopy

import torch.nn as nn

from cec.nn.feature.base import BaseFeatureExtractor
from cec.nn.feature.image.preprocess import ImgPreprocess
from cec.nn.feature.image.wrapper import ImpalaImgEncoder


class RGBEmb(nn.Module, BaseFeatureExtractor):
    def __init__(
        self,
        convnet_width: int = 1,
        convnet_chans: list[int, int, int] = [16, 32, 32],
        convnet_output_size: int = 256,
        convnet_n_blocks: int = 2,
        hidden_size: int = 512,
        img_shape: list[int, int, int] = [3, 56, 56],
        scale_input_img: bool = True,
        init_norm_kwargs: dict | None = None,
        convnet_kwargs: dict | None = None,
        img_statistics: str | None = None,
        first_conv_norm: bool = False,
    ):
        super().__init__()

        init_norm_kwargs = init_norm_kwargs or {}
        convnet_kwargs = convnet_kwargs or {}

        chans = tuple(int(convnet_width * c) for c in convnet_chans)
        self.hidden_size = hidden_size

        # Dense init kwargs replaces batchnorm/groupnorm with layernorm
        dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            dense_init_norm_kwargs.pop("group_norm_groups", None)
            dense_init_norm_kwargs["layer_norm"] = True
        if dense_init_norm_kwargs.get("batch_norm", False):
            dense_init_norm_kwargs.pop("batch_norm", False)
            dense_init_norm_kwargs["layer_norm"] = True

        # Setup inputs
        self.img_preprocess = ImgPreprocess(
            img_statistics=img_statistics, scale_img=scale_input_img
        )
        self.img_process = ImpalaImgEncoder(
            cnn_outsize=convnet_output_size,
            output_size=hidden_size,
            inshape=img_shape,
            chans=chans,
            nblock=convnet_n_blocks,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=first_conv_norm,
            **convnet_kwargs,
        )

    @property
    def output_dim(self):
        return self.hidden_size

    def forward(self, x):
        """
        x: (B, T, 3, H, W)
        """
        x = self.img_preprocess(x)
        x = self.img_process(x)
        return x

from __future__ import annotations

import torch
import numpy as np
import torch.nn as nn


class ImgPreprocess(nn.Module):
    """Normalize incoming images.

    img_statistics: remote path to npz file with a mean and std image. If specified
        normalize images using this.
    scale_img: If true and img_statistics not specified, scale incoming images by 1/255.
    """

    def __init__(self, img_statistics: str | None = None, scale_img: bool = True):
        super().__init__()
        self.img_mean = None
        if img_statistics is not None:
            img_statistics = dict(**np.load(img_statistics))
            self.img_mean = nn.Parameter(
                torch.Tensor(img_statistics["mean"]), requires_grad=False
            )
            self.img_std = nn.Parameter(
                torch.Tensor(img_statistics["std"]), requires_grad=False
            )
        else:
            self.ob_scale = 255.0 if scale_img else 1.0

    def forward(self, img):
        x = img.to(dtype=torch.float16)
        if self.img_mean is not None:
            x = (x - self.img_mean) / self.img_std
        else:
            x = x / self.ob_scale
        return x

import torch.nn as nn
from robomimic.models.base_nets import CropRandomizer as _CropRandomizer, VisualCore

from cec.nn.feature.base import BaseFeatureExtractor


class CropRandomizer(_CropRandomizer):
    def forward_in(self, x):
        if not self.training:
            return x[:, :, 4:80, 4:80]
        else:
            return super().forward_in(x)


class RoboMimicRgbEncoder(nn.Module, BaseFeatureExtractor):
    def __init__(self):
        super().__init__()

        self._random_crop = CropRandomizer(
            input_shape=[3, 84, 84],
            crop_height=76,
            crop_width=76,
            num_crops=1,
        )

        self._visual_core = VisualCore(
            input_shape=(3, 76, 76),
            backbone_class="ResNet18Conv",
            backbone_kwargs={"pretrained": False, "input_coord_conv": False},
            pool_class="SpatialSoftmax",
            pool_kwargs={
                "num_kp": 32,
                "learnable_temperature": False,
                "temperature": 1.0,
                "noise_std": 0.0,
                "output_variance": False,
            },
            flatten=True,
            feature_dimension=64,
        )

        self._activation = nn.ReLU()

    @property
    def output_dim(self):
        return 64

    def forward(self, x):
        batch_dims = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        x = self._random_crop.forward_in(x)
        x = self._visual_core(x)
        x = self._activation(x)
        x = self._random_crop.forward_out(x)
        x = x.reshape(*batch_dims, 64)
        return x

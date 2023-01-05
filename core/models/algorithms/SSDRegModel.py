# -*- coding: utf-8 -*-
"""
Iterative group-to-reference/groupwise registration with sum of squared differences.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
from ..GroupRegModel import RegModel


class SSDRegModel(RegModel):
    def __init__(self, dimension, img_size, num_subjects=3, eps=1e-8, **kwargs):
        super(SSDRegModel, self).__init__(dimension, img_size, num_subjects, eps, **kwargs)

    def loss_function(self, warped_images, **kwargs):
        with torch.no_grad():
            overlap_region = self._get_overlap_region()
            mask = torch.logical_and(self._get_overlap_mask(), overlap_region).to(self.dtype)
            sample_mask = torch.rand_like(mask[[0]]).le(self.sample_rate)
            mask = torch.logical_and(mask, sample_mask).to(self.dtype)
            mask = F.interpolate(mask, scale_factor=self.scale_factor).to(torch.bool)

        warped_images = torch.stack(warped_images, dim=1)
        B, N = warped_images.shape[:2]
        masked_warped_images = torch.masked_select(warped_images,
                                                   mask=mask.unsqueeze(1)).view(B, N, -1)
        avg_img = masked_warped_images.mean(dim=1)
        loss = 0.
        for i in range(self.num_subjects):
            loss += F.mse_loss(masked_warped_images[:, i], avg_img)

        loss += self._get_regularization()

        return loss


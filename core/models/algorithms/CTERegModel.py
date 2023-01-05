# -*- coding: utf-8 -*-
"""
Iterative group-to-reference/groupwise registration with conditional template entropy.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
from ..GroupRegModel import RegModel
from ..criteria.CTE import CTE


class CTERegModel(RegModel):
    def __init__(self, dimension, img_size, num_subjects=3, eps=1e-8, **kwargs):
        super(CTERegModel, self).__init__(dimension, img_size, num_subjects, eps, **kwargs)
        self.CTE = CTE(self.dimension, num_bins=self.num_bins, sample_rate=self.sample_rate,
                       kernel_sigma=self.kernel_sigma, eps=self.eps)

    def predict_template(self, warped_images):
        with torch.no_grad():
            template = self.CTE(torch.stack(warped_images, dim=1))
        return template

    def loss_function(self, warped_images, **kwargs):
        num_bins = kwargs.pop('num_bins', self.num_bins)

        with torch.no_grad():
            overlap_region = self._get_overlap_region()
            mask = torch.logical_and(self._get_overlap_mask(), overlap_region).to(self.dtype)
            mask = F.interpolate(mask, scale_factor=self.scale_factor)

        loss = self.CTE.loss(torch.stack(warped_images, dim=1), mask=mask, num_bins=num_bins)

        loss += self._get_regularization()

        return loss


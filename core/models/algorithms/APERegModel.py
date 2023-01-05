# -*- coding: utf-8 -*-
"""
Iterative group-to-reference/groupwise registration with aggregated pairwise estimates.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
from ..GroupRegModel import RegModel
from ..criteria.MI import MI


class APERegModel(RegModel):
    def __init__(self, dimension, img_size, num_subjects=3, eps=1e-8, **kwargs):
        super(APERegModel, self).__init__(dimension, img_size, num_subjects, eps, **kwargs)
        self.MI = MI(self.dimension, num_bins=self.num_bins, sample_rate=self.sample_rate,
                     kernel_sigma=self.kernel_sigma, normalized=False, eps=self.eps)

    def loss_function(self, warped_images, **kwargs):
        num_bins = kwargs.pop('num_bins', self.num_bins)

        with torch.no_grad():
            overlap_region = self._get_overlap_region()
            mask = torch.logical_and(self._get_overlap_mask(), overlap_region).to(self.dtype)
            mask = F.interpolate(mask, scale_factor=self.scale_factor)

        loss = 0.
        for i in range(1, self.num_subjects):
            for j in range(i):
                loss -= self.MI.mi(warped_images[i], warped_images[j], mask, num_bins=num_bins)

        loss /= ((self.num_subjects - 1) / 2)

        loss += self._get_regularization()

        return loss


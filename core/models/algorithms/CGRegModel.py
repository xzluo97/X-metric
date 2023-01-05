# -*- coding: utf-8 -*-
"""
Iterative group-to-reference/groupwise registration by congealing.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
from ..GroupRegModel import RegModel
from ..criteria.CG import CG


class CGRegModel(RegModel):
    def __init__(self, dimension, img_size, num_subjects=3, eps=1e-8, **kwargs):
        super(CGRegModel, self).__init__(dimension, img_size, num_subjects, eps, **kwargs)
        self.win = kwargs.pop('win', 1)
        self.CG = CG(self.dimension, win=self.win, num_bins=self.num_bins,
                     kernel_sigma=self.kernel_sigma, eps=self.eps)

    def loss_function(self, warped_images, **kwargs):
        with torch.no_grad():
            overlap_region = self._get_overlap_region()
            mask = torch.logical_and(self._get_overlap_mask(), overlap_region).to(self.dtype)
            mask = F.interpolate(mask, scale_factor=self.scale_factor)

        loss = self.CG.loss2(torch.stack(warped_images, dim=1), mask=mask)

        loss += self._get_regularization()

        return loss


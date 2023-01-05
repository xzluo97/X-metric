# -*- coding: utf-8 -*-
"""
Iterative group-to-reference/groupwise registration with average mutual information.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import sys
sys.path.append('..')
import torch
from ..GroupRegModel import RegModel
from ..criteria.MI import MI


class AMIRegModel(RegModel):
    def __init__(self, dimension, img_size, num_subjects=3, eps=1e-8, **kwargs):
        super(AMIRegModel, self).__init__(dimension, img_size, num_subjects, eps, **kwargs)
        self.MI = MI(self.dimension, num_bins=self.num_bins, sample_rate=self.sample_rate,
                     kernel_sigma=self.kernel_sigma, normalized=False, eps=self.eps)

    def loss_function(self, warped_images):
        with torch.no_grad():
            overlap_region = self._get_overlap_region()
            mask = torch.logical_and(self._get_overlap_mask(), overlap_region).to(self.dtype)

        avg_image = torch.stack(warped_images).mean(dim=0).detach()

        loss = 0
        for i in range(self.num_subjects):
            loss -= self.MI.mi(avg_image, warped_images[i], mask=mask)

        loss += self._get_regularization()

        return loss

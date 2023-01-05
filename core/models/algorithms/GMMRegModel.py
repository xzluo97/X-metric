# -*- coding: utf-8 -*-
"""
Iterative group-to-reference/groupwise registration with Gaussian mixture model.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import numpy as np
import torch
import torch.nn.functional as F
from core import utils
from ..GroupRegModel import RegModel


class GMMRegModel(RegModel):
    def __init__(self, dimension, img_size, num_subjects=3, eps=1e-8, **kwargs):
        super(GMMRegModel, self).__init__(dimension, img_size, num_subjects, eps, **kwargs)

    def init_gmm_params(self, T=0):
        with torch.no_grad():
            sample_mask = torch.rand_like(self.mask).le(self.sample_rate)
            mask = torch.logical_and(self.mask, sample_mask)

            images = torch.stack(self.forward(), dim=1)
            sampled_images = torch.masked_select(images, mask=mask.unsqueeze(1)).view(self.B, self.num_subjects, -1)
            self.tau = torch.zeros(self.B, self.num_classes, np.prod(self.img_size),
                                   device=self.device, dtype=self.dtype).softmax(dim=1)
            self.mu = torch.mean(sampled_images, dim=-1).unsqueeze(1).repeat(1, self.num_classes, 1)
            center_images_flat = sampled_images.unsqueeze(1) - self.mu.unsqueeze(-1)
            self.sigma = torch.matmul(center_images_flat, center_images_flat.transpose(2, 3))
            self.pi = utils.get_normalized_prob(torch.sum(self.tau, dim=-1), dim=1, eps=self.eps)

            for _ in range(T):
                self.update_gmm_params(images)

        return

    def update_gmm_params(self, warped_images, mask=None):
        with torch.no_grad():
            if isinstance(warped_images, (tuple, list)):
                warped_images = torch.stack(warped_images, dim=1)
            if mask is None:
                mask = torch.ones_like(warped_images[:, [0]])
            vol_shape = warped_images.shape[3:]
            warped_images_flat = warped_images.view(self.B, 1, self.num_subjects, -1)
            mask_flat = mask.view(self.B, 1, -1)
            masked_warped_images = torch.masked_select(warped_images_flat,
                                                       mask_flat.unsqueeze(1)).view(self.B, 1, self.num_subjects, -1)

            p = self._log_prob(warped_images_flat).exp()

            self.tau = utils.get_normalized_prob(self.pi.unsqueeze(-1) * p, dim=1, eps=self.eps)
            self.posterior = self.tau.view(self.B, self.num_classes, *vol_shape)

            masked_tau = torch.masked_select(self.tau,
                                             mask_flat).view(self.B,
                                                             self.num_classes, -1).clamp(min=self.eps)
            self.mu = torch.sum(masked_warped_images * masked_tau.unsqueeze(2),
                                dim=-1) / torch.sum(masked_tau, dim=-1, keepdim=True)

            center_images_flat = masked_warped_images - self.mu.unsqueeze(-1)
            self.sigma = (masked_tau.unsqueeze(2) * center_images_flat) @ \
                         center_images_flat.transpose(-1, -2) / torch.sum(masked_tau,
                                                                          dim=-1, keepdim=True).unsqueeze(-1)
            self.pi = utils.get_normalized_prob(torch.sum(masked_tau, dim=-1), dim=1, eps=self.eps)

    def _log_prob(self, images_flat):
        L, Q = torch.linalg.eigh(self.sigma)
        half_inv_sigma = Q @ torch.diag_embed(1 / torch.sqrt(L).clamp(min=self.eps)) @ Q.transpose(-1, -2)
        norm_images = images_flat - self.mu.unsqueeze(-1)
        return - 0.5 * self.num_subjects * torch.log(torch.as_tensor(torch.pi * 2,
                                                                     dtype=self.dtype, device=self.device)) \
               - 0.5 * torch.prod(L, dim=-1, keepdim=True).clamp(min=self.eps).log() \
               - 0.5 * torch.square(half_inv_sigma @ norm_images).sum(dim=2)

    def loss_function(self, warped_images, **kwargs):
        with torch.no_grad():
            overlap_region = self._get_overlap_region()
            mask = torch.logical_and(self._get_overlap_mask(), overlap_region).to(self.dtype)
            mask = F.interpolate(mask, scale_factor=self.scale_factor)

            sample_mask = torch.rand_like(mask).le(self.sample_rate)
            mask = torch.logical_and(mask, sample_mask)
            self.update_gmm_params(warped_images, mask)

        warped_images = torch.stack(warped_images, dim=1)
        masked_warped_images = torch.masked_select(warped_images,
                                                   mask=mask.unsqueeze(1)).view(self.B, 1, self.num_subjects, -1)
        p = self._log_prob(masked_warped_images).exp()
        loss = - torch.sum(self.pi.unsqueeze(-1) * p, dim=1).clamp(min=self.eps).log().mean()

        loss += self._get_regularization()

        return loss


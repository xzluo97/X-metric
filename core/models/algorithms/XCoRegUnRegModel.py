# -*- coding: utf-8 -*-
"""
Iterative group-to-reference/groupwise registration with X-CoReg.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
from core import utils
from ..GroupRegModel import RegModel
from ..criteria.SVCD import SVCD


class XCoRegUnRegModel(RegModel):
    def __init__(self, dimension, img_size, num_subjects=3, eps=1e-8, **kwargs):
        super(XCoRegUnRegModel, self).__init__(dimension, img_size, num_subjects, eps, **kwargs)
        self.app_model = SVCD(self.dimension, num_classes=self.num_classes, num_bins=self.num_bins,
                              sample_rate=self.sample_rate, kernel_sigma=self.kernel_sigma, eps=self.eps)
        self.momentum = self.kwargs.pop('momentum', 0)
        self.min_prob = self.kwargs.pop('min_prob', 0.001)

    def init_app_params(self, images=None, template=None, T=0):
        if images is None:
            images = self.forward()

        self.pi = torch.full(size=(self.B, self.num_classes, *[1] * self.dimension),
                             fill_value=1 / self.num_classes, dtype=self.dtype, device=self.device)

        self.template = template
        if self.template is None:
            self.posterior = torch.randn(self.B, self.num_classes, *self.img_size,
                                         dtype=self.dtype, device=self.device).softmax(dim=1)
            self.template = self.posterior.clone()
        else:
            self.posterior = self.template.clone()

        for _ in range(T):
            self._update_app_params(images)

        return

    def _update_app_params(self, warped_images, mask=None, **kwargs):
        num_bins = kwargs.pop('num_bins', self.num_bins)
        if isinstance(warped_images, torch.Tensor):
            warped_images = torch.unbind(warped_images, dim=1)

        if mask is None:
            mask = torch.ones_like(warped_images[0])

        self.template = F.interpolate(self.template, size=warped_images[0].shape[2:],
                                      mode='bilinear' if self.dimension == 2 else 'trilinear',
                                      align_corners=True)

        warped_app_maps = []
        for i in range(self.num_subjects):
            warped_app_maps.append(self.app_model(warped_images[i], weight=self.template.detach(), mask=mask,
                                                  num_bins=num_bins))
        self.posterior = self.get_posterior(torch.stack(warped_app_maps, dim=1))
        self.pi = self.posterior.mul(mask).sum(
            dim=tuple(range(2, 2 + self.dimension)), keepdim=True).div(mask.sum(
            dim=tuple(range(2, 2 + self.dimension)), keepdim=True)
        ).detach()

        self.template = self.template.detach() * self.momentum + self.posterior * (1 - self.momentum)

        return

    def get_posterior(self, warped_cpds):
        joint_logit = 0
        for i in range(self.num_subjects):
            if self.group2ref and i == 0:
                joint_logit += warped_cpds[:, i].clamp(min=self.eps).log() * self.num_subjects
            else:
                joint_logit += warped_cpds[:, i].clamp(min=self.eps).log()

        posterior = utils.get_normalized_prob(joint_logit.exp() * self.pi, dim=1, eps=self.eps)
        posterior = torch.clamp(posterior,
                                min=self.min_prob,
                                max=1 - self.min_prob * (self.num_classes - 1))
        return posterior

    def X_metric(self, warped_images, posterior, **kwargs):
        mask = kwargs.pop('mask', None)
        num_bins = kwargs.pop('num_bins', self.num_bins)

        metrics = []
        for i in range(1 if self.group2ref else 0, self.num_subjects):
            joint_density, _, _ = self.app_model(warped_images[i], weight=posterior, mask=mask,
                                                 return_density=True, num_bins=num_bins)
            intensity_density = joint_density.sum(dim=1)

            joint_entropy = - torch.sum(joint_density * joint_density.clamp(min=self.eps).log(), dim=(1, 2)).mean()
            intensity_entropy = - torch.sum(intensity_density * intensity_density.clamp(min=self.eps).log(),
                                            dim=-1).mean()

            metrics.append(- joint_entropy + intensity_entropy)

        metric = torch.sum(torch.stack(metrics))

        return metric

    def loss_function(self, warped_images, **kwargs):
        num_bins = kwargs.pop('num_bins', self.num_bins)
        with torch.no_grad():
            overlap_region = self._get_overlap_region()
            mask = torch.logical_and(self._get_overlap_mask(), overlap_region).to(self.dtype)
            mask = F.interpolate(mask, scale_factor=self.scale_factor)

            self._update_app_params(warped_images, mask, num_bins=num_bins)

        post_X_metric = self.X_metric(warped_images, self.template, mask=mask, num_bins=num_bins)
        loss = - post_X_metric

        loss += self._get_regularization()

        return loss


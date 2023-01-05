# -*- coding: utf-8 -*-
"""
Iterative group-to-reference/groupwise registration with X-CoReg.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import torch
from core import utils
from ..GroupRegModel import RegModel
from ..criteria.SVCD import SVCD


class XCoRegGTRegModel(RegModel):
    def __init__(self, dimension, img_size, num_subjects=3, eps=1e-8, **kwargs):
        super(XCoRegGTRegModel, self).__init__(dimension, img_size, num_subjects, eps, **kwargs)
        self.app_model = SVCD(self.dimension, num_classes=self.num_classes, num_bins=self.num_bins,
                              sample_rate=1, kernel_sigma=self.kernel_sigma, eps=self.eps)
    def init_app_params(self, labels, **kwargs):
        images = kwargs.pop('images', self.images)

        if isinstance(labels, torch.Tensor):
            if labels.ndim == 3 + self.dimension:
                if isinstance(images, torch.Tensor):
                    images = torch.unbind(images, dim=1)
                labels = torch.unbind(labels, dim=1)
                app_model_params = [self.app_model(images[i], weight=labels[i],
                                                   return_density=True) for i in range(self.num_subjects)]
            elif labels.ndim == 2 + self.dimension:
                assert isinstance(images, torch.Tensor)
                app_model_params = [self.app_model(images, weight=labels,
                                                   return_density=True) for _ in range(self.num_subjects)]
            else:
                raise NotImplementedError
        elif isinstance(labels, (list, tuple)):
            if isinstance(images, torch.Tensor):
                images = torch.unbind(images, dim=1)
            app_model_params = [self.app_model(images[i], weight=labels[i],
                                               return_density=True) for i in range(self.num_subjects)]
        else:
            raise NotImplementedError

        self.pi = utils.get_normalized_prob(
            torch.stack(labels).sum(0).mul(self.mask).sum(dim=tuple(range(2, 2 + self.dimension)),
                                                          keepdim=True), dim=1, eps=self.eps)

        self.cond_densities = [param[0] / param[0].sum(dim=2, keepdim=True).clamp(min=self.eps)
                               for param in app_model_params]
        self.pad_min_vs = [param[1] for param in app_model_params]
        self.bin_widths = [param[2] for param in app_model_params]

        return

    def get_posterior(self, warped_cpds):
        joint_logit = 0
        for i in range(self.num_subjects):
            if self.group2ref and i == 0:
                joint_logit += warped_cpds[:, i].clamp(min=self.esp).log() * self.num_subjects
            else:
                joint_logit += warped_cpds[:, i].clamp(min=self.eps).log()

        posterior = utils.get_normalized_prob(joint_logit.exp().mul(self.pi), dim=1, eps=self.eps)

        return posterior

    def loss_function(self, warped_images, **kwargs):
        with torch.no_grad():
            overlap_region = self._get_overlap_region()
            mask = torch.logical_and(self._get_overlap_mask(), overlap_region).to(self.dtype)

        warped_app_maps = []
        for i in range(self.num_subjects):
            warped_app_maps.append(self.app_model.resample_density(self.cond_densities[i],
                                                                   image=warped_images[i], mask=mask,
                                                                   pad_min_v=self.pad_min_vs[i],
                                                                   bin_width=self.bin_widths[i]))

        self.posterior = self.get_posterior(torch.stack(warped_app_maps, dim=1))
        self.pi = self.posterior.detach().mul(mask).sum(
            dim=tuple(range(2, 2 + self.dimension)), keepdim=True).div(mask.sum(
            dim=tuple(range(2, 2 + self.dimension)), keepdim=True)
        )

        losses = []
        for i in range(1 if self.group2ref else 0, self.num_subjects):
            joint_density, _, _ = self.app_model(warped_images[i], weight=self.posterior.detach(), mask=mask, return_density=True)
            intensity_density = joint_density.sum(dim=1)

            joint_entropy = - torch.sum(joint_density * joint_density.clamp(min=self.eps).log(), dim=(1, 2)).mean()
            intensity_entropy = - torch.sum(intensity_density * intensity_density.clamp(min=self.eps).log(), dim=-1).mean()

            losses.append(joint_entropy - intensity_entropy)

        loss = torch.sum(torch.stack(losses))

        loss += self._get_regularization()

        return loss


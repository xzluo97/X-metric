# -*- coding: utf-8 -*-
"""
Iterative group-to-reference/groupwise registration with multivariate mixture model.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
from core import utils
from ..GroupRegModel import RegModel


class MvMMRegModel(RegModel):
    """
    Modified from https://github.com/xzluo97/MvMM-Demo/blob/main/src/MvMMVEM.py

    """
    def __init__(self, dimension, img_size, num_subjects=4, eps=1e-8, **kwargs):
        super(MvMMRegModel, self).__init__(dimension, img_size, num_subjects, eps, **kwargs)
        num_subtypes = self.kwargs.pop('num_subtypes', (2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1))
        assert len(num_subtypes) == self.num_classes * (self.num_subjects - 1)
        self.num_subtypes = [num_subtypes[self.num_classes * i : self.num_classes * (i + 1)]
                             for i in range(self.num_subjects - 1)]
        self.clamp_prob = self.kwargs.pop('clamp_prob', False)
        self.prob_interval = self.kwargs.pop('prob_interval', (0.05, 0.85))

    def init_app_params(self, images=None, atlas=None):
        self.pi = torch.sum(atlas * self.mask, dim=list(range(2, 2 + self.dimension)),
                            keepdim=True) / torch.sum(atlas * self.mask, dim=list(range(1, 2 + self.dimension)),
                                                      keepdim=True)
        self.atlas = utils.compute_normalized_prob(self.pi * atlas, dim=1)

        self.tau = [[torch.full((1, self.num_subtypes[i][k]), 1 / self.num_subtypes[i][k],
                                device=self.device, dtype=self.dtype) for k in range(self.num_classes)]
                    for i in range(self.num_subjects - 1)]

        mu_k = torch.sum(images * atlas.unsqueeze(1) * self.mask,
                         dim=list(range(3,
                                        3 + self.dimension))) / torch.sum(atlas.unsqueeze(1) * self.mask,
                                                                             dim=list(
                                                                                 range(3, 3 + self.dimension))).clamp(
            min=self.eps)
        sigma2_k = torch.sum(
            (images - mu_k.view(1, -1, self.num_classes, *[1] * self.dimension)) ** 2 * atlas.unsqueeze(1) * self.mask,
            dim=list(range(3, 3 + self.dimension))) / torch.sum(atlas.unsqueeze(1) * self.mask,
                                                                dim=list(range(3, 3 + self.dimension))).clamp(
            min=self.eps)

        self.mu = []
        for i in range(self.num_subjects - 1):
            self.mu.append([])
            for k in range(self.num_classes):
                if self.num_subtypes[i][k] == 1:
                    self.mu[i].append(mu_k[:, i, [k]].squeeze(0))
                else:
                    a = torch.linspace(-1, 1, steps=self.num_subtypes[i][k],
                                       device=self.device, dtype=self.dtype)
                    self.mu[i].append(mu_k[:, i, [k]] + a.unsqueeze(0) * sigma2_k[:, i, [k]].sqrt())
        self.sigma2 = [[sigma2_k[:, i, [k]].mul(self.num_subtypes[i][k]).repeat(1, self.num_subtypes[i][k])
                       for k in range(self.num_classes)] for i in range(self.num_subjects - 1)]

        self.posterior = self.atlas

    def _compute_class_cpd(self, tau, subtype_cpds, img_idx):
        class_cpd = []
        for k in range(self.num_classes):
            tau_ = tau[k].view(1, self.num_subtypes[img_idx][k], *[1] * self.dimension)
            class_cpd.append(torch.sum(tau_ * subtype_cpds[k], dim=1, keepdim=True))
        cpd = torch.cat(class_cpd, dim=1)
        return cpd

    def _compute_subtype_cpds(self, image, mu, sigma2, img_idx):
        subtype_cpds = []
        for k in range(self.num_classes):
            mu_ = mu[k].view(1, self.num_subtypes[img_idx][k], *[1] * self.dimension)
            sigma_ = sigma2[k].view(1, self.num_subtypes[img_idx][k], *[1] * self.dimension).sqrt()
            subtype_cpds.append(utils.gaussian_pdf(image, mu_, sigma_, eps=self.eps))
        return subtype_cpds

    def compute_subtype_class_cpds(self, images):
        subtype_cpds = [self._compute_subtype_cpds(images[:, i],
                                                   self.mu[i],
                                                   self.sigma2[i],
                                                   img_idx=i) for i in range(self.num_subjects - 1)]

        class_cpds = [self._compute_class_cpd(self.tau[i], subtype_cpds[i],
                                              img_idx=i) for i in range(self.num_subjects - 1)]

        return class_cpds, subtype_cpds

    def _update(self, warped_images_grad, warped_atlas_grad, mask, **kwargs):
        class_cpds_grad, subtype_cpds_grad = self.compute_subtype_class_cpds(warped_images_grad)

        subtype_cpds = [[cpd.detach() for cpd in subtype_cpds_grad[i]] for i in range(self.num_subjects - 1)]
        class_cpds = [cpd.detach() for cpd in class_cpds_grad]
        warped_images = warped_images_grad.detach()
        warped_atlas = warped_atlas_grad.detach()

        data_likelihood = torch.stack(class_cpds, dim=1).clamp(min=self.eps).log().sum(dim=1).exp()
        self.posterior = utils.compute_normalized_prob(data_likelihood * warped_atlas,
                                                       dim=1)

        self.pi = torch.sum(self.posterior * mask,
                            dim=list(range(2, 2 + self.dimension)),
                            keepdim=True) / torch.sum(warped_atlas / torch.sum(warped_atlas * self.pi,
                                                                               dim=1,
                                                                               keepdim=True).clamp(
            min=self.eps) * mask,
                                                      dim=list(range(2, 2 + self.dimension)),
                                                      keepdim=True).clamp(min=self.eps)

        for i in range(self.num_subjects - 1):
            for k in range(self.num_classes):
                tau_ = utils.compute_normalized_prob(
                    self.tau[i][k].view(1, self.num_subtypes[i][k],
                                        *[1] * self.dimension) * subtype_cpds[i][k],
                    dim=1) * self.posterior[:, [k]]

                tau_ = tau_ * mask

                self.tau[i][k] = utils.compute_normalized_prob(
                    tau_.sum(dim=[i + 2 for i in range(self.dimension)]),
                    dim=1)

                self.mu[i][k] = torch.sum(
                    tau_ * warped_images[:, i], dim=(0, *[i + 2 for i in range(self.dimension)])) / tau_.sum(
                    dim=[i + 2 for i in range(self.dimension)]
                ).clamp(min=self.eps)

                self.sigma2[i][k] = torch.sum(
                    tau_ * (warped_images[:, i] - self.mu[i][k].view(1, self.num_subtypes[i][k], *[1] * self.dimension)
                            ) ** 2, dim=[i + 2 for i in range(self.dimension)]) / tau_.sum(
                    dim=[i + 2 for i in range(self.dimension)]
                ).clamp(min=self.eps)

        return class_cpds_grad

    def _compute_data_likelihood(self, class_cpds_grad):
        data_likelihood = torch.stack(class_cpds_grad, dim=1).clamp(min=self.eps).log().sum(dim=1).exp()

        return data_likelihood

    def loss_function(self, warped_images, **kwargs):
        with torch.no_grad():
            mask = torch.logical_and(self._get_overlap_mask(), self.overlap_region).to(self.dtype)
            mask = F.interpolate(mask, scale_factor=self.scale_factor)

        warped_atlas = warped_images[-1]
        if self.clamp_prob:
            prob_interval = kwargs.pop('prob_interval', self.prob_interval)
            warped_atlas = torch.clamp(warped_atlas, min=prob_interval[0], max=prob_interval[1])

        class_cpds_grad = self._update(warped_images_grad=torch.stack(warped_images[:-1], dim=1),
                                       warped_atlas_grad=warped_atlas,
                                       mask=mask)

        likelihood = self._compute_data_likelihood(class_cpds_grad) * warped_atlas

        log_likelihood = likelihood.sum(dim=1).clamp_min(self.eps).log()

        loss = - torch.sum(log_likelihood * mask) / torch.sum(mask).add(self.eps)

        loss += self._get_regularization()

        return loss


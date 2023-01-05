# -*- coding: utf-8 -*-
"""
Deep Combined Computing.

__author__ = Xinzhe Luo
__version__ = 0.1

"""
import sys
sys.path.append('../../..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.networks.DCCNets import DCCNet
from core.register.SpatialTransformer import SpatialTransformer
from core.register.LocalDisplacementEnergy import BendingEnergy
from core.losses import CrossEntropy, DiceLoss
from ..criteria.SVCD import SVCD
from core import utils
from core.metrics import GWI
import numpy as np


class XCoRegDCCModel(nn.Module):
    def __init__(self, num_classes=4, eps=1e-8, **kwargs):
        super(XCoRegDCCModel, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.kwargs = kwargs
        self.dimension = self.kwargs.pop('dimension', 2)
        self.img_size = self.kwargs.pop('img_size', [224] * self.dimension)
        self.modalities = self.kwargs.pop('modalities', ('DE', 'C0', 'T2'))
        self.num_subjects = len(self.modalities)
        self.init_features = self.kwargs.pop('init_features', 32)
        self.num_blocks = self.kwargs.pop('num_blocks', 4)
        self.norm_type = self.kwargs.pop('norm_type', 'batch')
        self.dropout = self.kwargs.pop('seg_dropout', 0.2)
        self.num_bins = self.kwargs.pop('num_bins', 32)
        self.sample_rate = self.kwargs.pop('sample_rate', 1)
        self.kernel_sigma = self.kwargs.pop('kernel_sigma', 1)
        self.mask_radius = self.kwargs.pop('mask_radius', 10)
        self.prob_sigma = self.kwargs.pop('prob_sigma', 1)
        self.alpha = self.kwargs.pop('alpha', 1)
        self.ffd_spacings = self.kwargs.pop('ffd_spacings', [40])
        self.update_steps = self.kwargs.pop('update_steps', 1)
        self.sup_mods = self.kwargs.pop('sup_mods', None)
        if self.sup_mods is not None:
            self.sup_idx = [self.modalities.index(m) for m in self.sup_mods]
        else:
            self.sup_idx = []
        self.clamp_prob = self.kwargs.pop('clamp_prob', False)
        self.prob_interval = self.kwargs.pop('prob_interval', (0.05, 0.85))
        self.use_atlas = self.kwargs.pop('use_atlas', False)
        self.num_images = self.num_subjects + 1 if self.use_atlas else self.num_subjects

        self.net = DCCNet(in_channels=1, dimension=self.dimension,
                          init_features=self.init_features, num_blocks=self.num_blocks,
                          norm_type=self.norm_type, dropout=self.dropout, switch_bn=True,
                          modalities=self.modalities, compose_ddf=False, use_atlas=self.use_atlas)

        self.app_model = SVCD(self.dimension, num_classes=self.num_classes, num_bins=self.num_bins,
                              sample_rate=self.sample_rate, kernel_sigma=self.kernel_sigma, eps=self.eps)
        self.transform = SpatialTransformer(size=self.img_size)
        self.bending_energy = BendingEnergy(alpha=self.alpha, dimension=self.dimension)
        self.ce = CrossEntropy.CrossEntropyLoss()
        self.dice_loss = DiceLoss.DiceLoss()

        self.gwi = GWI(unbiased=True, size=self.img_size, padding_mode='zeros')

    def train(self, mode=True, seg_mode=True, reg_mode=True):
        super(XCoRegDCCModel, self).train(mode)

        self.net.seg_decoder.train(seg_mode)
        self.net.seg_output_conv.train(seg_mode)

        self.net.reg_decoder.train(reg_mode)
        self.net.reg_trans.train(reg_mode)
        self.net.reg_output_convs.train(reg_mode)

        return self

    def forward(self, images, init_flows=None):
        self.init_flows = init_flows
        if init_flows is not None:
            self.images = self.transform_images(images, init_flows)[1]
        x = self._normalize_images(torch.stack(self.images, dim=1))

        flows, self.seg_probs = self.net(x)

        self.flows = flows - torch.mean(flows, dim=1, keepdim=True)

        self.warped_images = self.transform_images(self.images, self.flows)[1]

        return self.warped_images

    def compute_segs(self, seg_probs):
        with torch.no_grad():
            return F.one_hot(torch.argmax(seg_probs, dim=2),
                             seg_probs.shape[2]).permute(0, 1, -1, *range(2, seg_probs.ndim - 1)).to(torch.float32)

    def _normalize_images(self, images):
        norm_img = images - torch.mean(images, dim=list(range(3, 3 + self.dimension)), keepdim=True)
        norm_img /= torch.std(images, dim=list(range(3, 3 + self.dimension)), keepdim=True)
        return norm_img

    def get_prior(self, meanshape, mask=None):
        with torch.no_grad():
            if mask is None:
                mask = torch.ones_like(meanshape)
            prior = torch.sum(meanshape * mask, dim=tuple(range(2, 2 + self.dimension)),
                              keepdim=True) / torch.sum(mask, dim=tuple(range(2, 2 + self.dimension)),
                                                        keepdim=True)
            prior = utils.get_normalized_prob(prior, dim=1)

        return prior

    def get_posterior(self, warped_images=None, warped_probs=None, warped_atlas=None, prior=None, posterior=None,
                      use_probs=False):
        if isinstance(warped_images, torch.Tensor):
            warped_images = torch.unbind(warped_images, dim=1)
        if isinstance(warped_probs, torch.Tensor):
            warped_probs = torch.unbind(warped_probs, dim=1)

        if warped_images is not None:
            B = warped_images[0].shape[0]
            device = warped_images[0].device
            dtype = warped_images[0].dtype
        elif warped_probs is not None:
            B = warped_probs[0].shape[0]
            device = warped_probs[0].device
            dtype = warped_probs[0].dtype
        else:
            raise NotImplementedError

        if hasattr(self, 'mask'):
            mask = self.mask
        else:
            mask = self.get_overlap_region()

        if prior is None:
            prior = torch.full([B, self.num_classes, *[1] * self.dimension],
                               fill_value=1 / self.num_classes, device=device, dtype=dtype)

        warped_cpds = []
        if warped_probs is None:
            assert posterior is not None
            warped_probs = [posterior] * self.num_subjects
        else:
            if use_probs:
                for i in range(self.num_subjects):
                    if i in self.sup_idx:
                        prob = warped_probs[i]
                        if self.clamp_prob:
                            prob = utils.get_normalized_prob(torch.clamp(prob, self.prob_interval[0],
                                                                         self.prob_interval[1]), dim=1)
                        warped_cpds.append(prob)

        if self.use_atlas:
            if warped_atlas is not None:
                if self.clamp_prob:
                    warped_atlas = utils.get_normalized_prob(torch.clamp(warped_atlas, self.prob_interval[0],
                                                                         self.prob_interval[1]), dim=1)
                warped_cpds.append(warped_atlas)

        if warped_images is not None:
            for i in range(self.num_subjects):
                warped_cpds.append(self.app_model(warped_images[i], weight=warped_probs[i], mask=mask))

        posterior = utils.get_normalized_prob(torch.clamp_min(torch.stack(warped_cpds, dim=1),
                                                              self.eps).log().sum(dim=1).exp().mul(prior),
                                              dim=1)

        return posterior

    def transform_images(self, images, pred_flows=None, init_flows=None, **kwargs):
        if isinstance(images, torch.Tensor):
            images = torch.unbind(images, dim=1)
        if isinstance(pred_flows, torch.Tensor):
            pred_flows = torch.unbind(pred_flows, dim=1)
        if isinstance(init_flows, torch.Tensor):
            init_flows = torch.unbind(init_flows, dim=1)
        if init_flows is None:
            warped_images = [self.transform(images[i], flows=None if pred_flows is None else pred_flows[i], **kwargs)
                             for i in range(self.num_subjects)]
            return images, warped_images
        else:
            with torch.no_grad():
                init_images = [self.transform(images[i], flows=None if init_flows is None else init_flows[i], **kwargs)
                               for i in range(self.num_subjects)]
            warped_images = [self.transform(init_images[i], flows=None if pred_flows is None else pred_flows[i], **kwargs)
                             for i in range(self.num_subjects)]
            return init_images, warped_images

    def update_posterior(self, warped_images, init_posterior=None, mask=None, warped_probs=None, T=1, **kwargs):
        if init_posterior is None:
            init_posterior = torch.randn(warped_images[0].shape[0], self.num_classes, *self.img_size,
                                         dtype=warped_images[0].dtype, device=warped_images[0].device).softmax(dim=1)
        if mask is None:
            if hasattr(self, 'mask'):
                mask = self.mask
            else:
                mask = self.get_overlap_region()

        posterior = init_posterior
        for t in range(T):
            self.prior = self.get_prior(posterior, mask=mask)
            posterior = self.get_posterior(warped_images, warped_probs=warped_probs, prior=self.prior,
                                           posterior=posterior, **kwargs)

        return posterior

    def X_metric(self, images, probs=None, posterior=None):
        if isinstance(images, torch.Tensor):
            images = torch.unbind(images, dim=1)
        if isinstance(probs, torch.Tensor):
            probs = torch.unbind(probs, dim=1)
        if probs is None:
            probs = [posterior] * self.num_subjects

        losses = []
        for i in range(self.num_subjects):
            joint_density, _, _ = self.app_model(images[i], weight=probs[i], mask=self.mask, return_density=True)
            intensity_density = joint_density.sum(dim=1)
            class_density = joint_density.sum(dim=2)

            joint_entropy = - torch.sum(joint_density * joint_density.clamp(min=self.eps).log(), dim=(1, 2)).mean()
            intensity_entropy = - torch.sum(intensity_density * intensity_density.clamp(min=self.eps).log(),
                                            dim=-1).mean()
            class_entropy = - torch.sum(class_density * class_density.clamp(min=self.eps).log(), dim=-1).mean()

            losses.append(joint_entropy - intensity_entropy - class_entropy)

        loss = torch.sum(torch.stack(losses))

        return loss

    def seg_loss_posterior(self, warped_probs, posterior, warped_atlas=None):
        if isinstance(warped_probs, torch.Tensor):
            warped_probs = torch.unbind(warped_probs, dim=1)

        losses = []
        for i in range(self.num_subjects):
            prob = warped_probs[i]
            loss = self.ce(posterior, prob, mask=self.mask)
            losses.append(loss)

        if warped_atlas is not None:
            losses.append(self.ce(posterior, warped_atlas, mask=self.mask))

        return torch.sum(torch.stack(losses))

    def seg_loss_probs(self, seg_probs, probs=None):
        losses = []
        for i in range(self.num_subjects):
            loss = 0.
            if i in self.sup_idx:
                if probs is not None:
                    init_prob = self.transform(probs[:, i], flows=self.init_flows[:, i])
                    loss += self.ce(init_prob, seg_probs[:, i])
                    loss += self.dice_loss(init_prob, seg_probs[:, i])
            else:
                prob = seg_probs[:, i]
                loss += torch.sum(prob * prob.clamp(min=self.eps).log(), dim=1).mean().neg()
            losses.append(loss)

        return torch.sum(torch.stack(losses))

    def transform_probs(self, probs=None, detach_seg_flows=False):
        if probs is None:
            warped_probs = self.transform_images(self.seg_probs,
                                                 self.flows.detach() if detach_seg_flows else self.flows)[1]
        else:
            warped_probs = []
            for i in range(self.num_subjects):
                if i in self.sup_idx:
                    warped_probs.append(self.transform(probs[:, i], flows=[self.flows[:, i], self.init_flows[:, i]]))
                else:
                    warped_probs.append(self.transform(self.seg_probs[:, i],
                                                       flows=self.flows[:, i].detach() if detach_seg_flows else self.flows[:, i]))

        return warped_probs

    def get_overlap_region(self):
        with torch.no_grad():
            overlap_masks = torch.stack([self.transform.getOverlapMask(self.warped_images[i], flows=self.flows[:, i])
                                         for i in range(self.num_subjects)], dim=1)
            overlap_region = torch.all(overlap_masks, dim=1).to(torch.float32)

        return overlap_region

    def loss_function(self, probs=None, **kwargs):
        atlas_prob = kwargs.pop('atlas_prob', None)
        if self.use_atlas:
            assert atlas_prob is not None
            warped_atlas = self.transform(atlas_prob, flows=self.flows[:, -1])
        else:
            warped_atlas = None

        self.mask = self.get_overlap_region()

        warped_probs = self.transform_probs(probs, detach_seg_flows=False)

        with torch.no_grad():
            init_posterior = self.get_posterior(self.warped_images, warped_probs=warped_probs,
                                                warped_atlas=warped_atlas, use_probs=True)
            self.posterior = self.update_posterior(self.warped_images, init_posterior,
                                                   warped_probs=warped_probs, warped_atlas=warped_atlas,
                                                   use_probs=True, T=self.update_steps)

        loss = self.X_metric(self.images, probs=self.seg_probs)

        loss += self.X_metric(self.warped_images, posterior=self.posterior)

        loss += self.seg_loss_posterior(warped_probs, self.posterior, warped_atlas=warped_atlas)

        loss += self.seg_loss_probs(self.seg_probs, probs)

        loss += self.bending_energy(self.flows) * self.num_images

        return loss

    def evaluateGWI(self, labels=None, phantom=None, reduce_batch=True):
        if labels is None:
            assert phantom is not None
            labels = torch.stack([phantom] * self.num_subjects, dim=1)

        with torch.no_grad():
            foreground_masks = torch.sum(labels[:, :, 1:], dim=2, keepdim=True)

            pre_gWI, post_gWI = self.gwi(self.init_flows, self.flows, foreground_masks)

            if reduce_batch:
                pre_gWI = torch.mean(pre_gWI, dim=1)
                post_gWI = torch.mean(post_gWI, dim=1)

            pre_gWI, post_gWI = pre_gWI.cpu().numpy(), post_gWI.cpu().numpy()

        return pre_gWI, post_gWI

    def transform_grids(self, spacing=8, thickness=1):
        with torch.no_grad():
            grids = torch.zeros_like(torch.stack(self.images, dim=1))

            num_grids = np.asarray(self.img_size) // spacing

            for d in range(self.dimension):
                for i in range(num_grids[d]):
                    grids[(*[slice(None)] * (d + 3), slice(i * spacing, i * spacing + thickness), *[slice(None)] * (self.dimension - 1 - d))] = 1

            warped_grids = self.transform_images(grids, self.flows, padding_mode='zeros')[1]

        return warped_grids

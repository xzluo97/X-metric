# -*- coding: utf-8 -*-
"""
Iterative group-to-reference/groupwise registration on Brainweb dataset.

__author__ = Xinzhe Luo
__version__ = 0.1

"""
import torch
import torch.nn as nn
from core.register.FFDGenerator import FFDGenerator
from core.register.SpatialTransformer import SpatialTransformer
from ..algorithms import XCoRegUnRegModel, XCoRegGTRegModel, GMMRegModel, CTERegModel, APERegModel
from core.data import image_utils
from core import utils
import numpy as np
from core.metrics import OverlapMetrics, GWI

Dice = OverlapMetrics()
CSFDice = OverlapMetrics(type='class_specific_dice', class_index=1)
GMDice = OverlapMetrics(type='class_specific_dice', class_index=2)
WMDice = OverlapMetrics(type='class_specific_dice', class_index=3)


class BrainWebRegModel(nn.Module):
    def __init__(self, dimension, img_size, eps=1e-8, **kwargs):
        super(BrainWebRegModel, self).__init__()
        self.dimension = dimension
        self.img_size = img_size
        self.eps = eps
        self.kwargs = kwargs
        self.modalities = self.kwargs.pop('modalities', ('pd', 't1', 't2'))
        self.model_type = self.kwargs.pop('model_type', None)
        assert self.model_type in ['XCoRegUn', 'XCoRegGT', 'GMM', 'APE', 'CTE']
        if self.model_type == 'XCoRegUn':
            Model = XCoRegUnRegModel.XCoRegUnRegModel
        elif self.model_type == 'XCoRegGT':
            Model = XCoRegGTRegModel.XCoRegGTRegModel
        elif self.model_type == 'GMM':
            Model = GMMRegModel.GMMRegModel
        elif self.model_type == 'APE':
            Model = APERegModel.APERegModel
        elif self.model_type == 'CTE':
            Model = CTERegModel.CTERegModel
        else:
            raise NotImplementedError
        self.reg = Model(self.dimension, self.img_size, eps=self.eps, num_subjects=len(self.modalities), **self.kwargs)
        assert not 'AFF' in self.reg.transform_type, "No affine transformation on Brainweb dataset!"

        self.num_subjects = self.reg.num_subjects
        self.mask_sigma = self.kwargs.pop('mask_sigma', -1)
        self.prior_sigma = self.kwargs.pop('prior_sigma', -1)
        self.gt_ffd_spacing = self.kwargs.pop('gt_ffd_spacing', (54, 45))
        self.label_noise_mode = self.kwargs.pop('label_noise_mode', 0)
        self.label_noise_param = self.kwargs.pop('label_noise_param', 0)

        self.gt_mesh2flow = FFDGenerator(size=self.reg.img_size, ffd_spacing=self.gt_ffd_spacing)
        self.gwi = GWI(unbiased=self.reg.zero_avg_flow, size=self.reg.img_size, padding_mode='zeros')

    def init_model_params(self, images, ffds, label):
        B = images.shape[0]
        assert images.shape[1] == self.num_subjects

        self.gt = label
        self.label = self.corrupt_label(label, mode=self.label_noise_mode, param=self.label_noise_param)
        self.gt_flows = []
        for i in range(self.num_subjects):
            if self.reg.group2ref and self.reg.inv_warp_ref and i == 0:
                self.gt_flows.append(torch.zeros(B, self.dimension, *self.reg.img_size,
                                                 device=images.device, dtype=images.dtype))
            else:
                self.gt_flows.append(self.gt_mesh2flow(ffds[:, i]))

        spatial_transformer = SpatialTransformer(size=label.shape[2:], padding_mode='zeros').to(label.device, label.dtype)
        self.inv_warped_labels = [spatial_transformer(self.label, flows=self.gt_flows[i], interp_mode='nearest')
                                  for i in range(self.num_subjects)]
        self.inv_warped_gts = [spatial_transformer(label, flows=self.gt_flows[i], interp_mode='nearest')
                               for i in range(self.num_subjects)]
        self.inv_warped_images = [spatial_transformer(images[:, i], flows=self.gt_flows[i])
                                  for i in range(self.num_subjects)]

        self.reg.init_reg_params(images=self.inv_warped_images)

        if label is None:
            prior = torch.full((B, self.reg.num_classes, *self.reg.img_size),
                                fill_value=1 / self.reg.num_classes, device=images.device, dtype=images.dtype)
            mask = torch.ones(B, 1, *self.reg.img_size, dtype=images.dtype, device=images.device)
        else:
            if self.mask_sigma == -1:
                mask = torch.ones(B, 1, *self.reg.img_size, dtype=images.dtype, device=images.device)
            else:
                mask = self._spatial_filter(label[:, 1:].sum(dim=1, keepdim=True),
                                            utils.gauss_kernel1d(self.mask_sigma)).gt(self.eps).to(self.images.dtype)

            if self.prior_sigma == -1:
                prior = torch.full((B, self.reg.num_classes, *self.reg.img_size),
                                    fill_value=1 / self.reg.num_classes, device=images.device, dtype=images.dtype)

            else:
                prior = utils.compute_normalized_prob(self._spatial_filter(label,
                                                                           utils.gauss_kernel1d(self.prior_sigma)),
                                                      dim=1)
        self.reg.mask = mask
        self.reg.prior = prior

        if self.model_type == 'ICMIUn':
            self.reg.init_app_params()
        if self.model_type == 'ICMIGT':
            self.reg.init_app_params(labels=self.inv_warped_labels)
        if self.model_type == 'GMM':
            self.reg.init_gmm_params()

    def corrupt_label(self, label, mode, param):
        corrupted_label = label.clone()
        if mode == 0:
            return label
        elif mode == 1:
            patch_size = np.asarray([s // 8 for s in self.reg.img_size], dtype=np.int16)
            for _ in range(param):
                patch_center = np.asarray([np.random.randint(patch_size[i] // 2,
                                                             self.reg.img_size[i] + patch_size[i] // 2 - patch_size[i] + 1)
                                           for i in range(self.dimension)], dtype=np.int16)
                begin = patch_center - patch_size // 2
                end = patch_center - patch_size // 2 + patch_size

                corrupted_label[:, 1:, begin[0]:end[0], begin[1]:end[1]] = 0
                corrupted_label[:, 0, begin[0]:end[0], begin[1]:end[1]] = 1
        elif mode == 2:
            mask = torch.rand_like(corrupted_label).le(param)
            corrupted_label[mask] = 0
        elif mode == 3:
            contour = image_utils.extract_contour(corrupted_label)
            corrupted_label = image_utils.random_permute_contour(corrupted_label, contour, rate=param)
        else:
            raise NotImplementedError
        return corrupted_label

    def evaluateForegroundWarpingIndex(self):
        with torch.no_grad():
            pred_flows = self.reg.predict_flows()
            init_overlaps = [self.reg.transform.getOverlapMask(self.inv_warped_images[i],
                                                               flows=self.gt_flows[i]) for i in range(self.num_subjects)]
            init_masks = [label[:, 1:].sum(dim=1, keepdim=True) for label in self.inv_warped_gts]
            masks = [torch.logical_and(init_overlaps[i], init_masks[i]) for i in range(self.num_subjects)]

            pre_gWI, post_gWI = self.gwi(torch.stack(self.gt_flows, dim=1),
                                         torch.stack(pred_flows, dim=1),
                                         torch.stack(masks, dim=1))

            pre_gWI, post_gWI = torch.mean(pre_gWI, dim=0).tolist(), torch.mean(post_gWI, dim=0).tolist()

        return pre_gWI, post_gWI

    def evaluateOverlapWarpingIndex(self):
        with torch.no_grad():
            pred_flows = self.reg.predict_flows()

            masks = [self.reg.transform.getOverlapMask(self.inv_warped_images[i],
                                                       flows=self.gt_flows[i]) for i in range(self.num_subjects)]

            pre_gWI, post_gWI = self.gwi(torch.stack(self.gt_flows, dim=1),
                                         torch.stack(pred_flows, dim=1),
                                         torch.stack(masks, dim=1))

            pre_gWI, post_gWI = torch.mean(pre_gWI, dim=0).tolist(), torch.mean(post_gWI, dim=0).tolist()


        return pre_gWI, post_gWI

    def evaluateOverlap(self):
        with torch.no_grad():
            pred_flows = self.reg.predict_flows()
            warped_labels = [self.reg.transform(self.inv_warped_gts[i], flows=pred_flows[i],
                                                interp_mode='nearest') for i in range(self.num_subjects)]
            if self.reg.group2ref:
                pre_Dice = [Dice(self.inv_warped_gts[0],
                                 self.inv_warped_gts[i]).mean().item() for i in range(self.num_subjects)]
                pre_CSFDice = [CSFDice(self.inv_warped_gts[0],
                                       self.inv_warped_gts[i]).mean().item() for i in range(self.num_subjects)]
                pre_GMDice = [GMDice(self.inv_warped_gts[0],
                                     self.inv_warped_gts[i]).mean().item() for i in range(self.num_subjects)]
                pre_WMDice = [WMDice(self.inv_warped_gts[0],
                                     self.inv_warped_gts[i]).mean().item() for i in range(self.num_subjects)]
                post_Dice = [Dice(self.inv_warped_gts[0], warped_labels[i]).mean().item() for i in range(self.num_subjects)]
                post_CSFDice = [CSFDice(self.inv_warped_gts[0], warped_labels[i]).mean().item() for i in range(self.num_subjects)]
                post_GMDice = [GMDice(self.inv_warped_gts[0], warped_labels[i]).mean().item() for i in range(self.num_subjects)]
                post_WMDice = [WMDice(self.inv_warped_gts[0], warped_labels[i]).mean().item() for i in range(self.num_subjects)]
            else:
                pre_Dice = []
                pre_CSFDice = []
                pre_GMDice = []
                pre_WMDice = []
                post_Dice = []
                post_CSFDice = []
                post_GMDice = []
                post_WMDice = []
                for i in range(self.num_subjects):
                    pre_Dice.append(torch.mean(torch.stack([Dice(self.inv_warped_gts[i], self.inv_warped_gts[j])
                                                            for j in range(self.num_subjects) if j != i]),
                                               dim=0).mean().item())
                    pre_CSFDice.append(torch.mean(torch.stack([CSFDice(self.inv_warped_gts[i], self.inv_warped_gts[j])
                                                               for j in range(self.num_subjects) if j != i]),
                                                  dim=0).mean().item())
                    pre_GMDice.append(torch.mean(torch.stack([GMDice(self.inv_warped_gts[i], self.inv_warped_gts[j])
                                                              for j in range(self.num_subjects) if j != i]),
                                                 dim=0).mean().item())
                    pre_WMDice.append(torch.mean(torch.stack([WMDice(self.inv_warped_gts[i], self.inv_warped_gts[j])
                                                              for j in range(self.num_subjects) if j != i]),
                                                 dim=0).mean().item())
                    post_Dice.append(torch.mean(torch.stack([Dice(warped_labels[i], warped_labels[j])
                                                             for j in range(self.num_subjects) if j != i]),
                                                dim=0).mean().item())
                    post_CSFDice.append(torch.mean(torch.stack([CSFDice(warped_labels[i], warped_labels[j])
                                                                for j in range(self.num_subjects) if j != i]),
                                                   dim=0).mean().item())
                    post_GMDice.append(torch.mean(torch.stack([GMDice(warped_labels[i], warped_labels[j])
                                                               for j in range(self.num_subjects) if j != i]),
                                                  dim=0).mean().item())
                    post_WMDice.append(torch.mean(torch.stack([WMDice(warped_labels[i], warped_labels[j])
                                                               for j in range(self.num_subjects) if j != i]),
                                                  dim=0).mean().item())

        return {'Dice': pre_Dice, 'CSF-Dice': pre_CSFDice, 'GM-Dice': pre_GMDice, 'WM-Dice': pre_WMDice}, \
               {'Dice': post_Dice, 'CSF-Dice': post_CSFDice, 'GM-Dice': post_GMDice, 'WM-Dice': post_WMDice}

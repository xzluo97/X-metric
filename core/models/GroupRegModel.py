# -*- coding: utf-8 -*-
"""
Iterative group-to-reference/groupwise registration.

__author__ = Xinzhe Luo
__version__ = 0.1

"""
import sys

import numpy as np

sys.path.append('../..')
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.register.FFDGenerator import FFDGenerator
from core.register.SpatialTransformer import SpatialTransformer
from core.register.VectorIntegration import VectorIntegration
from core.register.LocalDisplacementEnergy import BendingEnergy
from core import utils
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class RegModel(nn.Module):
    def __init__(self, dimension, img_size=None, num_subjects=None, eps=1e-8, **kwargs):
        super(RegModel, self).__init__()
        self.dimension = dimension
        self.img_size = img_size
        self.num_subjects = num_subjects
        self.eps = eps
        self.kwargs = kwargs
        self.num_classes = self.kwargs.pop('num_classes', 4)
        self.num_bins = self.kwargs.pop('num_bins', 64)
        self.sample_rate = self.kwargs.pop('sample_rate', 1)
        self.kernel_sigma = self.kwargs.pop('kernel_sigma', 1)
        self.alpha = self.kwargs.pop('alpha', 1)
        self.transform_type = self.kwargs.pop('transform_type', ['DDF'])
        self.padding_mode = self.kwargs.pop('padding_mode', 'wrap')
        self.group2ref = self.kwargs.pop('group2ref', False)
        self.zero_avg_flow = self.kwargs.pop('zero_avg_flow', False)
        self.zero_avg_vec = self.kwargs.pop('zero_avg_vec', False)
        self.zero_avg_disp = self.kwargs.pop('zero_avg_disp', False)
        self.inv_warp_ref = self.kwargs.pop('inv_warp_ref', False)
        self.norm_img = self.kwargs.pop('norm_img', False)
        self.norm_type = self.kwargs.pop('norm_type', 'z-score')
        self.int_steps = self.kwargs.pop('int_steps', 7)
        self.logger = self.kwargs.get("logger", logging)

        if 'FFD' in self.transform_type:
            pred_ffd_spacing = self.kwargs.pop('pred_ffd_spacing', None)
            self.pred_ffd_iso = self.kwargs.pop('pred_ffd_iso', True)
            if self.pred_ffd_iso:
                assert pred_ffd_spacing is not None
                self.pred_ffd_spacing = pred_ffd_spacing
                self.num_ffd_levels = len(self.pred_ffd_spacing)

            else:
                if self.pred_ffd_spacing is None:
                    self.pred_ffd_spacing = self.gt_ffd_spacing
                assert len(pred_ffd_spacing) % self.dimension == 0
                self.num_ffd_levels = len(pred_ffd_spacing) // self.dimension
                self.pred_ffd_spacing = [pred_ffd_spacing[i:(i + self.dimension)] for i in range(self.num_ffd_levels)]

            self.num_reg_levels = self.num_ffd_levels + len(self.transform_type) - 1
            self.ffd_idx = self.transform_type.index('FFD')
            self.reg_level_type = self.transform_type[:self.ffd_idx] + \
                                  ['FFD_%s' % j for j in range(self.num_ffd_levels)] + \
                                  self.transform_type[(self.ffd_idx + 1):]
        else:
            self.num_reg_levels = len(self.transform_type)
            self.reg_level_type = self.transform_type
        self.max_reg_level = self.num_reg_levels - 1
        self.activated_reg_levels = list(range(self.num_reg_levels))

        self.num_res = self.kwargs.pop('num_res', [[1] * self.dimension] * self.num_reg_levels)
        self._verify_hyper_parameters()
        self.max_num_res, self.resolutions = self._get_resolutions(self.num_res)

        self.bending_energy = BendingEnergy(alpha=self.alpha, dimension=self.dimension)

        self.device = None
        self.dtype = None

    def _get_resolutions(self, num_res):
        def get_res(res_lst):
            assert len(res_lst) == self.dimension
            res_levels = []
            max_num_res = np.max(num_res)
            for i in range(max_num_res - 1, -1, -1):
                res_levels.append([int(i * j / max_num_res) for j in res_lst])
            return max_num_res, res_levels

        num_res = np.asarray(num_res)
        if num_res.ndim == 1:
            num_res = [num_res[i:i + self.dimension] for i in range(0, len(num_res), self.dimension)]

        elif num_res.ndim == 2:
            pass
        else:
            raise NotImplementedError

        res = [get_res(res) for res in num_res]

        return [r[0] for r in res], [r[1] for r in res]

    def _verify_hyper_parameters(self):
        assert self.transform_type.count('TRA') <= 1
        assert self.transform_type.count('RIG') <= 1
        assert self.transform_type.count('AFF') <= 1
        assert self.transform_type.count('FFD') <= 1
        assert self.transform_type.count('DDF') + self.transform_type.count('SVF') <= 1
        if 'TRA' in self.transform_type:
            assert self.transform_type.index('TRA') == 0
        if 'RIG' in self.transform_type:
            warnings.warn('The rigid transformation must be defined under the physical spacing. '
                          'DO NOT forget to set the physical spacing manually!')
            assert self.transform_type.index('RIG') <= 1
        if 'AFF' in self.transform_type:
            assert self.transform_type.index('AFF') <= 2
        if self.zero_avg_flow:
            assert 'FFD' in self.transform_type or 'DDF' in self.transform_type or 'SVF' in self.transform_type
        if self.zero_avg_vec:
            assert not self.zero_avg_flow and 'SVF' in self.transform_type

        if all(isinstance(y, (list, tuple)) for y in self.num_res):
            assert len(self.num_res) == self.num_reg_levels
            assert all(len(y) == self.dimension for y in self.num_res)
        elif all(isinstance(y, (int, float)) for y in self.num_res):
            assert len(self.num_res) == self.num_reg_levels * self.dimension
        else:
            raise NotImplementedError

    def init_reg_params(self, images, **kwargs):
        if isinstance(images, torch.Tensor):
            self.B = images.shape[0]
            if not images.shape[1] == self.num_subjects:
                self.num_subjects = images.shape[1]
                self.logger.info("#Subjects changed to %s!" % self.num_subjects)
            self.device = images.device
            self.dtype = images.dtype
            self.img_size = images.shape[3:]
            images = torch.unbind(images, dim=1)
        elif isinstance(images, (list, tuple)):
            self.B = images[0].shape[0]
            if not len(images) == self.num_subjects:
                self.num_subjects = len(images)
                self.logger.info("#Subjects changed to %s!" % self.num_subjects)
            self.device = images[0].device
            self.dtype = images[0].dtype
            self.img_size = images[0].shape[2:]
        else:
            raise NotImplementedError

        self.group_num = (self.num_subjects - 1) if self.group2ref else self.num_subjects

        self.reference_frame = kwargs.pop('reference_frame', None)
        self.reference = kwargs.pop('reference', None)
        if self.group2ref:
            assert self.reference_frame is not None or self.reference is not None, \
                'Please set the reference or the reference frame index for group-to-reference registration!'

        self.mask_frame = kwargs.pop('mask_frame', self.reference_frame)
        masks = kwargs.pop('masks', None)
        if masks is None:
            self.masks = None
            self.mask_frame = None
            self.overlap_region = torch.ones(self.B, 1, *self.img_size, device=self.device, dtype=self.dtype)
        else:
            self.masks = masks.to(self.device, self.dtype)
            if self.mask_frame is None:
                assert self.masks.ndim == self.dimension + 3
                self.overlap_region = torch.all(self.masks, dim=1)
            else:
                assert self.masks.ndim == self.dimension + 2
                self.overlap_region = self.masks

        self.spacings = kwargs.pop('spacings', [[1] * self.dimension] * self.num_subjects)
        self.rho = self._get_spacing_matrices(spacings=self.spacings)

        self.transform = SpatialTransformer(size=self.img_size,
                                            padding_mode=self.padding_mode).to(self.device, self.dtype)
        if 'SVF' in self.transform_type:
            self.vec2flow = VectorIntegration(size=self.img_size, int_steps=self.int_steps).to(self.device, self.dtype)
        if 'FFD' in self.transform_type:
            self.pred_mesh2flow = [[FFDGenerator(size=self.img_size, ffd_spacing=s,
                                                 img_spacing=self.spacings[i]).to(self.device, self.dtype)
                                    for s in self.pred_ffd_spacing] for i in range(self.num_subjects)]
            self.max_control_point_size = np.max(np.asarray([[g.control_point_size for g in self.pred_mesh2flow[i]]
                                                             for i in range(self.num_subjects)]),
                                                 axis=0)

        self.params = nn.ParameterDict()
        for tt in self.transform_type:
            if tt == 'TRA':
                zetas = torch.zeros(self.B, self.group_num, self.dimension, device=self.device, dtype=self.dtype)
                self.params['TRA'] = nn.Parameter(zetas)
            elif tt == 'RIG':
                etas = torch.zeros(self.B, self.group_num, 2, self.dimension, device=self.device, dtype=self.dtype)
                self.params['RIG_r'] = nn.Parameter(etas[:, :, 0])
                self.params['RIG_t'] = nn.Parameter(etas[:, :, 1])
            elif tt == 'AFF':
                identity_theta = self._get_identity_theta().unsqueeze(0).unsqueeze(0)
                thetas = identity_theta.repeat(self.B, self.group_num, 1, 1)
                self.params['AFF'] = nn.Parameter(thetas)
            elif tt == 'FFD':
                meshes = [torch.zeros(self.B, self.group_num, self.dimension, *self.max_control_point_size[j],
                                      dtype=self.dtype, device=self.device)
                          for j in range(self.num_ffd_levels)]
                for j in range(self.num_ffd_levels):
                    self.params['FFD_%s' % j] = nn.Parameter(meshes[j])
            elif tt in ['DDF', 'SVF']:
                vectors = torch.zeros(self.B, self.group_num, self.dimension, *self.img_size,
                                      dtype=self.dtype, device=self.device)
                self.vectors = nn.Parameter(vectors)
                self.params[tt] = self.vectors
            else:
                raise NotImplementedError

        if self.norm_img:
            self.images = self.normalize_images(images, masks=self.masks)
        else:
            self.images = images

        roi_center = kwargs.pop('roi_center', None)
        if roi_center is None:
            center = np.asarray(self.img_size) / 2
        else:
            center = np.asarray(roi_center)
        if self.group2ref or self.reference_frame is not None:
            center *= np.asarray(self.spacings[self.reference_frame])
        else:
            avg_spacing = np.stack([np.asarray(s) for s in self.spacings]).mean(0)
            center *= avg_spacing
        center = np.repeat(np.expand_dims(center, axis=0), self.B, axis=0)
        self.center = nn.Parameter(torch.as_tensor(center, device=self.device, dtype=self.dtype))

    def activate_params(self, levels):
        for j in levels:
            for k, p in self.params.items():
                if self.reg_level_type[j] in k:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
        self.max_reg_level = max(levels)
        self.activated_reg_levels = levels

    def normalize_images(self, images, masks=None):
        if masks is None:
            masks = torch.ones(self.B, self.num_subjects, 1, *self.img_size, device=self.device, dtype=self.dtype)
        else:
            if masks.ndim == self.dimension + 2:
                masks = masks.unsqueeze(1)
                masks = masks.repeat(1, self.num_subjects, 1, *[1] * self.dimension)
        if isinstance(images, torch.Tensor):
            if self.norm_type == 'z-score':
                mean = torch.sum(images * masks,
                                 dim=tuple(range(3, 3 + self.dimension)),
                                 keepdim=True) / torch.sum(masks,
                                                           dim=tuple(range(3, 3 + self.dimension)),
                                                           keepdim=True).clamp(min=self.eps)

                imgs = images - mean
                var = torch.sum(torch.square(imgs) * masks,
                                dim=tuple(range(3, 3 + self.dimension)),
                                keepdim=True) / torch.sum(masks,
                                                          dim=tuple(range(3, 3 + self.dimension)),
                                                          keepdim=True).clamp(min=self.eps)
                norm_imgs = imgs / torch.sqrt(var)
            elif self.norm_type == 'min-max':
                min_v = utils.masked_min(images, mask=masks, dim=tuple(range(3, 3 + self.dimension)), keepdim=True)
                imgs = images - min_v * masks
                max_v = utils.masked_max(imgs, mask=masks, dim=tuple(range(3, 3 + self.dimension)), keepdim=True)
                norm_imgs = imgs * masks / max_v + (1 - masks) * imgs
            else:
                raise NotImplementedError

        elif isinstance(images, (list, tuple)):
            masks = torch.unbind(masks, dim=1)
            norm_imgs = []
            for i in range(len(images)):
                if self.norm_type == 'z-score':
                    mean = torch.sum(images[i] * masks[i],
                                     dim=tuple(range(2, 2 + self.dimension)),
                                     keepdim=True) / torch.sum(masks[i],
                                                               dim=tuple(range(2, 2 + self.dimension)),
                                                               keepdim=True).clamp(min=self.eps)
                    img = images[i] - mean
                    var = torch.sum(torch.square(img) * masks[i],
                                    dim=tuple(range(2, 2 + self.dimension)),
                                    keepdim=True) / torch.sum(masks[i],
                                                              dim=tuple(range(2, 2 + self.dimension)),
                                                              keepdim=True).clamp(min=self.eps)
                    norm_imgs.append(img / torch.sqrt(var))
                elif self.norm_type == 'min-max':
                    min_v = utils.masked_min(images[i], mask=masks[i],
                                             dim=tuple(range(2, 2 + self.dimension)), keepdim=True)
                    img = images[i] - min_v * masks[i]
                    max_v = utils.masked_max(img, mask=masks[i],
                                             dim=tuple(range(2, 2 + self.dimension)), keepdim=True)
                    norm_imgs.append(img * masks[i] / max_v + (1 - masks[i]) * img)
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError
        return norm_imgs

    def _get_spacing_matrices(self, spacings=None, **kwargs):
        device = kwargs.pop('device', self.device)
        dtype = kwargs.pop('dtype', self.dtype)
        B = kwargs.pop('B', 1)
        if spacings is None:
            num_subjects = kwargs.pop('num_subjects', self.num_subjects)
            rho = torch.cat([torch.diag(torch.ones(self.dimension, device=device, dtype=dtype)),
                             torch.zeros(self.dimension, 1, device=device, dtype=dtype)], dim=1)
            rho = rho.unsqueeze(0).unsqueeze(0).repeat(B, num_subjects, 1, 1)
        else:
            num_subjects = kwargs.pop('num_subjects', len(spacings))
            assert len(spacings) == num_subjects
            rhos = []
            for i in range(num_subjects):
                rhos.append(torch.cat([torch.diag(torch.as_tensor(spacings[i], device=device, dtype=dtype)),
                                       torch.zeros(self.dimension, 1, device=device, dtype=dtype)], dim=1))
            rho = torch.stack(rhos).unsqueeze(0).repeat(B, 1, 1, 1)

        return rho

    def forward(self, **kwargs):
        reg_level = kwargs.pop('reg_level', 0)
        res_level = kwargs.pop('res_level', -1)
        scale = kwargs.pop('scale', None)
        if scale is None:
            self.scale_factor = [2 ** (- s) for s in self.resolutions[reg_level][res_level]]
        else:
            if isinstance(scale, (int, float)):
                self.scale_factor = [2 ** (- scale)] * self.dimension
            elif isinstance(scale, (list, tuple)):
                assert len(scale) == self.dimension
                self.scale_factor = [2 ** (- s) for s in scale]
            else:
                raise NotImplementedError

        return self.warp_tensors(self.images, scale_factor=self.scale_factor, **kwargs)

    def warp_tensors(self, tensors, **kwargs):
        interp_mode = kwargs.pop('interp_mode', None)
        padding_mode = kwargs.pop('padding_mode', None)
        scale_factor = kwargs.pop('scale_factor', [1] * self.dimension)
        reset_params = kwargs.pop('reset_params', True)
        reset_masks = kwargs.pop('reset_masks', True)
        mode = kwargs.pop('mode', None)
        if mode is None:
            mode = 'bilinear' if self.dimension == 2 else 'trilinear'

        if reset_params:
            self.transform_params = self.get_transform_params()

        if isinstance(tensors, torch.Tensor):
            tensors = torch.unbind(tensors, dim=1)

        N = len(tensors)

        warped_tensors = []
        for i in range(N):
            w_t = self.transform(tensors[i],
                                 flows=self.transform_params[i]['abs_disp'],
                                 grid=self.transform_params[i]['ori_grid'],
                                 inv_thetas=self.transform_params[i]['inv_thetas'],
                                 interp_mode=interp_mode,
                                 padding_mode=padding_mode)
            warped_tensors.append(F.interpolate(w_t, scale_factor=scale_factor, mode=mode, align_corners=True))

        if reset_masks:
            with torch.no_grad():
                if self.masks is None:
                    self.warped_masks = None
                else:
                    if self.mask_frame is None:
                        self.warped_masks = torch.stack(self.warp_tensors(self.masks,
                                                                          interp_mode='nearest',
                                                                          padding_mode='zeros',
                                                                          reset_params=False,
                                                                          reset_masks=False),
                                                        dim=1)
                    else:
                        self.warped_masks = self.transform(self.masks,
                                                           flows=self.transform_params[self.mask_frame]['abs_disp'],
                                                           grid=self.transform_params[self.mask_frame]['ori_grid'],
                                                           inv_thetas=self.transform_params[self.mask_frame]['inv_thetas'],
                                                           interp_mode='nearest',
                                                           padding_mode='zeros')

                self.overlap_region = self._get_overlap_region()

        return warped_tensors

    def _get_overlap_mask(self):
        if self.masks is None:
            overlap_mask = torch.ones(self.B, 1, *self.img_size, device=self.device, dtype=self.dtype)
        else:
            if self.mask_frame is None:
                overlap_mask = torch.all(self.warped_masks, dim=1)
            else:
                overlap_mask = self.warped_masks

        return overlap_mask

    def get_transform_params(self):
        params = []
        thetas = self.predict_thetas()
        flows = self.predict_flows()
        for i in range(self.num_subjects):
            ori_grid = self.transform._get_new_locs(thetas=thetas[i][:2])
            abs_disp = self.transform.getAbsDisp(grid=ori_grid, thetas=thetas[i][2:], flows=flows[i])
            params.append({'ori_grid': ori_grid, 'thetas': thetas[i][2:], 'flows': flows[i],
                           'abs_disp': abs_disp,
                           'inv_thetas': [self._get_center_matrix(self.center),
                                          self._get_inverse_affine_matrix(self.rho[:, i])]
                           })

        if self.zero_avg_disp:
            avg_disp = torch.stack([p['abs_disp'] for p in params]).mean(dim=0)
            for i in range(self.num_subjects):
                params[i]['abs_disp'] = params[i]['abs_disp'] - avg_disp

        return params

    def predict_thetas(self):
        reg_thetas = []
        for i in range(self.num_subjects):
            if self.group2ref:
                subject_thetas = [self.rho[:, self.reference_frame]]
                subject_thetas.append(self._get_center_matrix(self.center.neg()))
                if not i == self.reference_frame:
                    for j in range(self.max_reg_level + 1):
                        if self.reg_level_type[j] == 'TRA':
                            subject_thetas.append(self._get_rigid_matrix(
                                translate=self.params['TRA'][:, i if i < self.reference_frame else i - 1]))
                        if self.reg_level_type[j] == 'RIG':
                            subject_thetas.append(self._get_rigid_matrix(
                                rotate=self.params['RIG_r'][:, i if i < self.reference_frame else i - 1],
                                translate=self.params['RIG_t'][:, i if i < self.reference_frame else i - 1]))
                        if self.reg_level_type[j] == 'AFF':
                            subject_thetas.append(self.params['AFF'][:, i if i < self.reference_frame else i - 1])
            else:
                subject_thetas = [self.rho[:, i]]
                subject_thetas.append(self._get_center_matrix(self.center.neg()))
                for j in range(self.max_reg_level + 1):
                    if self.reg_level_type[j] == 'TRA':
                        subject_thetas.append(self._get_rigid_matrix(translate=self.params['TRA'][:, i]))
                    if self.reg_level_type[j] == 'RIG':
                        subject_thetas.append(self._get_rigid_matrix(rotate=self.params['RIG_r'][:, i],
                                                                     translate=self.params['RIG_t'][:, i]))
                    if self.reg_level_type[j] == 'AFF':
                        subject_thetas.append(self.params['AFF'][:, i])
            reg_thetas.append(subject_thetas)

        return reg_thetas

    def predict_flows(self):
        if self.zero_avg_vec:
            avg_vec = torch.mean(self.vectors, dim=1)

        reg_flows = []
        for i in range(self.num_subjects):
            if self.group2ref and i == self.reference_frame:
                reg_flows.append(None)
            else:
                subject_flows = []
                for j in range(self.max_reg_level + 1):
                    if 'FFD' in self.reg_level_type[j]:
                        max_mesh = self.params[self.reg_level_type[j]][:,
                                   (i if i < self.reference_frame else i - 1) if self.group2ref else i]
                        mesh = max_mesh[(slice(None), slice(None),
                                         *[slice(s) for s in
                                           self.pred_mesh2flow[i][j-self.ffd_idx].control_point_size])]
                        subject_flows.append(self.pred_mesh2flow[i][j-self.ffd_idx](mesh))
                    if self.reg_level_type[j] == 'DDF':
                        subject_flows.append(
                            self.params['DDF'][:, (i if i < self.reference_frame else i - 1) if self.group2ref else i])
                    if self.reg_level_type[j] == 'SVF':
                        if self.zero_avg_vec:
                            subject_flows.append(
                                self.vec2flow(self.params['SVF'][:, (i if i < self.reference_frame else i - 1) if self.group2ref else i] - avg_vec))
                        else:
                            subject_flows.append(
                                self.vec2flow(self.params['SVF'][:, (i if i < self.reference_frame else i - 1) if self.group2ref else i]))
                reg_flows.append(self.transform.getComposedFlows(flows=subject_flows))

        if self.zero_avg_flow:
            tmp_flows = [f for f in reg_flows if f is not None]
            if len(tmp_flows) > 0:
                avg_flow = torch.mean(torch.stack(tmp_flows), dim=0)
                reg_flows = [f - avg_flow if f is not None else None for f in reg_flows]

        return reg_flows

    def _spatial_filter(self, *args, **kwargs):
        if self.dimension == 2:
            spatial_filter = utils.separable_filter2d
        elif self.dimension == 3:
            spatial_filter = utils.separable_filter3d
        else:
            raise NotImplementedError

        return spatial_filter(*args, **kwargs)

    def _get_rigid_matrix(self, rotate=None, translate=None):
        if self.dimension == 2:
            A = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.B, 1, 1)[:, :2]
            if translate is not None:
                A[:, :, 2] = translate
            if rotate is not None:
                A[:, 0, 0] = torch.cos(rotate[:, 0])
                A[:, 0, 1] = torch.sin(rotate[:, 0])
                A[:, 1, 0] = - torch.sin(rotate[:, 0])
                A[:, 1, 1] = torch.cos(rotate[:, 0])
            return A
        elif self.dimension == 3:
            T = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.B, 1, 1)
            Rx, Ry, Rz = T.clone(), T.clone(), T.clone()
            if translate is not None:
                T[:, :3, 3] = translate
            if rotate is not None:
                Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = torch.cos(rotate[:, 0]), - torch.sin(rotate[:, 0]), \
                                                                     torch.sin(rotate[:, 0]), torch.cos(rotate[:, 0])
                Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 2, 0], Ry[:, 2, 2] = torch.cos(rotate[:, 1]), torch.sin(rotate[:, 1]),\
                                                                     - torch.sin(rotate[:, 1]), torch.cos(rotate[:, 1])
                Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = torch.cos(rotate[:, 2]), - torch.sin(rotate[:, 2]),\
                                                                     torch.sin(rotate[:, 2]), torch.cos(rotate[:, 2])
            A = (T @ Rz @ Ry @ Rx)[:, :3]
            return A
        else:
            raise NotImplementedError

    def _get_center_matrix(self, center):
        C = torch.eye(self.dimension + 1,
                      device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.B, 1, 1)[:, :self.dimension]
        C[:, :, -1] = center
        return C

    def _get_inverse_affine_matrix(self, theta):
        aug_theta = torch.cat([theta,
                               torch.as_tensor([*[0]*self.dimension, 1],
                                               device=theta.device,
                                               dtype=theta.dtype).view(1, 1, -1).repeat(theta.shape[0], 1, 1)],
                              dim=1)

        return torch.inverse(aug_theta)[:, :self.dimension]

    def _get_identity_theta(self):
        theta = torch.cat([torch.diag(torch.ones(self.dimension, device=self.device, dtype=self.dtype)),
                           torch.zeros(self.dimension, 1, device=self.device, dtype=self.dtype)], dim=1)

        return theta

    def _get_regularization(self):
        r = 0
        for j in self.activated_reg_levels:
            if self.reg_level_type[j] not in ['TRA', 'AFF', 'RIG']:
                r += self.bending_energy(self.params[self.reg_level_type[j]]) * self.group_num

        return r

    def _get_overlap_region(self):
        with torch.no_grad():
            overlap_masks = torch.stack([self.transform.getOverlapMask(self.images[i],
                                                                       flows=self.transform_params[i]['abs_disp'],
                                                                       grid=self.transform_params[i]['ori_grid'],
                                                                       inv_thetas=self.transform_params[i]['inv_thetas']
                                                                       )
                                         for i in range(self.num_subjects)], dim=1)

            overlap_region = torch.all(overlap_masks, dim=1).to(self.dtype)

        return overlap_region


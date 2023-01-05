# -*- coding: utf-8 -*-
"""
Construct module for spatially-variant conditional density.

__author__ = Xinzhe Luo

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SVCD(nn.Module):
    def __init__(self, dimension, num_classes, num_bins=64, sample_rate=0.1, kernel_sigma=1,
                 eps=1e-8, **kwargs):
        super(SVCD, self).__init__()
        self.dimension = dimension
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.sample_rate = sample_rate
        self.kernel_sigma = kernel_sigma
        self._kernel_radius = math.ceil(2 * self.kernel_sigma)
        self.eps = eps
        self.kwargs = kwargs
        self.multiplier = self.kwargs.pop('multiplier', 1)
        self.bk_threshold = self.kwargs.pop('bk_threshold', float('-inf'))
        self.weight_bins = self.kwargs.pop('weight_bins', 1)
        self.pmf_noise = self.kwargs.pop('pmf_noise', 0)
        self.label_intensities = self.kwargs.pop('label_intensities', None)
        if self.label_intensities is None:
            self.label_intensities = np.linspace(0, 1000, self.num_classes).tolist()
        assert len(self.label_intensities) == self.num_classes
        self.method = self.kwargs.pop('method', "PW")
        assert self.method in ['PW', 'PV']

        if self.dimension == 2:
            self.scale_mode = 'bicubic'
        elif self.dimension == 3:
            self.scale_mode = 'trilinear'
        else:
            raise NotImplementedError

    def forward(self, image, weight=None, mask=None, return_density=False, **kwargs):
        scale = kwargs.pop('scale', 0)
        num_bins = kwargs.pop('num_bins', self.num_bins)

        if mask is None:
            mask = torch.ones_like(image)

        if weight is None:
            weight = torch.ones_like(image).repeat(1, self.num_classes, *[1] * self.dimension)

        original_image = image.clone()
        vol_shape = original_image.shape[2:]
        with torch.no_grad():
            original_mask = mask.to(torch.bool) & (image > self.bk_threshold)
            image_mask = original_mask.clone()

        if scale > 0:
            image = F.interpolate(image, scale_factor=2 ** (- scale), mode=self.scale_mode)
            weight = F.interpolate(weight, scale_factor=2 ** (- scale), mode=self.scale_mode)
            image_mask = F.interpolate(image_mask.to(image.dtype), scale_factor=2 ** (- scale),
                                       mode='nearest').to(torch.bool)

        W = weight.size(1)
        B = image.shape[0]

        masked_image = [torch.masked_select(image[i], mask=image_mask[i]) for i in range(B)]

        masked_weight = [torch.stack([torch.masked_select(weight[i, [k]], mask=image_mask[i])
                                      for k in range(W)]) for i in range(B)]

        sample_mask = [torch.rand_like(mi).le(self.sample_rate) for mi in masked_image]
        sampled_image = [torch.masked_select(masked_image[i], mask=sample_mask[i]) for i in range(B)]
        sampled_weight = [torch.stack([torch.masked_select(masked_weight[i][k], mask=sample_mask[i])
                                       for k in range(W)]) for i in range(B)]

        max_v = torch.stack([i.amax().detach() for i in sampled_image])
        min_v = torch.stack([i.amin().detach() for i in sampled_image])
        bin_width = (max_v - min_v) / num_bins
        pad_min_v = min_v - bin_width * self._kernel_radius
        bin_centers = torch.arange(num_bins + 2 * self._kernel_radius,
                                   dtype=image.dtype, device=image.device)

        bin_pos = [torch.divide(sampled_image[i] - pad_min_v[i], bin_width[i].clamp(min=self.eps)) for i in range(B)]
        bin_idx = [b.floor().clamp(min=self._kernel_radius,
                                   max=num_bins - 1 + self._kernel_radius).detach() for b in bin_pos]

        min_win_idx = [(b - self._kernel_radius + 1).to(torch.int64) for b in bin_idx]
        win_idx = [torch.stack([(mwi + i) for i in range(self._kernel_radius * 2)]) for mwi in min_win_idx]
        win_bin_centers = [torch.gather(bin_centers.unsqueeze(1).repeat(1, win_idx[i].size(1)),
                                        dim=0, index=win_idx[i])
                           for i in range(B)]

        win_kernel = [self._bspline_kernel(bin_pos[i].unsqueeze(0) - win_bin_centers[i]) for i in range(B)]

        bin_kernel = [torch.stack([torch.sum(win_idx[i].eq(idx) * win_kernel[i], dim=0)
                                   for idx in range(num_bins + self._kernel_radius * 2)], dim=0)
                      for i in range(B)]
        hist = torch.stack([torch.mm(sampled_weight[i], bin_kernel[i].transpose(0, 1)) for i in range(B)])

        density = hist / hist.sum(dim=(1, 2), keepdim=True).clamp(min=self.eps)
        if return_density:
            return density, pad_min_v, bin_width

        density = density.view(B, W, self.weight_bins, -1).sum(2)

        image_idx = torch.div(original_image.detach().reshape(B, 1, -1) - pad_min_v.view(-1, 1, 1),
                              bin_width.view(-1, 1, 1), rounding_mode='trunc')
        image_idx = image_idx.clamp(min=0,
                                    max=num_bins + self._kernel_radius * 2 - 1).to(torch.int64).repeat(1, W, 1)

        conditional_density = density / density.sum(dim=2, keepdim=True).clamp(min=self.eps)
        if self.pmf_noise > 0:
            conditional_density += torch.randn_like(conditional_density) * torch.amax(conditional_density, dim=-1, keepdim=True) * self.pmf_noise
            conditional_density /= conditional_density.sum(dim=-1, keepdim=True).detach()

        output = torch.gather(conditional_density, dim=2, index=image_idx)

        return output.view(B, -1, *vol_shape) * self.multiplier

    def compute_label_density(self, label, weight, mask=None, return_density=False, **kwargs):
        if self.method == 'PW':
            if mask is None:
                mask = torch.ones_like(label[:, [0]], dtype=torch.bool)
            else:
                mask = mask.to(torch.bool)
            B, C = label.shape[:2]

            masked_label = [torch.stack([torch.masked_select(label[i, [k]], mask=mask[i])
                                         for k in range(C)]) for i in range(B)]
            masked_weight = [torch.stack([torch.masked_select(weight[i, [k]], mask=mask[i])
                                          for k in range(C)]) for i in range(B)]
            sample_mask = [torch.rand_like(label[0]).le(self.sample_rate) for label in masked_label]
            sampled_label = [torch.stack([torch.masked_select(masked_label[i][k], mask=sample_mask[i])
                                          for k in range(C)]) for i in range(B)]
            sampled_weight = [torch.stack([torch.masked_select(masked_weight[i][k], mask=sample_mask[i])
                                          for k in range(C)]) for i in range(B)]

            joint_hist = torch.stack([torch.mm(sampled_label[i], sampled_weight[i].transpose(0, 1)) for i in range(B)])

        elif self.method == 'PV':
            sampling_weights = kwargs.pop('sampling_weights')
            if mask is None:
                mask = torch.ones_like(weight[:, [0]], dtype=torch.bool)
            else:
                mask = mask.to(torch.bool)
            B, C = weight.shape[:2]

            label_mask = mask.unsqueeze(1).repeat(1, 2 ** self.dimension, 1, *[1] * self.dimension)

            masked_label = [torch.stack([torch.masked_select(label[i, :, [k]],
                                                             mask=label_mask[i]).view(2 ** self.dimension, -1)
                                         for k in range(C)]) for i in range(B)]
            masked_weight = [torch.stack([torch.masked_select(weight[i, [k]], mask=mask[i])
                                          for k in range(C)]) for i in range(B)]
            masked_sampling_weights = [torch.masked_select(sampling_weights[i],
                                                           mask=label_mask[i].squeeze(1)).view(2 ** self.dimension, -1)
                                       for i in range(B)]
            sample_mask = [torch.rand_like(w[0]).le(self.sample_rate) for w in masked_weight]
            sample_label_mask = [m.unsqueeze(0).repeat(2 ** self.dimension, 1) for m in sample_mask]

            sampled_label = [torch.stack([torch.masked_select(masked_label[i][k],
                                                              mask=sample_label_mask[i]).view(2 ** self.dimension, -1)
                                          for k in range(C)]) for i in range(B)]
            sampled_weight = [torch.stack([torch.masked_select(masked_weight[i][k], mask=sample_mask[i])
                                           for k in range(C)]) for i in range(B)]
            sampled_sampling_weights = [torch.masked_select(masked_sampling_weights[i],
                                                            mask=sample_label_mask[i]).view(2 ** self.dimension, -1)
                                        for i in range(B)]

            joint_hist = torch.stack([torch.sum(sampled_label[i].unsqueeze(1) *
                                                sampled_weight[i].unsqueeze(1).unsqueeze(0) *
                                                sampled_sampling_weights[i].unsqueeze(0).unsqueeze(0),
                                                dim=(-2, -1)) for i in range(B)])
        else:
            raise NotImplementedError

        joint_dens = joint_hist / joint_hist.sum(dim=(1, 2), keepdim=True).clamp(min=self.eps)
        if return_density:
            ideal_hist = torch.stack([torch.mm(sampled_weight[i], sampled_weight[i].transpose(0, 1)) for i in range(B)])
            ideal_dens = ideal_hist / ideal_hist.sum(dim=(1, 2), keepdim=True).clamp(min=self.eps)
            return joint_dens, ideal_dens

        cond_dens = joint_dens / joint_dens.sum(dim=1, keepdim=True).clamp(min=self.eps)
        return torch.sum(cond_dens.view(B, C, C, *[1] * self.dimension) * label.unsqueeze(2), dim=1) * mask

    def _bspline_kernel(self, d):
        d /= self.kernel_sigma
        return torch.where(d.abs() < 1.,
                           (3. * d.abs() ** 3 - 6. * d.abs() ** 2 + 4.) / 6.,
                           torch.where(d.abs() < 2.,
                                       (2. - d.abs()) ** 3 / 6.,
                                       torch.zeros_like(d))
                           )

    def resample_density(self, density, image, mask=None, interp_mode='nearest', **kwargs):
        assert interp_mode in ['nearest', 'linear']
        pad_min_v = kwargs.pop('pad_min_v', None)
        bin_width = kwargs.pop('bin_width', None)
        num_bins = kwargs.pop('num_bins', self.num_bins)

        if mask is None:
            mask = torch.ones_like(image)

        vol_shape = image.shape[2:]
        B = image.shape[0]

        image_mask = mask.to(torch.bool) & (image > self.bk_threshold)
        W = density.size(1)

        if pad_min_v is None or bin_width is None:
            masked_image = [torch.masked_select(image[i], mask=image_mask[i]) for i in range(B)]
            max_v = torch.stack([i.amax() for i in masked_image])
            min_v = torch.stack([i.amin() for i in masked_image])
            bin_width = (max_v - min_v) / num_bins
            pad_min_v = min_v - bin_width * self._kernel_radius

        pad_min_v = pad_min_v.view(-1, 1, 1)
        bin_width = bin_width.view(-1, 1, 1)

        image = image.view(B, 1, -1)
        image_pos = torch.divide(image - pad_min_v, bin_width.clamp(min=self.eps))
        image_pos = image_pos.clamp(min=0, max=num_bins + 2 * self._kernel_radius - 1)

        if interp_mode == 'nearest':
            image_idx = torch.floor(image_pos).to(torch.int64)
            value = torch.gather(density, dim=2, index=image_idx.repeat(1, W, 1))
            return value.view(B, -1, *vol_shape) * image_mask

        elif interp_mode == 'linear':
            image_loc0 = torch.floor(image_pos).detach()
            image_loc1 = torch.clamp_max(image_loc0 + 1, num_bins + 2 * self._kernel_radius - 1)

            image_idx0 = image_loc0.to(torch.int64)
            image_idx1 = image_loc1.to(torch.int64)

            image_w1 = image_pos - image_loc0
            image_w0 = image_loc1 - image_pos

            value0 = torch.gather(density, dim=2, index=image_idx0.repeat(1, W, 1))
            value1 = torch.gather(density, dim=2, index=image_idx1.repeat(1, W, 1))

            interp_val = image_w0 * value0 + image_w1 * value1

            return interp_val.view(B, -1, *vol_shape) * image_mask
        else:
            raise NotImplementedError


# -*- coding: utf-8 -*-
"""
Construct module for congealing based groupwise registration.

__author__ = Xinzhe Luo

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CG(nn.Module):
    def __init__(self, dimension, win=1, num_bins=16, kernel_sigma=1, eps=1e-8, **kwargs):
        super(CG, self).__init__()
        self.dimension = dimension
        self.win = win
        assert self.win % 2 == 1
        self.num_bins = num_bins
        self.kernel_sigma = kernel_sigma
        self._kernel_radius = math.ceil(2 * self.kernel_sigma)
        self.eps = eps
        self.kwargs = kwargs
        self.multiplier = self.kwargs.pop('multiplier', 1)
        self.bk_threshold = self.kwargs.pop('bk_threshold', float('-inf'))

    def forward(self, images, mask=None, return_density=False):
        B, N = images.shape[:2]
        vol_shape = images.shape[3:]

        if mask is None:
            mask = torch.ones_like(images.squeeze(2))

        image_mask = mask.to(torch.bool) & (images.squeeze(2) > self.bk_threshold)

        if self.win > 1:
            if self.dimension == 2:
                assert len(vol_shape) == 2
                cylinder = F.unfold(images.squeeze(2), kernel_size=self.win, padding=self.win // 2)
            elif self.dimension == 3:
                assert len(vol_shape) == 3
                kernel = self._get_ones_kernel().repeat(N, 1, 1, 1, 1).to(images.device)
                cylinder = F.conv3d(images.squeeze(2), weight=kernel, padding=self.win // 2, groups=N)
                cylinder = cylinder.view(B, N * self.win ** 3, -1)
            else:
                raise NotImplementedError
        else:
            cylinder = images.view(B, N, -1)

        max_v = cylinder.amax(dim=1, keepdim=True).detach()
        min_v = cylinder.amin(dim=1, keepdim=True).detach()
        bin_width = (max_v -  min_v) / self.num_bins
        pad_min_v = min_v - bin_width * self._kernel_radius
        bin_center = torch.arange(self.num_bins + 2 * self._kernel_radius, dtype=images.dtype, device=images.device)

        bin_pos = (cylinder - pad_min_v) / bin_width.clamp(min=self.eps)
        bin_idx = bin_pos.detach().floor().clamp(min=self._kernel_radius,
                                                 max=self._kernel_radius + self.num_bins - 1)
        min_win_idx = bin_idx.sub(self._kernel_radius - 1).to(torch.int64)
        win_idx = torch.stack([min_win_idx.add(r) for r in range(self._kernel_radius * 2)])

        win_bin_center = torch.gather(bin_center.view(-1, 1, 1, 1).repeat(1, *win_idx.shape[1:]),
                                      dim=0, index=win_idx)

        win_weight = self._bspline_kernel(bin_pos.unsqueeze(0) - win_bin_center)

        bin_weight = torch.stack([torch.sum(win_idx.eq(i) * win_weight, dim=0)
                                  for i in range(self.num_bins + self._kernel_radius * 2)])
        hist = bin_weight.sum(2)
        density = hist / hist.sum(dim=0, keepdims=True).clamp(min=self.eps)

        if return_density:
            return density.view(-1, B, *vol_shape)

        image_idx = torch.floor_divide(images.squeeze(2) - pad_min_v.view(B, 1, *vol_shape),
                                       bin_width.clamp(self.eps).view(B, 1, *vol_shape))
        image_idx = image_idx.clamp(min=0, max=self._kernel_radius * 2 + self.num_bins - 1).to(torch.int64)

        output = torch.gather(density.transpose(0, 1).view(B, -1, *vol_shape),
                              dim=1, index=image_idx)

        return output * image_mask * self.multiplier

    def loss(self, images, mask=None):
        p = self.forward(images, mask, return_density=True)
        p_lnp = p * p.clamp(min=self.eps).log()

        if mask is None:
            return p_lnp.sum(0).mean().neg()
        else:
            mask = mask.transpose(0, 1)
            return p_lnp.sum(dim=0, keepdims=True).mul(mask).sum().div(mask.sum()).neg()

    def loss2(self, images, mask=None):
        B, N = images.shape[:2]
        vol_shape = images.shape[3:]

        if mask is None:
            mask = torch.ones_like(images.squeeze(2))

        if self.win > 1:
            if self.dimension == 2:
                assert len(vol_shape) == 2
                cylinder = F.unfold(images.squeeze(2), kernel_size=self.win, padding=self.win // 2)
            elif self.dimension == 3:
                assert len(vol_shape) == 3
                kernel = self._get_ones_kernel().repeat(N, 1, 1, 1, 1).to(images.device)
                cylinder = F.conv3d(images.squeeze(2), weight=kernel, padding=self.win // 2, groups=N)
                cylinder = cylinder.view(B, N * self.win ** 3, -1)
            else:
                raise NotImplementedError
        else:
            cylinder = images.view(B, N, -1)

        d = cylinder.unsqueeze(1) - cylinder.unsqueeze(2)
        p = self._gauss_kernel(d).mean(dim=1)
        ln_p = p.clamp(min=self.eps).log().view(B, -1, *vol_shape)

        return ln_p.mean(dim=1, keepdim=True).mul(mask).sum().div(mask.sum()).neg()

    def _bspline_kernel(self, d):
        d /= self.kernel_sigma
        return torch.where(d.abs() < 1.,
                           (3. * d.abs() ** 3 - 6. * d.abs() ** 2 + 4.) / 6.,
                           torch.where(d.abs() < 2.,
                                       (2. - d.abs()) ** 3 / 6.,
                                       torch.zeros_like(d))
                           )

    def _gauss_kernel(self, d, sigma=None):
        if sigma is None:
            sigma = self.kernel_sigma
        if sigma == 0:
            return torch.ones_like(d)
        else:
            return 1 / (math.sqrt(2 * math.pi) * sigma) * d.square().div(2 * sigma ** 2).neg().exp()

    def _box_kernel(self, d):
        d /= self.kernel_sigma
        return torch.where(d.abs() < 0.5, 1, 0)

    def _get_ones_kernel(self):
        win_size = self.win ** self.dimension
        ones = [torch.tensor([0] * i + [1] + [0] * (win_size - 1 - i), dtype=torch.float32) for i in range(win_size)]
        return torch.stack([o.view(*[self.win] * self.dimension) for o in ones]).unsqueeze(1)

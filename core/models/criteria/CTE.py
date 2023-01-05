# -*- coding: utf-8 -*-
"""
Construct module for conditional template entropy based groupwise registration.

__author__ = Xinzhe Luo

"""
import torch
import torch.nn as nn
import math
import MI


class CTE(nn.Module):
    def __init__(self, dimension, num_bins=64, sample_rate=1, kernel_sigma=1, eps=1e-8, **kwargs):
        super(CTE, self).__init__()
        self.dimension = dimension
        self.num_bins = num_bins
        self.sample_rate = sample_rate
        self.kernel_sigma = kernel_sigma
        self._kernel_radius = math.ceil(2 * self.kernel_sigma)
        self.eps = eps
        self.kwargs = kwargs
        self.bk_threshold = self.kwargs.pop('bk_threshold', float('-inf'))

        self.mi = MI(self.dimension, self.num_bins, self.sample_rate, self.kernel_sigma,
                     bk_threshold=self.bk_threshold, eps=self.eps)

    def forward(self, images, mask=None):
        B, N = images.shape[:2]
        vol_shape = images.shape[3:]

        if mask is None:
            mask = torch.ones(B, 1, *vol_shape, device=images.device, dtype=images.dtype)

        mask = mask.view(B, 1, -1)
        I = images.view(B, N, -1)
        masked_I = torch.masked_select(I, mask=mask.to(torch.bool)).view(B, N, -1)

        std, mean = torch.std_mean(masked_I, dim=-1, keepdim=True)
        X = I.sub(mean.detach()).div(std.detach())
        A = torch.bmm(X, X.transpose(1, 2)).div(X.shape[-1])

        _, Q = torch.linalg.eigh(A)
        eigvec = Q[:, :, [-1]]

        template = I.mul(eigvec).sum(dim=1, keepdim=True).view(B, 1, *vol_shape)
        return template

    def loss(self, images, mask=None, **kwargs):
        N = images.shape[1]
        template = self.forward(images, mask=mask)

        loss = 0.
        for i in range(N):
            loss += self.mi.ce(source=template, target=images[:, i], mask=mask, **kwargs)

        return loss


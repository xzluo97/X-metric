# -*- coding: utf-8 -*-
"""
Affine grid module for image registration.

__author__ = Xinzhe Luo
__version__ = 0.1

"""
import torch
import torch.nn as nn
# import torch.nn.functional as F


class AffineGrid(nn.Module):
    def __init__(self, size):
        super(AffineGrid, self).__init__()
        self.size = size
        self.dimension = len(self.size)

        vectors = [torch.arange(0, s) for s in self.size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.to(torch.float32)
        self.register_buffer('grid', grid)

    def forward(self, thetas):
        return self.get_new_locs(self.grid, thetas)

    def get_new_locs(self, grid, thetas):
        B = grid.shape[0]
        size = grid.shape[2:]

        if isinstance(thetas, torch.Tensor):
            assert thetas.shape[1] == self.dimension
            theta = thetas
        elif isinstance(thetas, (tuple, list)):
            if len(thetas) > 0:
                new_theta = self._augment_theta(thetas[0])
                for theta in thetas[1:]:
                    if theta is not None:
                        new_theta = torch.matmul(self._augment_theta(theta), new_theta)
                theta = new_theta[:, :self.dimension]
            else:
                return grid.clone()
        else:
            raise NotImplementedError

        mesh = grid.view(B, self.dimension, -1)
        mesh_aug = torch.cat([mesh, torch.ones(B, 1, mesh.size(2), dtype=mesh.dtype, device=mesh.device)],
                             dim=1)

        affine_grid = torch.matmul(theta, mesh_aug)

        return affine_grid.view(B, self.dimension, *size)

    def _augment_theta(self, theta):
        assert theta.shape[1] == self.dimension, 'Expected shape for theta in 1-axis: %s, ' \
                                                 'got %s!' % (self.dimension, theta.shape[1])
        if self.dimension == 2:
            aug_tensor = torch.as_tensor([[0, 0, 1]], device=theta.device, dtype=theta.dtype)
        elif self.dimension == 3:
            aug_tensor = torch.as_tensor([[0, 0, 0, 1]], device=theta.device, dtype=theta.dtype)
        else:
            raise NotImplementedError

        aug_theta = torch.cat([theta, aug_tensor.unsqueeze(0).repeat(theta.shape[0], 1, 1)], dim=1)
        return aug_theta


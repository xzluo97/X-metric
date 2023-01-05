# -*- coding: utf-8 -*-
"""
Generate free-form deformations.

__author__ = Xinzhe Luo
__credit__ = Xin Wang
__version__ = 0.1

"""
import torch
import torch.nn as nn
import numpy as np


class FFDGenerator(nn.Module):
    def __init__(self, size, ffd_spacing, **kwargs):
        super(FFDGenerator, self).__init__()
        self.size = size
        self.dimension = len(self.size)
        if isinstance(ffd_spacing, (tuple, list)):
            if len(ffd_spacing) == 1:
                ffd_spacing = ffd_spacing * self.dimension
            assert len(ffd_spacing) == self.dimension
            self.ffd_spacing = ffd_spacing
        elif isinstance(ffd_spacing, (int, float)):
            self.ffd_spacing = (ffd_spacing,) * self.dimension
        else:
            raise NotImplementedError
        self.kwargs = kwargs
        img_spacing = kwargs.pop('img_spacing', None)
        if img_spacing is None:
            self.img_spacing = [1] * self.dimension
        else:
            if isinstance(img_spacing, (tuple, list, np.ndarray)):
                assert len(img_spacing) == self.dimension
                self.img_spacing = img_spacing
            elif isinstance(img_spacing, (int, float)):
                self.img_spacing = (img_spacing,) * self.dimension
            else:
                raise NotImplementedError

        self.control_point_size = [int((self.size[i] * self.img_spacing[i] - 1) // self.ffd_spacing[i] + 4)
                                   for i in range(self.dimension)]

        vectors = [torch.arange(0., self.size[i] * self.img_spacing[i], self.img_spacing[i]) / self.ffd_spacing[i]
                   for i in range(self.dimension)]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid_floor = torch.floor(grid)
        grid_decimal = grid - grid_floor
        self.register_buffer('grid_floor', grid_floor, persistent=False)
        self.register_buffer('grid_decimal', grid_decimal, persistent=False)

        mesh_indices = torch.stack(torch.meshgrid(*[torch.arange(4)] * self.dimension)).flatten(1)
        self.register_buffer('mesh_indices', mesh_indices.T, persistent=False)

    def forward(self, mesh):
        mesh_shape = mesh.shape[2:]
        assert len(mesh_shape) == self.dimension
        assert all([mesh_shape[i] == self.control_point_size[i] for i in range(self.dimension)]), \
            "Expected control point size %s, got %s!" % (self.control_point_size, list(mesh_shape))

        flow = torch.zeros(*mesh.shape[:2], *self.size, dtype=mesh.dtype, device=mesh.device)
        for idx in self.mesh_indices:
            B = self.Bspline(self.grid_decimal, idx)
            pivots = self.grid_floor.squeeze(0) + idx.view(self.dimension, *[1] * self.dimension)
            pivots = pivots.to(torch.int64)
            if self.dimension == 2:
                flow += B.prod(dim=1, keepdim=True) * mesh[:, :, pivots[0], pivots[1]]
            elif self.dimension == 3:
                flow += B.prod(dim=1, keepdim=True) * mesh[:, :, pivots[0], pivots[1], pivots[2]]
            else:
                raise NotImplementedError
        return flow

    def Bspline(self, decimal, idx):
        idx = idx.view(self.dimension, *[1] * self.dimension).unsqueeze(0)

        return torch.where(idx == 0,
                           (1 - decimal) ** 3 / 6,
                           torch.where(idx == 1,
                                       decimal ** 3 / 2 - decimal ** 2 + 2 / 3,
                                       torch.where(idx == 2,
                                                   - decimal ** 3 / 2 + decimal ** 2 / 2 + decimal / 2 + 1 / 6,
                                                   torch.where(idx == 3,
                                                               decimal ** 3 / 6,
                                                               torch.zeros_like(decimal)
                                                               )
                                                   )
                                       )
                           )


# -*- coding: utf-8 -*-
"""
Modules for loss computation.

__author__ = "Xinzhe Luo"
__version__ = 0.1

"""

import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-5, **kwargs):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.kwargs = kwargs

    def forward(self, y_pred, y_true, mask=None):
        assert y_pred.size() == y_true.size(), "The prediction and ground truth must be of the same size!"
        n_dims = y_true.ndimension()
        if mask is None:
            mask = torch.ones_like(y_true[:, [0]])
        numerator = 2 * torch.sum(y_true * y_pred * mask, dim=[0,] + list(range(2, n_dims)))
        denominator = torch.sum((y_true ** 2 + y_pred ** 2) * mask, dim=[0,] + list(range(2, n_dims)))
        dice = torch.mean(numerator / (denominator + self.eps))

        return 1 - dice

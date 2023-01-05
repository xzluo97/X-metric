# -*- coding: utf-8 -*-
"""
Modules for computing metrics.

@author: Xinzhe Luo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from register.SpatialTransformer import SpatialTransformer


def get_segmentation(predictor, mode='torch'):
    assert mode in ['torch', 'np'], "The mode must be either 'torch' or 'np'!"
    if mode == 'torch':
        assert isinstance(predictor, torch.Tensor)
        ndim = predictor.dim()
        return F.one_hot(torch.argmax(predictor, dim=1),
                         predictor.shape[1]).permute(0, -1, *range(1, ndim - 1)).to(torch.float32)

    elif mode == 'np':
        assert isinstance(predictor, np.ndarray)
        ndim = predictor.ndim
        return np.eye(predictor.shape[1])[np.argmax(predictor, axis=1)].transpose((0, -1, *range(1, ndim - 1)))


class GWI(nn.Module):
    def __init__(self, unbiased=True, **kwargs):
        super(GWI, self).__init__()
        self.unbiased = unbiased

        self.transform = SpatialTransformer(**kwargs)

    def forward(self, init_flows, pred_flows, masks=None):
        assert init_flows.shape == pred_flows.shape
        if masks is not None:
            assert init_flows.ndim == masks.ndim
            assert init_flows.shape[:2] == masks.shape[:2]
            assert init_flows.shape[3:] == masks.shape[3:]
            assert masks.shape[2] == 1
        n = init_flows.shape[1]
        b = init_flows.shape[0]

        with torch.no_grad():
            if masks is None:
                masks = torch.ones_like(init_flows[:, [0]]).repeat(1, n, *[1] * (init_flows.ndim - 2))
            else:
                masks = masks.to(init_flows.dtype)
            masks_warped = self.transform(rearrange(masks, 'B M ... -> (B M) ...'),
                                          rearrange(pred_flows, 'B M ... -> (B M) ...'), mode='nearest')
            masks_warped = rearrange(masks_warped, '(B M) ... -> B M ...', M=n)

            res = self.transform.getComposedFlows(
                flows=[rearrange(pred_flows, 'B M ... -> (B M) ...'),
                       rearrange(init_flows, 'B M ... -> (B M) ...')])
            res = rearrange(res, '(B M) ... -> B M ...', M=n)
            if self.unbiased:
                res -= torch.mean(res, dim=1, keepdim=True)
                init_flows -= torch.mean(init_flows, dim=1, keepdim=True)

            dims_sum = list(range(2, init_flows.ndim))
            pre_gWI = torch.sqrt(torch.sum((init_flows**2) * masks, dim=dims_sum) / masks.sum(dim=dims_sum))
            post_gWI = torch.sqrt(torch.sum((res**2) * masks_warped, dim=dims_sum) / masks_warped.sum(dim=dims_sum))
        assert list(pre_gWI.shape) == list(post_gWI.shape) == [b, n]

        return pre_gWI, post_gWI


class OverlapMetrics(nn.Module):
    def __init__(self, eps=1, mode='torch', type='average_foreground_dice', **kwargs):
        super(OverlapMetrics, self).__init__()
        self.eps = eps
        self.mode = mode
        self.type = type
        self.kwargs = kwargs
        self.class_index = kwargs.get('class_index', None)
        self.channel_last = kwargs.get('channel_last', False)

        assert mode in ['torch', 'np'], "The mode must be either 'torch' or 'np'!"
        assert type in ['average_foreground_dice', 'class_specific_dice', 'average_foreground_jaccard']

    def forward(self, y_true, y_seg):
        if self.mode == 'np':
            y_true = torch.from_numpy(y_true)
            y_seg = torch.from_numpy(y_seg)

        dimension = y_true.dim() - 2

        if self.channel_last:
            y_true = y_true.permute(0, -1, *list(range(1, 1 + dimension)))
            y_seg = y_seg.permute(0, -1, *list(range(1, 1 + dimension)))

        assert y_true.size()[1:] == y_seg.size()[1:], "The ground truth and prediction must be of equal shape! " \
                                                      "Ground truth shape: %s, " \
                                                      "prediction shape: %s" % (tuple(y_true.size()),
                                                                                tuple(y_seg.size()))

        n_class = y_seg.size()[1]

        y_seg = get_segmentation(y_seg, mode='torch')

        if self.type == 'average_foreground_dice':
            dice = []
            for i in range(1, n_class):
                top = 2 * torch.sum(y_true[:, i] * y_seg[:, i], dim=tuple(range(1, 1 + dimension)))
                bottom = torch.sum(y_true[:, i] + y_seg[:, i], dim=tuple(range(1, 1 + dimension)))
                dice.append(top.clamp(min=self.eps) / bottom.clamp(min=self.eps))

            metric = torch.stack(dice, dim=1).mean(dim=1)

        elif self.type == 'class_specific_dice':
            assert self.class_index is not None, "The class index must be provided!"
            top = 2 * torch.sum(y_true[:, self.class_index] * y_seg[:, self.class_index],
                                dim=tuple(range(1, 1 + dimension)))
            bottom = torch.sum(y_true[:, self.class_index] + y_seg[:, self.class_index],
                               dim=tuple(range(1, 1 + dimension)))
            metric = top.clamp(min=self.eps) / bottom.clamp(min=self.eps)

        elif self.type == 'average_foreground_jaccard':
            jaccard = []
            y_true = y_true.type(torch.bool)
            y_seg = y_seg.type(torch.bool)
            for i in range(1, n_class):
                top = torch.sum(y_true[:, i] & y_seg[:, i], dtype=torch.float32, dim=tuple(range(1, 1 + dimension)))
                bottom = torch.sum(y_true[:, i] | y_seg[:, i], dtype=torch.float32, dim=tuple(range(1, 1 + dimension)))
                jaccard += top.clamp(min=self.eps) / bottom.clamp(min=self.eps)

            metric = torch.stack(jaccard, dim=1).mean(dim=1)

        else:
            raise ValueError("Unknown overlap metric: %s" % self.type)

        if self.mode == 'np':
            return metric.detach().cpu().numpy()

        return metric


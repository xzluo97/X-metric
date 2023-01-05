# -*- coding: utf-8 -*-
"""
Modules for loss computation.

__author__ = "Xinzhe Luo"
__version__ = 0.1

"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class CrossEntropyLoss(nn.Module):
    def __init__(self, class_weight=None, ignore_index=None, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.eps = kwargs.pop('eps', 1e-8)

    def forward(self, label, prob=None, logit=None, mask=None):
        if logit is not None:
            assert logit.size() == label.size(), "The logits and labels must be of the same size!"
        if prob is not None:
            assert prob.size() == label.size(), "The probs and labels must be of the same size!"

        if self.ignore_index is not None:
            if logit is not None:
                logit = logit[:, list(range(self.ignore_index)) + list(range(self.ignore_index + 1, logit.size()[1])), ]
            if prob is not None:
                prob = prob[:, list(range(self.ignore_index)) + list(range(self.ignore_index + 1, logit.size()[1])), ]
            label = label[:, list(range(self.ignore_index)) + list(range(self.ignore_index + 1, label.size()[1])), ]

        if self.class_weight is not None:
            weight = torch.tensor(self.class_weight, dtype=torch.float32)
            if logit is not None:
                assert weight.dim() == logit.size()[1]
                weight = weight.view(1, -1, *[1]*(logit.dim() - 2))
            if prob is not None:
                assert weight.dim() == prob.size()[1]
                weight = weight.view(1, -1, *[1]*(prob.dim() - 2))
        else:
            weight = 1.

        if logit is not None:
            log_prob = F.log_softmax(logit, dim=1)
        if prob is not None:
            log_prob = prob.clamp(min=self.eps).log()

        if mask is None:
            mask = torch.ones_like(label[:, [0]])
        loss = - torch.sum(torch.sum(weight * label * log_prob * mask, dim=1)) / torch.sum(mask)

        return loss

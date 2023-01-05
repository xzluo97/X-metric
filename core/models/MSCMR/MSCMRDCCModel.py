# -*- coding: utf-8 -*-
"""
Deep Combined Computing.

__author__ = Xinzhe Luo
__version__ = 0.1

"""
import torch
from ..algorithms.XCoRegDCCModel import XCoRegDCCModel
from core.metrics import OverlapMetrics
import numpy as np


class MSCMRDCCModel(XCoRegDCCModel):
    def __init__(self, num_classes=4, eps=1e-8, **kwargs):
        super(MSCMRDCCModel, self).__init__(num_classes, eps, **kwargs)

        self.Dice = OverlapMetrics()
        self.MyoDice = OverlapMetrics(type='class_specific_dice', class_index=1)
        self.LVDice = OverlapMetrics(type='class_specific_dice', class_index=2)
        self.RVDice = OverlapMetrics(type='class_specific_dice', class_index=3)

    def evaluateRegDice(self, labels=None, phantom=None, reduce_batch=True):
        if isinstance(labels, torch.Tensor):
            labels = torch.unbind(labels, dim=1)
        if labels is None:
            assert phantom is not None
            labels = [phantom] * self.num_subjects
        with torch.no_grad():
            pre_Dice = []
            pre_MyoDice = []
            pre_LVDice = []
            pre_RVDice = []
            post_Dice = []
            post_MyoDice = []
            post_LVDice = []
            post_RVDice = []
            if self.init_flows is None:
                init_labels = labels
                warped_labels = [self.transform(labels[i], flows=self.flows[:, i], interp_mode='nearest')
                                 for i in range(self.num_subjects)]
            else:
                init_labels = [self.transform(labels[i], flows=self.init_flows[:, i], interp_mode='nearest')
                               for i in range(self.num_subjects)]
                warped_labels = [self.transform(init_labels[i], flows=self.flows[:, i], interp_mode='nearest')
                                 for i in range(self.num_subjects)]
            for i in range(self.num_subjects):
                pre_Dice.append(np.mean(np.asarray([self.Dice(init_labels[j], init_labels[i]).tolist()
                                                    for j in range(self.num_subjects) if j != i]), axis=0))
                pre_MyoDice.append(np.mean(np.asarray([self.MyoDice(init_labels[j], init_labels[i]).tolist()
                                                    for j in range(self.num_subjects) if j != i]), axis=0))
                pre_LVDice.append(np.mean(np.asarray([self.LVDice(init_labels[j], init_labels[i]).tolist()
                                                    for j in range(self.num_subjects) if j != i]), axis=0))
                pre_RVDice.append(np.mean(np.asarray([self.RVDice(init_labels[j], init_labels[i]).tolist()
                                                    for j in range(self.num_subjects) if j != i]), axis=0))
                post_Dice.append(np.mean(np.asarray([self.Dice(warped_labels[j], warped_labels[i]).tolist()
                                                     for j in range(self.num_subjects) if j != i]), axis=0))
                post_MyoDice.append(np.mean(np.asarray([self.MyoDice(warped_labels[j], warped_labels[i]).tolist()
                                                     for j in range(self.num_subjects) if j != i]), axis=0))
                post_LVDice.append(np.mean(np.asarray([self.LVDice(warped_labels[j], warped_labels[i]).tolist()
                                                     for j in range(self.num_subjects) if j != i]), axis=0))
                post_RVDice.append(np.mean(np.asarray([self.RVDice(warped_labels[j], warped_labels[i]).tolist()
                                                     for j in range(self.num_subjects) if j != i]), axis=0))

        pre_Dice = np.asarray(pre_Dice).transpose((1, 0))
        pre_MyoDice = np.asarray(pre_MyoDice).transpose((1, 0))
        pre_LVDice = np.asarray(pre_LVDice).transpose((1, 0))
        pre_RVDice = np.asarray(pre_RVDice).transpose((1, 0))
        post_Dice = np.asarray(post_Dice).transpose((1, 0))
        post_MyoDice = np.asarray(post_MyoDice).transpose((1, 0))
        post_LVDice = np.asarray(post_LVDice).transpose((1, 0))
        post_RVDice = np.asarray(post_RVDice).transpose((1, 0))

        if reduce_batch:
            pre_Dice = np.mean(pre_Dice, axis=0)
            pre_MyoDice = np.mean(pre_MyoDice, axis=0)
            pre_LVDice = np.mean(pre_LVDice, axis=0)
            pre_RVDice = np.mean(pre_RVDice, axis=0)
            post_Dice = np.mean(post_Dice, axis=0)
            post_MyoDice = np.mean(post_MyoDice, axis=0)
            post_LVDice = np.mean(post_LVDice, axis=0)
            post_RVDice = np.mean(post_RVDice, axis=0)

        return {'Dice': pre_Dice, 'Myo-Dice': pre_MyoDice, 'LV-Dice': pre_LVDice, 'RV-Dice': pre_RVDice}, \
               {'Dice': post_Dice, 'Myo-Dice': post_MyoDice, 'LV-Dice': post_LVDice, 'RV-Dice': post_RVDice}

    def evaluateSegDice(self, labels=None, phantom=None, reduce_batch=True, **kwargs):
        if isinstance(labels, torch.Tensor):
            labels = torch.unbind(labels, dim=1)
        if labels is None:
            assert phantom is not None
            labels = [phantom] * self.num_subjects

        if self.use_atlas:
            atlas_prob = kwargs.pop('atlas_prob')
            warped_atlas = self.transform(atlas_prob, flows=self.flows[:, -1])
        else:
            warped_atlas = None

        with torch.no_grad():
            warped_seg_probs = self.transform_probs()
            init_posterior = self.get_posterior(self.warped_images, warped_probs=warped_seg_probs,
                                                warped_atlas=warped_atlas, use_probs=True)
            self.pred_posterior = self.update_posterior(self.warped_images, init_posterior, warped_probs=warped_seg_probs,
                                                        warped_atlas=warped_atlas, use_probs=True, T=self.update_steps)

            Dice = []
            MyoDice = []
            LVDice = []
            RVDice = []

            warped_labels = self.transform_images(labels, self.flows, self.init_flows, interp_mode='nearest')[1]

            for i in range(self.num_subjects):
                Dice.append(self.Dice(warped_labels[i], self.pred_posterior).tolist())
                MyoDice.append(self.MyoDice(warped_labels[i], self.pred_posterior).tolist())
                LVDice.append(self.LVDice(warped_labels[i], self.pred_posterior).tolist())
                RVDice.append(self.RVDice(warped_labels[i], self.pred_posterior).tolist())

            Dice = np.asarray(Dice).transpose((1, 0))
            MyoDice = np.asarray(MyoDice).transpose((1, 0))
            LVDice = np.asarray(LVDice).transpose((1, 0))
            RVDice = np.asarray(RVDice).transpose((1, 0))

            if reduce_batch:
                Dice = np.mean(Dice, axis=0)
                MyoDice = np.mean(MyoDice, axis=0)
                LVDice = np.mean(LVDice, axis=0)
                RVDice = np.mean(RVDice, axis=0)

        return {'Dice': Dice, 'Myo-Dice': MyoDice, 'LV-Dice': LVDice, 'RV-Dice': RVDice}

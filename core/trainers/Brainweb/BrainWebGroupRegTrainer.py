# -*- coding: utf-8 -*-
"""
Iterative groupwise registration with X-CoReg on Brainweb dataset.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import os
import torch
import random
from core import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.models.Brainweb.BrainWebRegModel import BrainWebRegModel
from core.data.data_providers.Brainweb.BrainwebImageProvider import ImageDataProvider
from ..genericTrainer import Trainer
from core.configs.ConfigBrainWeb import config
from torch.utils.data import DataLoader, Subset
from core.register.LocalDisplacementEnergy import JacobianDeterminant
from core.data import image_utils


jet_cm = plt.get_cmap('jet')
gray_cm = plt.get_cmap('gray')


class IterGroupRegTrainer(Trainer):
    def train(self, train_dataset, valid_dataset=None, epochs=100, display_step=1, device='cuda:0', **kwargs):
        steps = kwargs.pop('steps', [50])
        assert len(steps) == self.net.reg.num_reg_levels
        cum_steps = np.cumsum(steps)
        if isinstance(self.lr, float):
            lr = [self.lr] * self.net.reg.num_reg_levels
        elif isinstance(self.lr, (tuple, list)):
            if len(self.lr) == 1:
                lr = list(self.lr) * self.net.reg.num_reg_levels
            else:
                assert len(self.lr) == self.net.reg.num_reg_levels
                lr = self.lr
        else:
            raise NotImplementedError
        num_workers = kwargs.pop('num_workers', 0)
        data_collate_fn = kwargs.pop('data_collate_fn', train_dataset.data_collate_fn)
        phantom = kwargs.pop('phantom').to(device)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                                  collate_fn=data_collate_fn, pin_memory=False)

        self.logger.info("------ Number of training pairs %s ------" % len(train_loader))

        self.writer = self._get_writer(self.save_path)

        model = self.net.to(device)

        pre_metrics = {}
        pre_metrics.update(dict(zip(['Pre-reg %s foreground warping index' % m.upper()
                                     for m in train_dataset.modalities],
                                    [[] for _ in range(len(train_dataset.modalities))])))
        pre_metrics.update(dict(zip(['Pre-reg %s overlap warping index' % m.upper()
                                     for m in train_dataset.modalities],
                                    [[] for _ in range(len(train_dataset.modalities))])))
        pre_metrics.update(dict(zip(['Pre-reg %s Average Dice' % m.upper()
                                     for m in train_dataset.modalities],
                                    [[] for _ in range(len(train_dataset.modalities))])))
        pre_metrics.update(dict(zip(['Pre-reg %s CSF Dice' % m.upper()
                                     for m in train_dataset.modalities],
                                    [[] for _ in range(len(train_dataset.modalities))])))
        pre_metrics.update(dict(zip(['Pre-reg %s GM Dice' % m.upper()
                                     for m in train_dataset.modalities],
                                    [[] for _ in range(len(train_dataset.modalities))])))
        pre_metrics.update(dict(zip(['Pre-reg %s WM Dice' % m.upper()
                                     for m in train_dataset.modalities],
                                    [[] for _ in range(len(train_dataset.modalities))])))

        post_metrics = {}
        post_metrics.update(dict(zip(['Post-reg %s foreground warping index' % m.upper()
                                      for m in train_dataset.modalities],
                                     [[] for _ in range(len(train_dataset.modalities))])))
        post_metrics.update(dict(zip(['Post-reg %s overlap warping index' % m.upper()
                                      for m in train_dataset.modalities],
                                     [[] for _ in range(len(train_dataset.modalities))])))
        post_metrics.update(dict(zip(['Post-reg %s Average Dice' % m.upper()
                                      for m in train_dataset.modalities],
                                     [[] for _ in range(len(train_dataset.modalities))])))
        post_metrics.update(dict(zip(['Post-reg %s CSF Dice' % m.upper()
                                      for m in train_dataset.modalities],
                                     [[] for _ in range(len(train_dataset.modalities))])))
        post_metrics.update(dict(zip(['Post-reg %s GM Dice' % m.upper()
                                      for m in train_dataset.modalities],
                                     [[] for _ in range(len(train_dataset.modalities))])))
        post_metrics.update(dict(zip(['Post-reg %s WM Dice' % m.upper()
                                      for m in train_dataset.modalities],
                                     [[] for _ in range(len(train_dataset.modalities))])))

        indices = []

        for idx, data in enumerate(train_loader):
            pair_names = train_dataset.get_image_name(idx)
            ffd_names = [os.path.basename(name[-1])[:-7] for name in pair_names]
            indices.append('&'.join(ffd_names))

            images = data['images'].to(device)
            ffds = data['ffds'].to(device)
            modalities = data['modalities']
            affines = data['affines']
            headers = data['headers']

            model.init_model_params(images, ffds, phantom)

            opts = [self._get_optimizer([model.reg.params[model.reg.reg_level_type[j]]], lr=lr[j])
                    for j in range(model.reg.num_reg_levels)]

            with torch.no_grad():
                warped_images = model.reg()
                self.writer.add_images('Label/%s' % indices[-1],
                                       np.stack([utils.rgba2rgb(jet_cm(p))
                                                 for p in model.label.squeeze().cpu().numpy()]),
                                       global_step=-1, dataformats='NHWC'
                                       )
                for i in range(len(modalities[0])):
                    self.writer.add_images('WarpedImage%s/%s' % (modalities[0][i].upper(), ffd_names[i]),
                                           np.expand_dims(utils.rgba2rgb(gray_cm(utils.normalize_gray_img(
                                               warped_images[i].squeeze().cpu().numpy()))), axis=0),
                                           global_step=-1, dataformats='NHWC')

            for j in range(model.reg.num_reg_levels):
                model.reg.activate_params([j])
                opt = opts[j]
                for step in range(0 if j == 0 else cum_steps[j - 1], cum_steps[j]):

                    opt.zero_grad()

                    warped_images = model.reg()

                    loss = model.reg.loss_function(warped_images)
                    loss.backward()

                    opt.step()

                    if step % display_step == (display_step - 1):
                        self.writer.add_scalar('Loss/%s' % indices[-1], loss, global_step=step)
                        foreground_warping_indices, overlap_warping_indices,\
                            num_neg_jacobs, post_Dice = self._output_minibatch_stats(loss, model, idx, step, modalities)
                        self.writer.add_scalars('ForegroundWarpingIndex/%s' % indices[-1],
                                                foreground_warping_indices, global_step=step)
                        self.writer.add_scalars('OverlapWarpingIndex/%s' % indices[-1],
                                                overlap_warping_indices, global_step=step)
                        self.writer.add_scalar('#NJD/%s' % indices[-1], num_neg_jacobs, global_step=step)
                        self.writer.add_scalars('PostRegDice/%s' % indices[-1], post_Dice, global_step=step)
                        if model.model_type in ['XCoRegUn', 'XCoRegGT', 'GMM']:
                            self.writer.add_images('Posterior/%s' % indices[-1],
                                                   np.stack([utils.rgba2rgb(jet_cm(p)) for p in
                                                             self.reorder_posterior(
                                                                 model.reg.posterior).squeeze().detach().cpu().numpy()]),
                                                   global_step=step, dataformats='NHWC')
                        if model.model_type == 'CTE':
                            template = model.reg.predict_template(warped_images)
                            self.writer.add_images('TemplateImage/%s' % indices[-1],
                                                   np.expand_dims(utils.rgba2rgb(jet_cm(utils.normalize_gray_img(
                                                       template.squeeze().cpu().numpy()))), axis=0),
                                                   global_step=step, dataformats='NHWC')
                        for i in range(len(modalities[0])):
                            self.writer.add_images('WarpedImage%s/%s' % (modalities[0][i].upper(), ffd_names[i]),
                                                   np.expand_dims(utils.rgba2rgb(gray_cm(utils.normalize_gray_img(
                                                       warped_images[i].squeeze().detach().cpu().numpy()))), axis=0),
                                                   global_step=step, dataformats='NHWC')

            pre_foreground_warping_index, post_foreground_warping_index = model.evaluateForegroundWarpingIndex()
            pre_foreground_warping_indices = dict(zip(modalities[0], pre_foreground_warping_index))
            post_foreground_warping_indices = dict(zip(modalities[0], post_foreground_warping_index))
            pre_overlap_warping_index, post_overlap_warping_index = model.evaluateOverlapWarpingIndex()
            pre_overlap_warping_indices = dict(zip(modalities[0], pre_overlap_warping_index))
            post_overlap_warping_indices = dict(zip(modalities[0], post_overlap_warping_index))

            pre_dice, post_dice = model.evaluateOverlap()
            pre_Dice = dict(zip(modalities[0], pre_dice['Dice']))
            pre_CSFDice = dict(zip(modalities[0], pre_dice['CSF-Dice']))
            pre_GMDice = dict(zip(modalities[0], pre_dice['GM-Dice']))
            pre_WMDice = dict(zip(modalities[0], pre_dice['WM-Dice']))
            post_Dice = dict(zip(modalities[0], post_dice['Dice']))
            post_CSFDice = dict(zip(modalities[0], post_dice['CSF-Dice']))
            post_GMDice = dict(zip(modalities[0], post_dice['GM-Dice']))
            post_WMDice = dict(zip(modalities[0], post_dice['WM-Dice']))
            for m in modalities[0]:
                pre_metrics['Pre-reg %s foreground warping index' % m.upper()].append(pre_foreground_warping_indices[m])
                pre_metrics['Pre-reg %s overlap warping index' % m.upper()].append(pre_overlap_warping_indices[m])
                pre_metrics['Pre-reg %s Average Dice' % m.upper()].append(pre_Dice[m])
                pre_metrics['Pre-reg %s CSF Dice' % m.upper()].append(pre_CSFDice[m])
                pre_metrics['Pre-reg %s GM Dice' % m.upper()].append(pre_GMDice[m])
                pre_metrics['Pre-reg %s WM Dice' % m.upper()].append(pre_WMDice[m])
                post_metrics['Post-reg %s foreground warping index' % m.upper()].append(post_foreground_warping_indices[m])
                post_metrics['Post-reg %s overlap warping index' % m.upper()].append(post_overlap_warping_indices[m])
                post_metrics['Post-reg %s Average Dice' % m.upper()].append(post_Dice[m])
                post_metrics['Post-reg %s CSF Dice' % m.upper()].append(post_CSFDice[m])
                post_metrics['Post-reg %s GM Dice' % m.upper()].append(post_GMDice[m])
                post_metrics['Post-reg %s WM Dice' % m.upper()].append(post_WMDice[m])

            self.logger.info("[Train] Batch index: {:}, "
                             "Average pre-reg foreground warping index: {:.4f}, "
                             "Average pre-reg overlap warping index: {:.4f}, "
                             "Average post-reg foreground warping index: {:.4f}, "
                             "Average post-reg overlap warping index: {:.4f}, "
                             "Average pre-reg Dice: {:.4f}, "
                             "Average post-reg Dice: {:.4f}".format(idx,
                                                                    np.mean(pre_foreground_warping_index[-model.reg.group_num:]),
                                                                    np.mean(pre_overlap_warping_index[-model.reg.group_num:]),
                                                                    np.mean(post_foreground_warping_index[-model.reg.group_num:]),
                                                                    np.mean(post_overlap_warping_index[-model.reg.group_num:]),
                                                                    np.mean(list(pre_Dice.values())[-model.reg.group_num:]),
                                                                    np.mean(list(post_Dice.values())[-model.reg.group_num:])
                                                                    )
                             )

            self.store_predictions(model, pair_names, affines, headers)

        return indices, pre_metrics, post_metrics

    def _output_minibatch_stats(self, loss, model, idx, step, modalities):
        _, post_foreground_warping_index = model.evaluateForegroundWarpingIndex()
        foreground_warping_indices = dict(zip(modalities[0], post_foreground_warping_index))
        _, post_overlap_warping_index = model.evaluateOverlapWarpingIndex()
        overlap_warping_indices = dict(zip(modalities[0], post_overlap_warping_index))
        _, post_dice = model.evaluateOverlap()
        post_Dice = dict(zip(modalities[0], post_dice['Dice']))
        pred_flows = model.reg.predict_flows()
        num_neg_jacobs = np.sum([JacobianDeterminant(dimension=2)(f).lt(0).to(torch.float32).sum().detach().cpu().numpy()
                                 for f in pred_flows if f is not None])

        self.logger.info("[Train] Batch index: {:}, Step: {:}, Loss: {:.4f}, "
                         "Average foreground warping index: {:.4f}, "
                         "Average overlap warping index: {:.4f}, #NJD: {:.3e}, "
                         "Average post-reg Dice: {:.4f}".format(idx, step, loss,
                                                                np.mean(post_foreground_warping_index[-model.reg.group_num:]),
                                                                np.mean(post_overlap_warping_index[-model.reg.group_num:]),
                                                                num_neg_jacobs,
                                                                np.mean(list(post_Dice.values())[-model.reg.group_num:])
                                                                )
                         )
        return foreground_warping_indices, overlap_warping_indices, num_neg_jacobs, post_Dice

    def store_predictions(self, model, pair_names, affines, headers):
        with torch.no_grad():
            for i in range(1 if model.reg.group2ref else 0, model.num_subjects):
                img_name = os.path.basename(pair_names[i][0])[:-7]
                ffd_suffix = os.path.basename(pair_names[i][1])[:-7].split('_')[-1]
                for j in range(model.reg.num_reg_levels):
                    if model.reg.reg_level_type[j] == 'AFF':
                        np.save(file=os.path.join(self.save_path, img_name[:-7] + '_%s_' % ffd_suffix + '_AFF%s' % j),
                                arr=model.reg.params[j][0, i - 1 if model.reg.group2ref else i].cpu().numpy(),
                                )
                    else:
                        utils.save_prediction_nii(model.reg.params[model.reg.reg_level_type[j]][0, i - 1 if model.reg.group2ref
                                                                      else i].unsqueeze(1).permute(1, 2, 3, 0).cpu().numpy(),
                                                  data_type='vector_fields',
                                                  save_name=img_name + '_%s_%s%s.nii.gz' % (ffd_suffix, model.reg.reg_level_type[j], j),
                                                  save_path=self.save_path, save_dtype=np.float32,
                                                  affine=affines[0][i], header=headers[0][i])

    @staticmethod
    def reorder_posterior(posterior):
        d = posterior.dim()
        idx = torch.sort(torch.sum(posterior, dim=tuple(range(2, d))).squeeze(), descending=True)[1]
        return posterior[:, idx]


if __name__ == '__main__':
    os.chdir('')

    args = config()

    from datetime import datetime

    t = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = args.save_path
    if save_path is None:
        save_path = t + "_BrainWeb_%s_GroupRegTrainer" % args.model_type
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, 'config.txt'), 'w+') as f:
        f.write(str(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = 'cpu' if args.gpu == -1 else 'cuda:%s' % args.gpu

    logger = utils.config_logging(os.path.join(save_path, 'trainer.log'))
    logger.info("Working directory changed to: %s" % os.path.abspath(os.getcwd()))
    logger.info("Save path at: %s" % save_path)

    train_provider = ImageDataProvider(dimension=2, data_search_path=args.test_data_search_path, a_min=args.a_min,
                                       a_max=args.a_max, training=False, level_specific=True, data_type='Reg',
                                       rand_specific=True, ffd_levels=args.init_ffd_levels,
                                       ffd_samples=args.ffd_samples, modalities=args.modalities)

    net = BrainWebRegModel(dimension=2,
                           img_size=(217, 181),
                           model_type=args.model_type,
                           num_classes=args.num_classes,
                           num_bins=args.num_bins,
                           modalities=args.modalities,
                           sample_rate=args.sample_rate,
                           kernel_sigma=args.kernel_sigma,
                           mask_sigma=args.mask_sigma,
                           prior_sigma=args.prior_sigma,
                           alpha=args.alpha,
                           transform_type=args.transform_type,
                           pred_ffd_spacing=args.ffd_spacing,
                           pred_ffd_iso=args.ffd_iso,
                           group2ref=args.group2ref,
                           zero_avg_flow=args.zero_avg_flow,
                           inv_warp_ref=args.inv_warp_ref,
                           label_noise_mode=args.label_noise_mode,
                           label_noise_param=args.label_noise_param,
                           norm_img=True,
                           eps=args.eps)

    trainer = IterGroupRegTrainer(net, verbose=0, save_path=save_path,
                                  optimizer_name=args.optimizer,
                                  learning_rate=args.learning_rate,
                                  weight_decay=args.weight_decay,
                                  scheduler_name=args.scheduler,
                                  base_lr=args.base_learning_rate,
                                  max_lr=args.max_learning_rate,
                                  logger=logger)

    phantom = image_utils.load_image_nii(os.path.join(args.test_data_search_path,
                                                      'redefined_phantom_1.0mm_normal_crisp.nii.gz'))[0]
    phantom = phantom[phantom.shape[0] // 2]
    phantom = image_utils.get_one_hot_label(phantom, label_intensities=(0, 1, 2, 3), channel_first=True)
    phantom = torch.from_numpy(phantom).unsqueeze(0)

    indices, pre_metrics, post_metrics = trainer.train(train_provider, phantom=phantom, device=device,
                                                       steps=args.steps, display_step=args.display_step,
                                                       num_workers=args.num_workers)

    df1 = pd.DataFrame(data=dict([(k, v) for k, v in pre_metrics.items()
                                  if k.split(' ')[1].lower() in args.modalities]),
                       index=indices)
    df2 = pd.DataFrame(data=dict([(k, v) for k, v in post_metrics.items()
                                  if k.split(' ')[1].lower() in args.modalities]),
                       index=indices)

    with pd.ExcelWriter(os.path.join(save_path, 'results_%s.xlsx' % t)) as writer:
        df1.to_excel(writer, sheet_name='pre_reg_metrics')
        df2.to_excel(writer, sheet_name='post_reg_metrics')

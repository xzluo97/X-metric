# -*- coding: utf-8 -*-
"""
Train deep combined-computing model on MSCMR dataset.

__author__ = Xinzhe Luo
__version__ = 0.1

"""
import shutil
import sys

import pandas as pd

sys.path.append('../..')
import os
import torch
import random
from core import utils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from core.models.MSCMR.MSCMRDCCModel import MSCMRDCCModel
from core.data.data_providers.MSCMR.MSCMRImageProvider import ImageDataProvider
from core.trainers.genericTrainer import Trainer
from core.configs.ConfigMSCMR import config
from core.register.LocalDisplacementEnergy import JacobianDeterminant
from core.data import image_utils
from core.metrics import get_segmentation


jet_cm = plt.get_cmap('jet')
gray_cm = plt.get_cmap('gray')


class MSCMRDCCTrainer(Trainer):
    def train(self, train_dataset, valid_dataset=None, epochs=(100, ), display_step=1, device='cuda:0', **kwargs):
        batch_size = kwargs.pop('batch_size', 1)
        num_workers = kwargs.pop('num_workers', 0)
        num_samples = kwargs.pop("num_samples", None)
        data_collate_fn = kwargs.pop('data_collate_fn', train_dataset.data_collate_fn)
        ckpt = kwargs.pop('ckpt', None)
        test_dataset = kwargs.pop('test_dataset', None)
        inter_steps = kwargs.pop('inter_steps', 1)
        visualize_all = kwargs.pop('visualize_all', False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=data_collate_fn, pin_memory=True)
        training_iters = len(train_loader)

        validation_step = kwargs.pop("validation_step", None)
        if validation_step is None:
            validation_step = training_iters

        if num_samples is None:
            sample_indices = torch.arange(len(valid_dataset))
        else:
            sample_indices = torch.randperm(len(valid_dataset))[:num_samples]

        self.logger.info("------ Number of training iterations per epoch %s ------" % training_iters)

        self.writer = self._get_writer(self.save_path)

        model = self.net.to(device)

        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))

        opts = [self._get_optimizer([{'params': model.net.encoder.parameters()},
                                     {'params': model.net.seg_decoder.parameters()},
                                     {'params': model.net.seg_output_conv.parameters()}], lr=self.lr[0]),
                self._get_optimizer([{'params': model.net.reg_decoder.parameters()},
                                     {'params': model.net.reg_trans.parameters()},
                                     {'params': model.net.reg_output_convs.parameters()}], lr=self.lr[1])]
        train_metrics = {"Loss": {}}
        train_metrics.update(dict([("Post-reg %s Dice" % m.upper(), {}) for m in model.modalities]))

        if valid_dataset is not None:
            self.store_predictions(model, -1, 0, valid_dataset, device, sample_indices=sample_indices,
                                   batch_size=batch_size, num_workers=num_workers)
        if test_dataset is not None:
            _ = self.store_predictions(model, -1, 0, test_dataset, device,
                                       batch_size=batch_size, num_workers=num_workers,  mode='test',
                                       visualize_all=visualize_all)

        criterion = self.net.loss_function

        max_reg_dice = float('-inf')
        for epoch in range(epochs[0]):
            model.train()
            running_loss = 0

            for idx, data in enumerate(train_loader):
                step = epoch * training_iters + idx

                j = (step // inter_steps) % 2
                opt = opts[j]

                opt.zero_grad(set_to_none=True)

                images = data['images'].to(device)
                ffds = data['ffds']
                if isinstance(ffds, torch.Tensor):
                    ffds = ffds.to(device)
                probs = data['probs']
                if isinstance(probs, torch.Tensor):
                    probs = probs.to(device)
                atlas_prob = data['atlas_prob']
                if isinstance(atlas_prob, torch.Tensor):
                    atlas_prob = atlas_prob.to(device)

                _ = model(images, init_flows=ffds)
                loss = criterion(probs=probs, atlas_prob=atlas_prob)

                loss.backward()

                opt.step()

                running_loss += loss.item()

                if step % display_step == (display_step - 1):
                    loss = loss.item()
                    train_metrics['Loss'][step] = loss
                    self.writer.add_scalar("Loss/train", loss, global_step=step)
                    labels = data['labels'].to(device)
                    train_metrics = self.output_minibatch_stats(model, epoch, step, train_metrics, labels, atlas_prob)

                    with torch.no_grad():
                        init_labels = model.transform_images(labels, ffds, interp_mode='nearest')[1]
                        warped_labels = model.transform_images(init_labels, model.flows, interp_mode='nearest')[1]
                        posterior_seg = get_segmentation(model.pred_posterior)
                        warped_grids = model.transform_grids()

                    for n in range(1):
                        init_images_pair = [img[[n]] for img in model.images]
                        warped_images_pair = [img[[n]] for img in model.warped_images]

                        init_labels_pair = [lab[[n]] for lab in init_labels]
                        warped_labels_pair = [lab[[n]] for lab in warped_labels]

                        warped_grids_pair = [grid[[n]] for grid in warped_grids]

                        warped_images_tensor = \
                            self.visualize_warped_images(init_images_pair, warped_images_pair,
                                                         init_labels_pair, warped_labels_pair,
                                                         warped_grids_pair, model.flows[[n]])
                        self.writer.add_images('Train/WarpedImages/BatchIdx%s' % n,
                                               warped_images_tensor, global_step=step, dataformats='NHWC')
                        self.writer.add_images('Train/Posterior/BatchIdx%s' % n,
                                               np.stack([utils.rgba2rgb(jet_cm(p)) for p in
                                                         model.posterior[[n]].squeeze().detach().cpu().numpy()]),
                                               global_step=step, dataformats='NHWC')

                        seg_images_tensor = \
                            self.visualize_posterior_seg(warped_images=warped_images_pair,
                                                         posterior_seg=posterior_seg[[n]],
                                                         warped_grids=warped_grids_pair)
                        self.writer.add_images('Train/Segmentation/BatchIdx%s' % n,
                                               seg_images_tensor, global_step=step, dataformats='NHWC')

                    self.writer.flush()
                    model.train()

                if step % validation_step == (validation_step - 1) and valid_dataset is not None:
                    post_reg_dice = self.store_predictions(model, epoch, step, valid_dataset, device,
                                                           sample_indices=sample_indices, batch_size=batch_size,
                                                           num_workers=num_workers)
                    torch.save(model.state_dict(), os.path.join(self.save_path, 'model.pt'))
                    if post_reg_dice >= max_reg_dice:
                        torch.save(model.state_dict(), os.path.join(self.save_path, 'max_reg_dice_model.pt'))
                        max_reg_dice = post_reg_dice

                        if test_dataset is not None:
                            _ = self.store_predictions(model, epoch, step, test_dataset, device,
                                                       batch_size=batch_size, num_workers=num_workers,
                                                       mode='test', visualize_all=visualize_all)

                    model.train()

            self.logger.info("[Train] Epoch: {:}, Average Loss: {:.4f}, "
                             "Learning Rate: {:.1e}".format(epoch, running_loss / training_iters,
                                                            self.lr[j]
                                                            ))

        self.writer.close()

        return train_metrics

    def output_minibatch_stats(self, model, epoch, step, train_metrics, labels, atlas_prob):

        pre_reg_dice, post_reg_dice = model.evaluateRegDice(labels)
        seg_dice = model.evaluateSegDice(labels, atlas_prob=atlas_prob)

        for i in range(model.num_subjects):
            m = model.modalities[i]
            train_metrics["Post-reg %s Dice" % m.upper()][step] = post_reg_dice['Dice'][i]

        num_neg_jacobs = JacobianDeterminant(dimension=2)(model.flows).lt(0).to(torch.float32).mean(
            dim=0).sum().detach().cpu().numpy()

        self.writer.add_scalars('Train/PreReg-Dice',
                                dict(zip(model.modalities, pre_reg_dice['Dice'])),
                                global_step=step)
        self.writer.add_scalars('Train/PostReg-Dice',
                                dict(zip(model.modalities, post_reg_dice['Dice'])),
                                global_step=step)
        self.writer.add_scalars('Train/Seg-Dice',
                                dict(zip(model.modalities, seg_dice['Dice'])),
                                global_step=step)
        self.writer.add_scalar('Train/#NJD', num_neg_jacobs, global_step=step)

        self.logger.info("[TRAIN] Epoch: {:}, Step: {:}, Loss: {:.4f}, "
                         "#NJD: {:.3e}, "
                         "Average Pre-reg Dice: {:.4f}, "
                         "Average Post-reg Dice: {:.4f}, "
                         "Average Seg Dice: {:.4f}".format(epoch, step,
                                                           train_metrics['Loss'][step],
                                                           num_neg_jacobs,
                                                           np.mean(pre_reg_dice['Dice']),
                                                           np.mean(post_reg_dice['Dice']),
                                                           np.mean(seg_dice['Dice']))
                         )

        return train_metrics

    def store_predictions(self, model, epoch, step, dataset, current_device, **kwargs):
        model = model.eval()
        model = model.to(current_device)
        data_collate_fn = kwargs.pop('data_collate_fn', dataset.data_collate_fn)
        batch_size = kwargs.pop('batch_size', 1)
        num_workers = kwargs.pop('num_workers', 0)
        sample_indices = kwargs.pop('sample_indices', torch.arange(len(dataset)))
        mode = kwargs.pop('mode', 'valid')
        save_test_metrics = kwargs.pop('save_test_metrics', True)
        visualize_all = kwargs.pop('visualize_all', False)

        data_loader = DataLoader(Subset(dataset, indices=sample_indices), batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=data_collate_fn, drop_last=False)

        avg_pre_reg_dice = []
        avg_pre_reg_LV_dice = []
        avg_pre_reg_Myo_dice = []
        avg_pre_reg_RV_dice = []

        avg_post_reg_dice = []
        avg_post_reg_LV_dice = []
        avg_post_reg_Myo_dice = []
        avg_post_reg_RV_dice = []

        avg_seg_dice = []
        avg_seg_LV_dice = []
        avg_seg_Myo_dice = []
        avg_seg_RV_dice = []

        num_neg_jacobs = []

        with torch.no_grad():
            for idx, data in enumerate(data_loader):
                images = data['images'].to(current_device)
                ffds = data['ffds']
                if isinstance(ffds, torch.Tensor):
                    ffds = ffds.to(device)
                labels = data['labels'].to(current_device)
                atlas_prob = data['atlas_prob']
                if isinstance(atlas_prob, torch.Tensor):
                    atlas_prob = atlas_prob.to(device)

                B = images.shape[0] if visualize_all else 1

                _ = model(images, init_flows=ffds)

                reg_dice = model.evaluateRegDice(labels, reduce_batch=False)
                avg_pre_reg_dice.append(reg_dice[0]['Dice'])
                avg_pre_reg_Myo_dice.append(reg_dice[0]['Myo-Dice'])
                avg_pre_reg_LV_dice.append(reg_dice[0]['LV-Dice'])
                avg_pre_reg_RV_dice.append(reg_dice[0]['RV-Dice'])
                avg_post_reg_dice.append(reg_dice[1]['Dice'])
                avg_post_reg_Myo_dice.append(reg_dice[1]['Myo-Dice'])
                avg_post_reg_LV_dice.append(reg_dice[1]['LV-Dice'])
                avg_post_reg_RV_dice.append(reg_dice[1]['RV-Dice'])

                seg_dice = model.evaluateSegDice(labels, reduce_batch=False, atlas_prob=atlas_prob)
                avg_seg_dice.append(seg_dice['Dice'])
                avg_seg_Myo_dice.append(seg_dice['Myo-Dice'])
                avg_seg_LV_dice.append(seg_dice['LV-Dice'])
                avg_seg_RV_dice.append(seg_dice['RV-Dice'])

                num_neg_jacobs.append(JacobianDeterminant(dimension=2)(model.flows).lt(0).to(
                    torch.float32).sum(dim=(2, 3)).detach().cpu().numpy())

                init_labels = model.transform_images(labels, ffds, interp_mode='nearest')[1]
                warped_labels = model.transform_images(init_labels, model.flows, interp_mode='nearest')[1]

                posterior_seg = get_segmentation(model.pred_posterior)

                warped_grids = model.transform_grids()

                for n in range(B):
                    img_idx = B * idx + n
                    pair_names = dataset.get_image_name(img_idx)
                    subject_name = '_'.join(os.path.basename(pair_names[0][0]).split('_')[:3])
                    ffd_name = '_'.join(os.path.basename(pair_names[1][0][:-7]).split('_')[1:])

                    init_images_pair = [img[[n]] for img in model.images]
                    warped_images_pair = [img[[n]] for img in model.warped_images]

                    init_labels_pair = [lab[[n]] for lab in init_labels]
                    warped_labels_pair = [lab[[n]] for lab in warped_labels]

                    warped_grids_pair = [grid[[n]] for grid in warped_grids]

                    warped_images_tensor = \
                        self.visualize_warped_images(init_images_pair, warped_images_pair,
                                                     init_labels_pair, warped_labels_pair, warped_grids_pair,
                                                     model.flows[[n]])
                    self.writer.add_images('%s/WarpedImages/%s' % (mode, "_".join([subject_name, ffd_name])),
                                           warped_images_tensor, global_step=step, dataformats='NHWC')

                    seg_images_tensor = \
                        self.visualize_posterior_seg(warped_images=warped_images_pair,
                                                     posterior_seg=posterior_seg[[n]],
                                                     warped_grids=warped_grids_pair)
                    self.writer.add_images('%s/Segmentation/%s' % (mode, "_".join([subject_name, ffd_name])),
                                           seg_images_tensor, global_step=step, dataformats='NHWC')

                    seg_probs_pair = model.seg_probs[[n]]
                    seg_probs_tensor = self.visualize_seg_probs(seg_probs_pair)
                    self.writer.add_images('%s/ProbabilityMaps/%s' % (mode, "_".join([subject_name, ffd_name])),
                                           seg_probs_tensor, global_step=step, dataformats='NHWC')

                self.writer.flush()

        pre_reg_dice = np.concatenate(avg_pre_reg_dice, axis=0)
        pre_reg_Myo_dice = np.concatenate(avg_pre_reg_Myo_dice, axis=0)
        pre_reg_LV_dice = np.concatenate(avg_pre_reg_LV_dice, axis=0)
        pre_reg_RV_dice = np.concatenate(avg_pre_reg_RV_dice, axis=0)

        post_reg_Myo_dice = np.concatenate(avg_post_reg_Myo_dice, axis=0)
        post_reg_LV_dice = np.concatenate(avg_post_reg_LV_dice, axis=0)
        post_reg_RV_dice = np.concatenate(avg_post_reg_RV_dice, axis=0)
        post_reg_dice = np.concatenate(avg_post_reg_dice, axis=0)

        num_neg_jacobs = np.concatenate(num_neg_jacobs, axis=0)

        seg_dice = np.concatenate(avg_seg_dice, axis=0)
        seg_Myo_dice = np.concatenate(avg_seg_Myo_dice, axis=0)
        seg_LV_dice = np.concatenate(avg_seg_LV_dice, axis=0)
        seg_RV_dice = np.concatenate(avg_seg_RV_dice, axis=0)

        self.writer.add_scalars('%s/PreReg-Dice' % mode,
                                dict(zip(model.modalities, pre_reg_dice.mean(0))), global_step=step)
        self.writer.add_scalars('%s/PostReg-Dice' % mode,
                                dict(zip(model.modalities, post_reg_dice.mean(0))), global_step=step)
        self.writer.add_scalars('%s/Seg-Dice' % mode,
                                dict(zip(model.modalities, seg_dice.mean(0))), global_step=step)
        self.writer.add_scalars('%s/#NJD' % mode,
                                dict(zip(model.modalities, num_neg_jacobs.mean(0))), global_step=step)
        self.writer.flush()

        self.logger.info("[{:}] Epoch: {:}, Step: {:}, "
                         "#NJD: {:.3e}, "
                         "Average Pre-reg Dice: {:.4f}, "
                         "Average Post-reg Dice: {:.4f}, "
                         "Average Seg Dice: {:.4f}".format(mode.upper(), epoch, step,
                                                           np.mean(num_neg_jacobs),
                                                           np.mean(pre_reg_dice), np.mean(post_reg_dice),
                                                           np.mean(seg_dice))
                         )

        metrics = {}
        metrics.update(dict(zip(['Pre-reg %s Avg Dice' % m for m in model.modalities],
                                [pre_reg_dice[:, i] for i in range(len(model.modalities))])))
        metrics.update(dict(zip(['Pre-reg %s Myo Dice' % m for m in model.modalities],
                                [pre_reg_Myo_dice[:, i] for i in range(len(model.modalities))])))
        metrics.update(dict(zip(['Pre-reg %s LV Dice' % m for m in model.modalities],
                                [pre_reg_LV_dice[:, i] for i in range(len(model.modalities))])))
        metrics.update(dict(zip(['Pre-reg %s RV Dice' % m for m in model.modalities],
                                [pre_reg_RV_dice[:, i] for i in range(len(model.modalities))])))

        metrics.update(dict(zip(['Post-reg %s Avg Dice' % m for m in model.modalities],
                                [post_reg_dice[:, i] for i in range(len(model.modalities))])))
        metrics.update(dict(zip(['Post-reg %s Myo Dice' % m for m in model.modalities],
                                [post_reg_Myo_dice[:, i] for i in range(len(model.modalities))])))
        metrics.update(dict(zip(['Post-reg %s LV Dice' % m for m in model.modalities],
                                [post_reg_LV_dice[:, i] for i in range(len(model.modalities))])))
        metrics.update(dict(zip(['Post-reg %s RV Dice' % m for m in model.modalities],
                                [post_reg_RV_dice[:, i] for i in range(len(model.modalities))])))

        metrics.update(dict(zip(['Seg %s Avg Dice' % m for m in model.modalities],
                                [seg_dice[:, i] for i in range(len(model.modalities))])))
        metrics.update(dict(zip(['Seg %s Myo Dice' % m for m in model.modalities],
                                [seg_Myo_dice[:, i] for i in range(len(model.modalities))])))
        metrics.update(dict(zip(['Seg %s LV Dice' % m for m in model.modalities],
                                [seg_LV_dice[:, i] for i in range(len(model.modalities))])))
        metrics.update(dict(zip(['Seg %s RV Dice' % m for m in model.modalities],
                                [seg_RV_dice[:, i] for i in range(len(model.modalities))])))

        metrics.update(dict(zip(['%s #NJD' % m for m in model.modalities],
                                [num_neg_jacobs[:, i] for i in range(len(model.modalities))])))

        if save_test_metrics and mode == 'test':
            df = pd.DataFrame(metrics)
            df.to_excel(os.path.join(self.save_path, 'test_metrics.xlsx'))

        return np.mean(post_reg_dice)

    @staticmethod
    def visualize_warped_images(init_images, warped_images, init_labels, warped_labels, warped_grids,
                                flows,
                                color_names=('silver', 'red', 'cyan'), grid_color='white'):
        C, W, H = init_labels[0].shape[1:]
        N = len(init_images)

        img_tensor = np.zeros([N, H * 3, W, 3])

        for i in range(N):
            contours = image_utils.find_contours_marching_squares(init_labels[i][:, 1:].detach().cpu().numpy()).squeeze(
                0)
            contours_rgb = np.zeros([*contours.shape, 3])
            for c in [0, 2, 1]:
                contours_rgb[c] = utils.colorize_binary(contours[c], color_names[c])

            img = utils.rgba2rgb(gray_cm(utils.normalize_gray_img(init_images[i].squeeze().detach().cpu().numpy())))
            for c in [0, 2, 1]:
                img = np.where(np.expand_dims(contours[c], -1), contours_rgb[c], img)

            img_tensor[i, :H] = np.rot90(img)

        for i in range(N):
            contours = image_utils.find_contours_marching_squares(warped_labels[i][:, 1:].detach().cpu().numpy()).squeeze(
                0)
            contours_rgb = np.zeros([*contours.shape, 3])
            for c in [0, 2, 1]:
                contours_rgb[c] = utils.colorize_binary(contours[c], color_names[c])

            img = utils.rgba2rgb(gray_cm(utils.normalize_gray_img(warped_images[i].squeeze().detach().cpu().numpy())))
            for c in [0, 2, 1]:
                img = np.where(np.expand_dims(contours[c], -1), contours_rgb[c], img)

            grid_img = utils.colorize_binary(warped_grids[i].squeeze().cpu().numpy(), grid_color)

            img_tensor[i, H:(2 * H)] = np.rot90(img * 0.7 + grid_img * 0.3)

        for i in range(N):
            img = utils.normalize_rgb_img(flows[0, i].permute(1, 2, 0).detach().cpu().numpy())
            img_tensor[i, 2 * H:, :, :2] = np.rot90(img * 255)
        return img_tensor.astype(np.uint8)

    @staticmethod
    def visualize_posterior_seg(warped_images, posterior_seg, warped_grids=None, color_names=('snow', 'magenta', 'lime')):
        C = posterior_seg.shape[1]
        M = len(warped_images)
        W, H = warped_images[0].shape[2:]

        contours = image_utils.find_contours_marching_squares(posterior_seg[:, 1:].detach().cpu().numpy()).squeeze(
            0)
        rgb_contours = [utils.colorize_binary(contours[c], color_names[c]) for c in range(C - 1)]

        img_tensor = np.zeros([M, H, W, 3])
        for i in range(M):
            img = utils.rgba2rgb(gray_cm(utils.normalize_gray_img(warped_images[i].squeeze().detach().cpu().numpy())))
            for c in [0, 2, 1]:
                img = np.where(np.expand_dims(contours[c], -1), rgb_contours[c], img)

            img_tensor[i] = np.rot90(img)

        if warped_grids is not None:
            for i in range(M):
                grid_img = utils.colorize_binary(warped_grids[i].squeeze().cpu().numpy(), color_name='white')
                img_tensor[i] = img_tensor[i] * 0.7 + np.rot90(grid_img * 0.3)

        return img_tensor.astype(np.uint8)

    @staticmethod
    def visualize_seg_probs(seg_probs, color_maps=('Blues', 'Greens', 'Reds')):
        N, C, W, H = seg_probs.shape[1:]

        img_tensor = np.zeros([N, H * C, W, 3])

        for i in range(N):
            cm = plt.get_cmap(color_maps[i])
            for k in range(C):
                img = utils.rgba2rgb(cm(seg_probs[0, i, k].detach().cpu().numpy()))
                img_tensor[i, (k * H):((k + 1) * H)] = np.rot90(img)

        return img_tensor.astype(np.uint8)


if __name__ == '__main__':
    os.chdir('')

    args = config()

    from datetime import datetime

    t = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = args.save_path
    if save_path is None:
        save_path = t + "_MSCMR_DCCTrainer"
    if args.delete_former_path:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
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

    train_provider = ImageDataProvider(dimension=2,
                                       data_search_path=args.train_data_search_path,
                                       a_min=args.a_min,
                                       a_max=args.a_max,
                                       image_suffix=args.image_suffix,
                                       label_suffix=args.label_suffix,
                                       label_intensities=args.label_intensities,
                                       modalities=args.modalities,
                                       rand_ffd=True,
                                       compute_prob=args.use_prob,
                                       use_atlas=args.use_atlas,
                                       rho=args.rho)

    valid_provider = ImageDataProvider(dimension=2,
                                       data_search_path=args.valid_data_search_path,
                                       a_min=args.a_min,
                                       a_max=args.a_max,
                                       image_suffix=args.image_suffix,
                                       label_suffix=args.label_suffix,
                                       label_intensities=args.label_intensities,
                                       modalities=args.modalities,
                                       rand_ffd=True,
                                       rand_specific=True,
                                       use_atlas=args.use_atlas)

    test_provider = ImageDataProvider(dimension=2,
                                      data_search_path=args.test_data_search_path,
                                      a_min=args.a_min,
                                      a_max=args.a_max,
                                      image_suffix=args.image_suffix,
                                      label_suffix=args.label_suffix,
                                      label_intensities=args.label_intensities,
                                      modalities=args.modalities,
                                      rand_ffd=True,
                                      rand_specific=True,
                                      use_atlas=args.use_atlas)

    net = MSCMRDCCModel(img_size=(160, 160),
                        num_classes=args.num_classes,
                        modalities=args.modalities,
                        init_features=args.init_features,
                        dropout=args.dropout,
                        norm_type=args.norm_type,
                        num_blocks=args.num_blocks,
                        num_bins=args.num_bins,
                        sample_rate=args.sample_rate,
                        kernel_sigma=args.kernel_sigma,
                        mask_radius=args.mask_radius,
                        alpha=args.alpha,
                        sup_mods=args.sup_mods,
                        clamp_prob=args.clamp_prob,
                        prob_interval=args.prob_interval,
                        use_atlas=args.use_atlas,
                        eps=args.eps)

    trainer = MSCMRDCCTrainer(net,
                              verbose=0, save_path=save_path,
                              optimizer_name=args.optimizer,
                              learning_rate=args.learning_rate,
                              weight_decay=args.weight_decay,
                              scheduler_name=None,
                              base_lr=args.base_learning_rate,
                              max_lr=args.max_learning_rate,
                              logger=logger)

    train_metrics = trainer.train(train_provider, valid_provider,
                                  device=device, epochs=args.epochs, display_step=args.display_step,
                                  batch_size=args.batch_size, num_workers=args.num_workers,
                                  validation_step=args.validation_step,
                                  num_samples=args.num_validation_samples,
                                  ckpt=args.checkpoint,
                                  test_dataset=test_provider,
                                  inter_steps=args.inter_steps,
                                  visualize_all=args.visualize_all)

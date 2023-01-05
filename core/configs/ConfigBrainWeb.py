# -*- coding: utf-8 -*-
"""
Configuration for Automatic nonparametric appearance model on Brainweb.

__author__ = 'Xinzhe Luo'
__version__ = 0.1
"""

import argparse

def config(name='AutoNonAppBrainweb'):
    parser = argparse.ArgumentParser(name)
    parser.add_argument('--gpu', '-g', default=0, type=int, choices=list(range(-1, 8)))
    parser.add_argument('--save_path', '-sp', default=None, type=str,
                        help='path where to save the model, if None use the default directory specified by the running script')
    parser.add_argument('--seed', '-s', default=2333, type=int, help='random seed generating random numbers')
    parser.add_argument('--modalities', '-m', default=('pd', 't1', 't2'), type=str,
                        choices=['t1', 't2', 'pd', 'avg'], nargs='+',
                        help='types of modality in groupwise registration')
    parser.add_argument('--a_min', default=None, type=float, help='min value for intensity clipping')
    parser.add_argument('--a_max', default=None, type=float, help='max value for intensity clipping')
    parser.add_argument('--level_specific', '-ls', default=False, action='store_true',
                        help='whether to perform groupwise registration on images with the same level of deformation')
    parser.add_argument('--data_type', '-dt', default='App', choices=['App', 'Reg'])
    parser.add_argument('--init_ffd_levels', '-ifl', default=None, nargs='+', type=str, help='initial FFD levels')
    parser.add_argument('--ffd_samples', '-fs', default=None, type=int, help='number of samples for each FFD level')
    parser.add_argument('--train_data_search_path', '-trdsp', type=str, default='')
    parser.add_argument('--valid_data_search_path', '-vdsp', type=str, default='')
    parser.add_argument('--test_data_search_path', '-tedsp', type=str, default='')

    parser.add_argument('--model_type', '-mt', default='XCoRegUn', choices=['XCoRegUn', 'XCoRegGT', 'GMM', 'APE', 'CTE'])
    parser.add_argument('--num_subjects', '-ns', default=3, type=int, help='number of subjects for groupwise registration')
    parser.add_argument('--num_classes', '-nc', default=4, type=int, help='number of tissue classes')
    parser.add_argument('--num_bins', '-nb', default=64, type=int, help='number of bins in the appearance model')
    parser.add_argument('--arch', default='resnet18', type=str, help='network architecture')
    parser.add_argument('--sample_rate', '-sr', default=0.1, type=float, help='sample rate for density estimation')
    parser.add_argument('--kernel_sigma', '-ks', default=1, type=float, help='std for the Gaussian kernel')
    parser.add_argument('--mask_sigma', '-ms', default=-1, type=int, help='std for producing the mask')
    parser.add_argument('--prior_sigma', '-ps', default=-1, type=int, help='std for producing the spatial prior')
    parser.add_argument('--noise_level', '-nl', default=0.03, type=float, help='noise level')
    parser.add_argument('--INU_level', '-il', default=0.2, type=float, help='INU level')
    parser.add_argument('--alpha', default=1, type=float, help='bending energy coefficient')
    parser.add_argument('--transform_type', '-tt', default=['DDF'], nargs='+', choices=['DDF', 'FFD'],
                        help='transformation types')
    parser.add_argument('--ffd_spacing', '-ffds', default=None, type=float, nargs='+', help='multi-level FFD spacings')
    parser.add_argument('--ffd_iso', '-ffdi', default=False, action='store_true',
                        help='whether to use isotropic FFD spacing')
    parser.add_argument('--group2ref', '-g2r', default=False, action='store_true',
                        help='whether to apply group-to-reference registration')
    parser.add_argument('--zero_avg_flow', '-zaf', default=False, action='store_true',
                        help='whether to constrain the average deformation to be zero')
    parser.add_argument('--inv_warp_ref', '-iwr', default=False, action='store_true',
                        help='whether to inverse warp the reference image')
    parser.add_argument('--label_noise_mode', '-lnm', default=0, type=int, help='label noise mode')
    parser.add_argument('--label_noise_param', '-lnp', default=0, type=float, help='parameters for label noise')
    parser.add_argument('--init_features', default=16, type=int, help='initial features for the network')
    parser.add_argument('--num_blocks', default=4, type=int, help='number of blocks for the network')
    parser.add_argument('--norm_type', default='batch', type=str, help='normalization type')
    parser.add_argument('--dropout', '-dp', default=0.2, type=float, help='dropout probability for network training')
    parser.add_argument('--eps', '-eps', default=1e-8, type=float)

    parser.add_argument('--optimizer', '-opt', default='Adam', help='optimizer name')
    parser.add_argument('--learning_rate', '-lr', default=(0.1, ), type=float, nargs='+', help='network learning rate')
    parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay coefficient')
    parser.add_argument('--scheduler', default=None, help='scheduler name', choices=['CyclicLR', 'OneCyclicLR'])
    parser.add_argument('--max_learning_rate', '-max_lr', default=1e-4, type=float, help='maximum learning rate')
    parser.add_argument('--base_learning_rate', '-base_lr', default=1e-5, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', '-bs', default=1, type=int, help='training batch size')
    parser.add_argument('--num_workers', '-nw', default=0, type=int,
                        help='how many sub-processes to use for data loading')
    parser.add_argument('--epochs', '-epochs', default=50, type=int, help='training epochs')
    parser.add_argument('--steps', '-steps', default=(50,), type=int, nargs='+',
                        help='steps for multi-level registration')
    parser.add_argument('--inter_steps', '-isp', default=1, type=int, help='interleaving steps')
    parser.add_argument('--display_step', '-ds', default=1, type=int, help='number of steps till outputting stats')
    parser.add_argument('--validation_step', '-vas', default=None, type=int,
                        help='number of steps till validation, default as number of training iterations in one epoch')
    parser.add_argument('--num_validation_samples', '-nvs', default=None, type=int, help='number of validation samples')
    parser.add_argument('--checkpoint', '-ckpt', default=None, type=str,
                        help='path where to load the pretrained model')
    parser.add_argument('--app_ckpt', '-ackpt', default=None, type=str,
                        help='path where to load the pretrained appearance model')
    parser.add_argument('--seg_ckpt', '-sckpt', default=None, type=str,
                        help='path where to load the pretrained segmentation model')
    parser.add_argument('--reg_ckpt', '-rckpt', default=None, type=str,
                        help='path where to load the pretrained registration model')
    parser.add_argument('--compute_diff', '-cd', default=False, action='store_true',
                        help='whether to compute difference of the metric before and after updates of common space')

    args = parser.parse_args()
    return args

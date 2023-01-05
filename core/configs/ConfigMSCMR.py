# -*- coding: utf-8 -*-
"""
Configuration for Deep Combined Computing on MSCMR.

__author__ = 'Xinzhe Luo'
__version__ = 0.1
"""

import argparse

def config(name='MSCMR'):
    parser = argparse.ArgumentParser(name)
    parser.add_argument('--gpu', '-g', default=0, type=int, choices=list(range(-1, 8)))
    parser.add_argument('--dtype', '-d', default='float32', choices=['float32', 'float64'])
    parser.add_argument('--save_path', '-sp', default=None, type=str,
                        help='path where to save the model, if None use the default directory specified by the running script')
    parser.add_argument('--seed', '-s', default=2333, type=int, help='random seed generating random numbers')
    parser.add_argument('--modalities', '-m', default=('DE', 'C0', 'T2'), type=str, choices=['DE', 'C0', 'T2'], nargs='+',
                        help='types of modality in groupwise registration')
    parser.add_argument('--atlas_modality', '-am', default='C0', type=str, choices=['DE', 'C0', 'T2'],
                        help='type of atlas modality')
    parser.add_argument('--ffd_levels', '-ffdl', default=(5, 10, 20, 40), type=int, nargs='+',
                        help='FFD spacings of the initial random deformations')
    parser.add_argument('--a_min', default=None, type=float, help='min value for intensity clipping')
    parser.add_argument('--a_max', default=None, type=float, help='max value for intensity clipping')
    parser.add_argument('--label_intensities', '-lis', default=(0, 200, 500, 600), nargs='+', type=int,
                        help='intensity values of the label')
    parser.add_argument('--img_size', default=(160, 160), type=int, nargs='+', help='size of the input image')
    parser.add_argument('--train_data_search_path', '-trdsp', type=str, default='')
    parser.add_argument('--valid_data_search_path', '-vdsp', type=str, default='')
    parser.add_argument('--test_data_search_path', '-tedsp', type=str, default='')
    parser.add_argument('--image_suffix', '-is', type=str, default='image.nii.gz',
                        help='suffix pattern for the loading images')
    parser.add_argument('--label_suffix', '-ls', type=str, default='label.nii.gz',
                        help='suffix pattern for the loading labels')
    parser.add_argument('--rho', '-rho', type=float, default=1,
                        help='parameter for distance transform probability maps')
    parser.add_argument('--clamp_prob', '-cp', default=False, action='store_true',
                        help='whether to clamp the probability map')
    parser.add_argument('--prob_interval', '-pi', default=(0.05, 0.85), nargs=2, type=float)

    parser.add_argument('--num_classes', '-nc', default=4, type=int, help='number of tissue classes')
    parser.add_argument('--num_subtypes', '-ns', default=(2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1), type=int, nargs='+', help='number of tissue subtypes')
    parser.add_argument('--num_bins', '-nb', default=64, type=int, help='number of bins in the appearance model')
    parser.add_argument('--init_features', default=16, type=int, help='initial features for the network')
    parser.add_argument('--num_blocks', default=4, type=int, help='number of blocks for the network')
    parser.add_argument('--norm_type', default='batch', type=str, help='normalization type')
    parser.add_argument('--dropout', '-dp', default=0.2, type=float,
                        help='dropout probability of the registration network')
    parser.add_argument('--sample_rate', '-sr', default=0.1, type=float, help='sample rate for density estimation')
    parser.add_argument('--kernel_sigma', '-ks', default=1, type=float, help='std for the Gaussian kernel')
    parser.add_argument('--mask_radius', '-mr', default=10, type=int, help='radius for producing the mask')
    parser.add_argument('--prior_sigma', '-ps', default=-1, type=int, help='std for producing the spatial prior')
    parser.add_argument('--alpha', default=1, type=float, help='bending energy coefficient')
    parser.add_argument('--sup_mods', default=None, type=str, nargs='+', help='the modalities with supervision')
    parser.add_argument('--use_prob', '-up', default=False, action='store_true', help='whether to use probability maps')
    parser.add_argument('--fix_commonspace', '-fc', default=False, action='store_true', help='whether to fix the commonspace')
    parser.add_argument('--use_atlas', '-ua', default=False, action='store_true', help='whether to use the atlas probability map')

    parser.add_argument('--model_type', '-mt', default='XCoRegUn',
                        choices=['XCoRegUn', 'XCoRegGT', 'CTE', 'SSD', 'AMI', 'CG', 'APE', 'MvMM'])
    parser.add_argument('--transform_type', '-tt', default=['AFF'], nargs='+',
                        choices=['DDF', 'FFD', 'AFF', 'SVF', 'RIG', 'TRA'],
                        help='transformation type')
    parser.add_argument('--ffd_spacing', '-ffds', default=None, type=float, nargs='+', help='multi-level FFD spacings')
    parser.add_argument('--ffd_iso', '-ffdi', default=False, action='store_true',
                        help='whether to use isotropic FFD spacing')
    parser.add_argument('--int_steps', '-its', default=7, type=int, help='integration steps for the velocity field')
    parser.add_argument('--group2ref', '-g2r', default=False, action='store_true',
                        help='whether to apply group-to-reference registration')
    parser.add_argument('--zero_avg_flow', '-zaf', default=False, action='store_true',
                        help='whether to constrain the average deformation to be zero')
    parser.add_argument('--zero_avg_vec', '-zav', default=False, action='store_true',
                        help='whether to constrain the average velocity to be zero')
    parser.add_argument('--zero_avg_disp', '-zad', default=False, action='store_true',
                        help='whether to constrain the average absolute displacement to be zero')
    parser.add_argument('--eps', '-eps', default=1e-10, type=float)
    parser.add_argument('--num_res', '-nr', default=[1, 1], nargs='+', type=int, help='number of resolutions')

    parser.add_argument('--optimizer', '-opt', default='Adam', help='optimizer name')
    parser.add_argument('--learning_rate', '-lr', default=(1e-3, 1e-4), type=float, nargs='+', help='network learning rate')
    parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay coefficient')
    parser.add_argument('--scheduler', default=None, help='scheduler name', choices=['CyclicLR', 'OneCyclicLR'])
    parser.add_argument('--max_learning_rate', '-max_lr', default=1e-4, type=float, help='maximum learning rate')
    parser.add_argument('--base_learning_rate', '-base_lr', default=1e-5, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', '-bs', default=1, type=int, help='training batch size')
    parser.add_argument('--num_workers', '-nw', default=4, type=int,
                        help='how many sub-processes to use for data loading')
    parser.add_argument('--epochs', '-epochs', default=(50, ), type=int, nargs='+', help='training epochs')
    parser.add_argument('--steps', '-steps', default=(50,), type=int, nargs='+',
                        help='steps for multi-level registration')
    parser.add_argument('--inter_steps', '-isp', default=5, type=int, help='interleaving steps')
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
    parser.add_argument('--delete_former_path', '-dfp', default=False, action='store_true',
                        help='whether to delete former save path')
    parser.add_argument('--visualize_all', '-va', default=False, action='store_true',
                        help='whether to visualize all test results')

    args = parser.parse_args()
    return args

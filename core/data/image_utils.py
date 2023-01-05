# -*- coding: utf-8 -*-
"""
Data utility functions for image loader and processing.

@author: Xinzhe Luo
@version: 0.1
"""

import re
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.transform import rescale
from skimage import measure
from scipy import stats
from scipy.ndimage import distance_transform_edt as distance
from sklearn import mixture
import matplotlib.pyplot as plt
from core.utils import gauss_kernel1d, separable_filter2d, get_normalized_prob


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    """
    return [atoi(c) for c in re.split('(\d+)', text)]


def strsort(alist):
    alist.sort(key=natural_keys)
    return alist


def load_image_nii(path, dtype=np.float32, scale=0, order=1):
    img = nib.load(path)
    image = np.asarray(img.get_fdata(), dtype)
    if scale > 0:
        image = rescale(image, 1 / (2 ** scale), mode='reflect',
                        multichannel=False, anti_aliasing=False, order=order)
    return image, img.affine, img.header


def save_image_nii(array, save_path, **kwargs):
    affine = kwargs.pop("affine", np.eye(4))
    header = kwargs.pop("header", None)
    save_dtype = kwargs.pop("save_dtype", np.int16)
    img = nib.Nifti1Image(np.asarray(array, dtype=save_dtype), affine=affine, header=header)
    nib.save(img, save_path)


def save_label_nii(array, label_intensities, save_path, **kwargs):
    label = np.sum(array * np.asarray(label_intensities).reshape(array.shape[0], *[1] * (array.ndim - 1)), axis=0)
    save_image_nii(label, save_path, **kwargs)


def load_image_png(path, dtype=np.float32, scale=0, order=1):
    img = Image.open(path)
    image = np.asarray(img, dtype)
    if scale > 0:
        image = rescale(image, 1 / (2 ** scale), mode='reflect',
                        multichannel=False, anti_aliasing=False, order=order)
    return image


def save_image_png(array, save_path, **kwargs):
    normalize = kwargs.pop('normalize', True)
    if normalize:
        array = normalize_image(array, 'min-max') * 255
    img = Image.fromarray(np.asarray(array, dtype=np.uint8))
    img.save(save_path)


def load_prob_file(path_list, dtype=np.float32, max_value=1000, scale=0):
    return np.asarray(np.stack([load_image_nii(name, order=0, scale=scale)[0]
                                for name in path_list], -1) / max_value,
                      dtype=dtype)


def normalize_image(image, normalization=None, **kwargs):
    image = np.copy(image)
    if normalization == 'min-max':
        image -= np.min(image)
        image /= np.max(image)

    elif normalization == 'z-score':
        image = stats.zscore(image, axis=None, ddof=1)
        if kwargs.pop('clip_value', None):
            image = np.clip(image, -3, 3)

    elif normalization == 'interval':
        image -= np.min(image)
        image /= np.max(image)
        a = kwargs.pop('a', -1)
        b = kwargs.pop('b', 1)
        image = (b-a) * image + a

    return image


def get_one_hot_label(gt, label_intensities=None, channel_first=False):
    if label_intensities is None:
        label_intensities = sorted(np.unique(gt))
    num_classes = len(label_intensities)
    label = np.around(gt)
    if channel_first:
        label = np.zeros((np.hstack((num_classes, label.shape))), dtype=np.float32)

        for k in range(num_classes):
            label[k] = (gt == label_intensities[k])

        label[0] = np.logical_not(np.sum(label[1:,], axis=0))
    else:
        label = np.zeros((np.hstack((label.shape, num_classes))), dtype=np.float32)

        for k in range(num_classes):
            label[..., k] = (gt == label_intensities[k])

        label[..., 0] = np.logical_not(np.sum(label[..., 1:], axis=-1))

    return label


def get_distance_prob(one_hot_label, clip_value=50, rho=0.1):
    C = one_hot_label.shape[0]
    dist_map = []
    for c in range(C):
        mask = one_hot_label[c].astype(np.bool_)
        pos_dist = distance(mask)
        neg_dist = distance(~mask)

        dist = pos_dist - neg_dist
        dist_map.append(dist)

    dist = np.stack(dist_map)
    prob = get_normalized_prob(np.exp(np.clip(rho * dist, - clip_value, clip_value)), mode='np', dim=0)

    return prob.astype(np.float32)


def visualize_image2d(image, **kwargs):
    plt.imshow(image, **kwargs)
    plt.show()


def extract_contour(label, combine=False):
    c = label[:, :, 1:-1, 1:-1]
    c0 = label[:, :, :-2, :-2]
    c1 = label[:, :, 1:-1, :-2]
    c2 = label[:, :, 2:, :-2]
    c3 = label[:, :, :-2, 1:-1]
    c4 = label[:, :, 2:, 1:-1]
    c5 = label[:, :, :-2, 2:]
    c6 = label[:, :, 1:-1, 2:]
    c7 = label[:, :, 2:, 2:]
    contour = (c - c0 != 0) | (c - c1 != 0) | (c - c2 != 0) | (c - c3 != 0) | \
              (c - c4 != 0) | (c - c5 != 0) | (c - c6 != 0) | (c - c7 != 0)
    if combine:
        contour = torch.any(contour, dim=1, keepdim=True)
    contour = F.pad(contour, pad=(1, 1, 1, 1))
    return contour


def find_contours_marching_squares(label, **kwargs):
    contours = np.zeros_like(label)
    B, C = label.shape[:2]
    for i in range(B):
        for k in range(C):
            cs = measure.find_contours(label[i, k], level=0.5, **kwargs)
            for c in cs:
                contours[i, k, c[:, 0].astype(np.int), c[:, 1].astype(np.int)] = 1
    return contours


def random_permute_contour(label, contour, rate=0.01):
    B = label.shape[0]
    C = label.shape[1]
    N = np.prod(label.shape[2:])
    num_perm = int(rate * N)
    contour_combined = torch.any(contour, dim=1, keepdim=True)
    num_contour = torch.sum(contour_combined, dim=(1, 2, 3))
    new_label = []
    for b in range(B):
        perm_label = label[b].clone()
        perm_idx = torch.randperm(num_contour[b])[:num_perm]
        perm_coord = torch.nonzero(contour_combined[b])[perm_idx]  # [num_perm, 3]
        for c in perm_coord:
            p = torch.randperm(C)
            perm_label[:, c[1], c[2]] = perm_label[:, c[1], c[2]][p]

        new_label.append(torch.where(contour[b], perm_label, label[b]))

    return torch.stack(new_label)


# -*- coding: utf-8 -*-
"""
Utility functions and operations.

@author: Xinzhe Luo
"""
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import re
import os
import logging
from PIL import Image
import nibabel as nib
from matplotlib import colors
import math
import torch
from scipy import signal
from torch.nn import functional as F


def normalize_gray_img(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().clone()
        img -= img.amin()
        img /= img.amax()
    elif isinstance(img, np.ndarray):
        img -= np.min(img)
        img /= np.max(img)
    else:
        raise NotImplementedError
    return img


def normalize_rgb_img(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().clone()
        img -= img.amin(dim=(0, 1), keepdim=True)
        img /= img.amax(dim=(0, 1), keepdim=True)
    elif isinstance(img, np.ndarray):
        img -= np.min(img, axis=(0, 1), keepdims=True)
        img /= np.max(img, axis=(0, 1), keepdims=True)
    else:
        raise NotImplementedError
    return img


def colorize_binary(img, color_name):
    img_rgb = np.zeros([*img.shape, 3])
    rgb = colors.to_rgb(color_name)
    for c in range(3):
        img_rgb[..., c] = img * rgb[c]

    return to_rgb(img_rgb)


def to_rgb(img):
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=-1)
    channels = img.shape[-1]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    for k in range(np.shape(img)[2]):
        st = img[..., k]
        if np.amin(st) != np.amax(st):
            st -= np.amin(st)
            st /= np.amax(st)
        st *= 255
    return img.round().astype(np.uint8)


def rgba2rgb(rgba, background=(1, 1, 1)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray(a, dtype='float32')

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray(rgb * 255, dtype=np.uint8)


def save_image(imgs, path):
    for i in range(len(imgs)):
        img_path = os.path.join(os.path.split(path)[0], 'class%d_' % i + os.path.split(path)[1])
        Image.fromarray(imgs[i].round().astype(np.uint8)).save(img_path, 'PNG', dpi=[300, 300], quality=95)


def save_prediction_nii(pred, save_name, data_type='image', data_provider=None, **kwargs):
    save_path = kwargs.pop("save_path", os.path.dirname(save_name))
    save_name = os.path.basename(save_name)
    affine = kwargs.pop("affine", np.eye(4))
    header = kwargs.pop("header", None)
    save_prefix = kwargs.pop("save_prefix", '')
    if data_provider:
        original_size = kwargs.pop("original_size", data_provider.original_size)
        image_suffix = kwargs.pop('image_suffix', data_provider.image_suffix)
    else:
        original_size = kwargs.pop("original_size", pred.shape)
        image_suffix = kwargs.pop('image_suffix', 'image.nii.gz')

    if len(original_size) == 2:
        original_size = (*original_size, 1)

    if save_path is not None:
        abs_pred_path = os.path.abspath(save_path)
        if not os.path.exists(abs_pred_path):
            logging.info("Allocating '{:}'".format(abs_pred_path))
            os.makedirs(abs_pred_path)

    if data_type == 'image':
        save_suffix = kwargs.pop("save_suffix", 'image.nii.gz')
        save_dtype = kwargs.pop("save_dtype", np.uint16)
        squeeze_channel = kwargs.pop("squeeze_channel", False)
        pred_pad = np.pad(pred,
                          (((original_size[0] - pred.shape[0]) // 2,
                            original_size[0] - pred.shape[0] - (original_size[0] - pred.shape[0]) // 2),
                           ((original_size[1] - pred.shape[1]) // 2,
                            original_size[1] - pred.shape[1] - (original_size[1] - pred.shape[1]) // 2),
                           ((original_size[2] - pred.shape[2]) // 2,
                          original_size[2] - pred.shape[2] - (original_size[2] - pred.shape[2]) // 2),
                           (0, 0)), 'constant')
        if squeeze_channel:
            pred_pad = np.mean(pred_pad, axis=-1)
        img = nib.Nifti1Image(pred_pad.astype(save_dtype), affine=affine, header=header)

    elif data_type == 'vector_fields':
        save_suffix = kwargs.pop("save_suffix", 'vector.nii.gz')
        save_dtype = kwargs.pop("save_dtype", np.float32)
        if pred.shape[-1] <= 2:
            zero_fields = np.zeros([pred.shape[0], pred.shape[1], pred.shape[2], 3 - pred.shape[-1]])
            pred = np.concatenate([pred, zero_fields], axis=-1)
        pred_pad = np.pad(pred,
                          (((original_size[0] - pred.shape[0]) // 2,
                            original_size[0] - pred.shape[0] - (original_size[0] - pred.shape[0]) // 2),
                           ((original_size[1] - pred.shape[1]) // 2,
                            original_size[1] - pred.shape[1] - (original_size[1] - pred.shape[1]) // 2),
                           ((original_size[2] - pred.shape[2]) // 2,
                            original_size[2] - pred.shape[2] - (original_size[2] - pred.shape[2]) // 2),
                           (0, 0)), 'constant')
        img = nib.Nifti1Image(pred_pad.astype(save_dtype), affine=affine, header=header)

    elif data_type == 'label':
        save_suffix = kwargs.pop("save_suffix", 'seg.nii.gz')
        save_dtype = kwargs.pop("save_dtype", np.uint16)
        if data_provider:
            label_intensity = kwargs.pop("label_intensity", data_provider.label_intensity)
        else:
            label_intensity = kwargs['label_intensity']
        class_preds = []
        for i in range(pred.shape[-1]):
            if i == 0:
                class_preds.append(np.pad(pred[..., i],
                                          (((original_size[0] - pred.shape[0]) // 2,
                                            original_size[0] - pred.shape[0] - (original_size[0] - pred.shape[0]) // 2),
                                           ((original_size[1] - pred.shape[1]) // 2,
                                            original_size[1] - pred.shape[1] - (original_size[1] - pred.shape[1]) // 2),
                                           ((original_size[2] - pred.shape[2]) // 2,
                                            original_size[2] - pred.shape[2] - (original_size[2] - pred.shape[2]) // 2)
                                           ), 'constant', constant_values=1))
            else:
                class_preds.append(np.pad(pred[..., i],
                                          (((original_size[0] - pred.shape[0]) // 2,
                                            original_size[0] - pred.shape[0] - (original_size[0] - pred.shape[0]) // 2),
                                           ((original_size[1] - pred.shape[1]) // 2,
                                            original_size[1] - pred.shape[1] - (original_size[1] - pred.shape[1]) // 2),
                                           ((original_size[2] - pred.shape[2]) // 2,
                                            original_size[2] - pred.shape[2] - (original_size[2] - pred.shape[2]) // 2)
                                           ), 'constant'))

        pred = np.stack(class_preds, -1)
        intensity = np.tile(np.asarray(label_intensity), np.concatenate((pred.shape[:-1], [1])))
        mask = np.eye(pred.shape[-1])[np.argmax(pred, axis=-1)]
        img = nib.Nifti1Image(np.sum(mask * intensity, axis=-1).astype(save_dtype), affine=affine, header=header)
    else:
        raise NotImplementedError

    if image_suffix is None:
        save_name = save_prefix + save_name + save_suffix
    else:
        save_name = save_prefix + save_name.replace(image_suffix, save_suffix)
    nib.save(img, os.path.join(save_path, save_name))


def gaussian_pdf(x, mu, sigma, mode='torch', eps=1e-7):
    if mode == 'torch':
        pi = torch.as_tensor(math.pi, dtype=x.dtype, device=x.device)
        return torch.clamp_min(1 / (torch.sqrt(2*pi)*sigma+eps) * torch.exp(-(x-mu)**2 / (2*sigma**2+eps)), min=eps)
    elif mode == 'np':
        return np.clip(1 / (np.sqrt(2*math.pi)*sigma+eps) * np.exp(-(x-mu)**2 / (2*sigma**2+eps)), a_min=eps)
    else:
        raise NotImplementedError


def compute_normalized_prob(prob, dim=1, mode='torch', eps=1e-8):
    if mode == 'torch':
        prob = torch.clamp(prob, min=eps)
        return prob / torch.sum(prob, dim=dim, keepdim=True)
    elif mode == 'np':
        prob = np.clip(prob, a_min=eps, a_max=None)
        return prob / np.sum(prob, axis=dim, keepdims=True)
    else:
        raise NotImplementedError


def get_normalized_prob(prob, mode='torch', dim=1, **kwargs):
    eps = kwargs.pop('eps', 1e-8)
    if mode == 'torch':
        return prob / torch.sum(prob, dim=dim, keepdim=True).clamp(min=eps)
    elif mode == 'np':
        return prob / np.sum(prob, axis=dim, keepdims=True).clip(min=eps)
    else:
        raise NotImplementedError


def gauss_kernel1d(sigma):
    assert sigma >= 0
    if sigma == 0:
        return 1
    else:
        tail = int(sigma*3)
        k = np.exp([-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)])
        return k / np.sum(k)


def ones_kernel1d(r):
    assert r >= 0
    if r == 0:
        return 1
    else:
        s = int(r)
        return np.ones([s])


def separable_filter3d(vol, kernel, mode='torch'):
    if np.all(kernel == 0):
        return vol
    if mode == 'torch':
        kernel = torch.as_tensor(kernel, dtype=vol.dtype, device=vol.device)
        if vol.ndim == 3:
            vol = vol.unsqueeze(0).unsqueeze(0)
        channels = vol.size(1)
        kernel = kernel.repeat(channels, 1, 1, 1, 1)
        padding = kernel.size(-1) // 2
        return F.conv3d(
            F.conv3d(F.conv3d(vol, kernel.view(channels, 1, -1, 1, 1), padding=(padding, 0, 0), groups=channels),
                     kernel.view(channels, 1, 1, -1, 1), padding=(0, padding, 0), groups=channels),
            kernel.view(channels, 1, 1, 1, -1), padding=(0, 0, padding), groups=channels)

    elif mode == 'np':
        if vol.ndim == 2:
            vol = np.expand_dims(vol, axis=(0, 1))
        return signal.convolve(signal.convolve(signal.convolve(vol,
                                                               np.reshape(kernel, [1, 1, -1, 1, 1]), 'same'),
                                               np.reshape(kernel, [1, 1, 1, -1, 1]), 'same'),
                               np.reshape(kernel, [1, 1, 1, 1, -1]), 'same')


def separable_filter2d(vol, kernel, mode='torch'):
    if np.all(kernel == 0):
        return vol
    if mode == 'torch':
        kernel = torch.as_tensor(kernel, dtype=vol.dtype, device=vol.device)
        if vol.ndim == 2:
            vol = vol.unsqueeze(0).unsqueeze(0)
        channels = vol.size(1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        padding = kernel.size(-1) // 2
        return F.conv2d(F.conv2d(vol, kernel.view(channels, 1, -1, 1), padding=(padding, 0), groups=channels),
                        kernel.view(channels, 1, 1, -1), padding=(0, padding), groups=channels)
    elif mode == 'np':
        if vol.ndim == 2:
            vol = np.expand_dims(vol, axis=(0, 1))
        return signal.convolve(signal.convolve(vol, np.reshape(kernel, [1, 1, -1, 1]), 'same'),
                                               np.reshape(kernel, [1, 1, 1, -1]), 'same')


def get_inverse_affine_matrix(dimension, center=None, rotate=None, translate=None, scale=None, shear=None):
    if dimension == 2:
        if center is None:
            center = (0, 0)
        if rotate is None:
            rotate = 0
        if translate is None:
            translate = (0, 0)
        if scale is None:
            scale = (1, 1)
        if shear is None:
            shear = (0, 0)
        rotate = math.radians(rotate)
        shear = [math.radians(sh) for sh in shear]

        T_inv = np.asarray([[1, 0, - translate[0]],
                            [0, 1, - translate[1]],
                            [0, 0, 1]], dtype=np.float32)
        C = np.asarray([[1, 0, center[0]],
                        [0, 1, center[1]],
                        [0, 0, 1]], dtype=np.float32)
        C_inv = np.asarray([[1, 0, - center[0]],
                            [0, 1, - center[1]],
                            [0, 0, 1]], dtype=np.float32)

        R_inv = np.asarray([[math.cos(rotate), - math.sin(rotate), 0],
                            [math.sin(rotate), math.cos(rotate), 0],
                            [0, 0, 1]], dtype=np.float32)
        S_inv = np.asarray([[1 / scale[0], 0, 0],
                            [0, 1 / scale[1], 0],
                            [0, 0, 1]], dtype=np.float32)

        Shx_inv = np.asarray([[1, - math.tan(shear[0]), 0],
                              [0, 1, 0],
                              [0, 0, 1]], dtype=np.float32)
        Shy_inv = np.asarray([[1, 0, 0],
                              [- math.tan(shear[1]), 1, 0],
                              [0, 0, 1]], dtype=np.float32)

        M = np.matmul(C,
                      np.matmul(Shx_inv,
                                np.matmul(Shy_inv,
                                          np.matmul(S_inv,
                                                    np.matmul(R_inv,
                                                              np.matmul(C_inv, T_inv))))))
    elif dimension == 3:
        if center is None:
            center = (0, 0, 0)
        if rotate is None:
            rotate = (0, 0, 0)
        if translate is None:
            translate = (0, 0, 0)
        if scale is None:
            scale = (1, 1, 1)
        if shear is None:
            shear = ((0, 0), (0, 0), (0, 0))
        rotate = [math.radians(r) for r in rotate]
        shear = [[math.radians(s) for s in sh] for sh in shear]

        T_inv = np.asarray([[1, 0, 0, - translate[0]],
                            [0, 1, 0, - translate[1]],
                            [0, 0, 1, - translate[2]],
                            [0, 0, 0, 1]], dtype=np.float32)
        C = np.asarray([[1, 0, 0, center[0]],
                        [0, 1, 0, center[1]],
                        [0, 0, 1, center[2]],
                        [0, 0, 0, 1]], dtype=np.float32)
        C_inv = np.asarray([[1, 0, 0, - center[0]],
                            [0, 1, 0, - center[1]],
                            [0, 0, 1, - center[2]],
                            [0, 0, 0, 1]], dtype=np.float32)
        Rx_inv = np.asarray([[1, 0, 0, 0],
                             [0, math.cos(rotate[0]), math.sin(rotate[0]), 0],
                             [0, - math.sin(rotate[0]), math.cos(rotate[0]), 0],
                             [0, 0, 0, 1]], dtype=np.float32)
        Ry_inv = np.asarray([[math.cos(rotate[1]), 0, - math.sin(rotate[1]), 0],
                             [0, 1, 0, 0],
                             [math.sin(rotate[1]), 0, math.cos(rotate[1]), 0],
                             [0, 0, 0, 1]], dtype=np.float32)
        Rz_inv = np.asarray([[math.cos(rotate[2]), math.sin(rotate[2]), 0, 0],
                             [- math.sin(rotate[2]), math.cos(rotate[2]), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.float32)
        S_inv = np.asarray([[1 / scale[0], 0, 0, 0],
                            [0, 1 / scale[1], 0, 0],
                            [0, 0, 1 / scale[2], 0],
                            [0, 0, 0, 1]], dtype=np.float32)

        Shx_inv = np.asarray([[1, - math.tan(shear[0][0]), - math.tan(shear[0][1]), 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], dtype=np.float32)
        Shy_inv = np.asarray([[1, 0, 0, 0],
                              [- math.tan(shear[1][0]), 1, - math.tan(shear[1][1]), 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], dtype=np.float32)
        Shz_inv = np.asarray([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [- math.tan(shear[2][0]), - math.tan(shear[2][1]), 1, 0],
                              [0, 0, 0, 1]], dtype=np.float32)

        M = np.matmul(C,
                      np.matmul(Shx_inv,
                                np.matmul(Shy_inv,
                                          np.matmul(Shz_inv,
                                                    np.matmul(S_inv,
                                                              np.matmul(Rx_inv,
                                                                        np.matmul(Ry_inv,
                                                                                  np.matmul(Rz_inv,
                                                                                            np.matmul(C_inv, T_inv))))))
                                          )
                                )
                      )
    else:
        raise NotImplementedError

    return M[:dimension]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def strsort(alist):
    alist.sort(key=natural_keys)
    return alist


def config_logging(filename, stream=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    f_handler = logging.FileHandler(filename)
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)

    logger.addHandler(f_handler)

    if stream:
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

    return logger


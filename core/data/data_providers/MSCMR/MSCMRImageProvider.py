# -*- coding: utf-8 -*-
"""
Image data provider for groupwise image registration on MSCMR datasets.

@author: Xinzhe Luo
@version: 0.1
"""

import glob
import itertools
import logging
import os
# import random
import numpy as np
import torch
from core import utils
from core.data import image_utils
from torch.utils.data import Dataset
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class ImageDataProvider(Dataset):
    dimension = 2
    def __init__(self, data_search_path, a_min=None, a_max=None,
                 image_suffix="image.nii.gz", label_suffix='label.nii.gz',
                 channels=1, label_intensities=(0, 200, 500, 600),
                 modalities=('DE', 'C0', 'T2'), **kwargs):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        self.data_search_path = data_search_path
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.channels = channels
        self.label_intensities = label_intensities
        self.modalities = modalities
        self.kwargs = kwargs
        self.logger = self.kwargs.pop("logger", logging)
        self.rand_ffd = self.kwargs.pop('rand_ffd', False)
        self.ffd_levels = self.kwargs.pop('ffd_levels', (5, 10, 20, 40))
        self.rand_specific = self.kwargs.pop('rand_specific', False)
        self.logger = self.kwargs.pop("logger", logging)
        self.compute_prob = self.kwargs.pop('compute_prob', False)
        self.rho = self.kwargs.pop('rho', 1)
        self.use_atlas = self.kwargs.pop('use_atlas', False)
        if self.use_atlas:
            self.atlas_name = self.kwargs.pop('atlas_name', 'C0_commonspace.nii.gz')
            atlas_label = image_utils.load_image_nii(os.path.join(self.data_search_path, self.atlas_name))[0]
            self.atlas_label = image_utils.get_one_hot_label(atlas_label,
                                                             label_intensities=self.label_intensities,
                                                             channel_first=True)
            self.atlas_prob = image_utils.get_distance_prob(self.atlas_label, rho=self.rho)

        self.image_names = self._find_data_names(self.data_search_path)

        self.logger.info("Dataset length: %s" % (len(self.image_names)))

    def __len__(self):
        return len(self.image_names)

    def get_image_name(self, idx):
        return self.image_names[idx]

    def _find_data_names(self, data_search_path):
        all_names = glob.glob(os.path.join(data_search_path, '*.nii.gz'))
        image_names = utils.strsort([name for name in all_names if self.image_suffix in name])

        if self.rand_ffd:
            level_key = lambda x: int(os.path.basename(x).split('_')[1][3:])
            all_ffd_names = glob.glob(os.path.join(data_search_path, 'rand_FFDs', '*.nii.gz'))
            all_ffd_names = [name for name in all_ffd_names if level_key(name) in self.ffd_levels]
            ffd_names = []
            rand_key = lambda x: os.path.basename(x).split('_')[-1][:-7]
            for _, g in itertools.groupby(sorted(all_ffd_names, key=level_key), key=level_key):
                names = list(g)
                m_names = [sorted([name for name in names
                                   if os.path.basename(name).split('_')[0] == m], key=rand_key)
                           for m in self.modalities]
                if self.rand_specific:
                    ffd_names.extend(list(zip(*m_names)))
                else:
                    ffd_names.extend(list(itertools.product(*m_names)))

        data_names = []
        key = lambda x: '_'.join(os.path.basename(x).split('_')[1:3])
        m_key = lambda x: self.modalities.index(os.path.basename(x).split('_')[-2])
        for _, g in itertools.groupby(sorted(image_names, key=key), key=key):  # group by subject and slice index
            names = list(g)
            slice_image_names = sorted(names, key=m_key)

            if self.rand_ffd:
                data_names.extend(list(itertools.product([slice_image_names], ffd_names)))
            else:
                data_names.append([slice_image_names])

        return data_names

    def _load_image(self, name, **kwargs):
        if '.nii.gz' in name:
            data = list(image_utils.load_image_nii(name, **kwargs))
            if self.dimension == 2 and data[0].shape[-1] == 1:
                data[0] = data[0].squeeze(-1)
            return data
        elif '.png' in name:
            return image_utils.load_image_png(name, **kwargs), None, None
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        data_names = self.image_names[item]

        images = []
        labels = []
        ffds = []
        probs = []

        affines = []
        headers = []

        for name in data_names[0]:
            image, affine, header = self._load_image(name)
            image = np.expand_dims(image, axis=0)
            label = self._load_image(name.replace(self.image_suffix, self.label_suffix))[0]
            label = image_utils.get_one_hot_label(label, label_intensities=self.label_intensities, channel_first=True)

            if self.compute_prob:
                probs.append(image_utils.get_distance_prob(label, rho=self.rho))

            images.append(image)
            labels.append(label)

            affines.append(affine)
            headers.append(header)

        if self.rand_ffd:
            for name in data_names[1]:
                ffd = self._load_image(name)[0].squeeze()
                ffds.append(ffd)

        return {'images': np.stack(images), 'labels': np.stack(labels),
                'ffds': np.stack(ffds) if self.rand_ffd else None,
                'probs': np.stack(probs) if self.compute_prob else None,
                'atlas_prob': self.atlas_prob if self.use_atlas else None,
                'affines': affines, 'headers': headers}

    def data_collate_fn(self, batch):

        I = torch.stack([torch.from_numpy(data.pop('images')) for data in batch])
        L = torch.stack([torch.from_numpy(data.pop('labels')) for data in batch])
        F = torch.stack([torch.from_numpy(data.pop('ffds')) for data in batch]) if self.rand_ffd else None
        P = torch.stack([torch.from_numpy(data.pop('probs')) for data in batch]) if self.compute_prob else None
        CP = torch.stack([torch.from_numpy(data.pop('atlas_prob'))
                          for data in batch]) if self.use_atlas else None

        AF = [data.pop('affines') for data in batch]
        HE = [data.pop('headers') for data in batch]

        return {'images': I, 'labels': L, 'ffds': F, 'probs': P, 'atlas_prob': CP,
                'affines': AF, 'headers': HE}


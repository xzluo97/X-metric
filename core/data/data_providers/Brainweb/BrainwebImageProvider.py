# -*- coding: utf-8 -*-
"""
Image data provider for groupwise image registration on Brainweb datasets.

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
from torch.utils.data import Dataset
from core.data import image_utils
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class ImageDataProvider(Dataset):

    def __init__(self, dimension, data_search_path, a_min=None, a_max=None,
                 training=True, level_specific=True, data_type='App', **kwargs):
        self.dimension = dimension
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        self.data_search_path = data_search_path
        self.training = training
        self.level_specific = level_specific
        self.data_type = data_type
        assert self.data_type in ['App', 'Reg']
        if self.data_type == 'App':
            self.level_specific = True
        self.kwargs = kwargs
        self.logger = self.kwargs.pop('logger', logging)
        self.ffd_levels = self.kwargs.pop('ffd_levels', ('5mm', '10mm', '15mm', '20mm'))
        self.modalities = self.kwargs.pop('modalities', ('t1', 't2', 'pd'))
        self.ffd_samples = self.kwargs.pop('ffd_samples', None)
        self.rand_specific = self.kwargs.pop('rand_specific', False)  # applicable only when level_specific=True

        self.data_pair_names = self._find_data_names(self.data_search_path)

    def __len__(self):
        return len(self.data_pair_names)

    def get_image_name(self, index):
        return self.data_pair_names[index]

    def _find_data_names(self, data_search_path, **kwargs):

        data_pair_names = []

        nii_names = utils.strsort(glob.glob(os.path.join(data_search_path, '*.nii.gz')))  # all image names

        lev_ffd_dirs = dict([(lev, [name for name in os.listdir(data_search_path) if 'randffd%s' % lev in name])
                             for lev in self.ffd_levels])
        lev_ffd_names = dict([(lev, utils.strsort(list(itertools.chain(*[glob.glob(os.path.join(data_search_path,
                                                                                                dirname, '*.nii.gz')
                                                                                   ) for dirname in lev_ffd_dirs[lev]])
                                                       )
                                                  )) for lev in lev_ffd_dirs.keys()])

        mod_key = lambda x: os.path.basename(x).split('_')[0]  # modality key

        if not self.level_specific:
            mod_pair_names = dict([(m, []) for m in self.modalities])

        for lev in self.ffd_levels:
            mod_lev_pair_names = dict([(m, []) for m in self.modalities])
            mod_ffd_names = dict([(m, [name for name in lev_ffd_names[lev] if m == os.path.basename(name).split('_')[0]])
                                  for m in self.modalities])
            for k, g in itertools.groupby(sorted(nii_names, key=mod_key), key=mod_key):
                if k in self.modalities:
                    img_names = list(g)
                    if self.ffd_samples is None:
                        ffd_names = mod_ffd_names[k]
                    else:
                        ffd_names = mod_ffd_names[k][:self.ffd_samples]
                    if self.training:
                        pair_names = [img_names + [name] for name in ffd_names]
                    else:
                        pair_names = list(itertools.product(img_names, ffd_names))

                    mod_lev_pair_names[k].extend(list(pair_names))

            if self.level_specific:
                if self.data_type == 'App':
                    data_pair_names.extend(list(itertools.chain(*mod_lev_pair_names.values())))
                elif self.data_type == 'Reg':
                    if self.rand_specific:
                        data_pair_names.extend(list(zip(*mod_lev_pair_names.values())))
                    else:
                        data_pair_names.extend(list(itertools.product(*mod_lev_pair_names.values())))
            else:
                for m in mod_pair_names.keys():
                    mod_pair_names[m].extend(mod_lev_pair_names[m])

        if not self.level_specific:
            data_pair_names.extend(list(itertools.product(*mod_pair_names.values())))

        return data_pair_names

    def _load_image(self, name, **kwargs):
        if '.nii.gz' in name:
            data = list(image_utils.load_image_nii(name, **kwargs))
            if self.dimension == 2:
                data[0] = data[0][data[0].shape[0] // 2]
            return data
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        pair_names = self.data_pair_names[item]
        if self.data_type == 'App':
            ffd = self._load_image(pair_names[-1])[0]
            ffd = ffd.transpose((-1, *range(self.dimension)))
            ffd = ffd[-self.dimension:]
            if self.training:
                im_img, aff, head = self._load_image(pair_names[0])
                im_img = np.expand_dims(im_img, 0)
                re_img = self._load_image(pair_names[1])[0]
                re_img = np.expand_dims(re_img, 0)
                return {'im_img': im_img, 're_img': re_img, 'ffd': ffd,
                        'aff': aff, 'head': head, 'mod': os.path.basename(pair_names[0]).split('_')[0]}
            else:
                image, aff, head = self._load_image(pair_names[0])
                image = np.expand_dims(image, 0)
                return {'image': image, 'ffd': ffd, 'aff': aff, 'head': head,
                        'mod': os.path.basename(pair_names[0]).split('_')[0]}
        elif self.data_type == 'Reg':
            ffds = []
            affines = []
            headers = []
            modalities = []
            if self.training:
                im_images = []
                re_images = []
                for i in range(len(pair_names)):
                    im_img, aff, head = self._load_image(pair_names[i][0])
                    im_img = np.expand_dims(im_img, 0)
                    re_img = self._load_image(pair_names[i][1])[0]
                    re_img = np.expand_dims(re_img, 0)
                    ffd = self._load_image(pair_names[i][2])[0]
                    ffd = ffd.transpose((-1, *range(self.dimension)))
                    ffd = ffd[-self.dimension:]
                    im_images.append(im_img)
                    re_images.append(re_img)
                    affines.append(aff)
                    headers.append(head)
                    ffds.append(ffd)
                    modalities.append(os.path.basename(pair_names[i][0]).split('_')[0])
                return {'im_images': np.stack(im_images), 're_images': np.stack(re_images),
                        'ffds': np.stack(ffds), 'headers': headers, 'affines': affines, 'modalities': modalities}
            else:
                images = []
                for i in range(len(pair_names)):
                    img, aff, head = self._load_image(pair_names[i][0])
                    img = np.expand_dims(img, 0)
                    ffd = self._load_image(pair_names[i][1])[0]
                    ffd = ffd.transpose((-1, *range(self.dimension)))
                    ffd = ffd[-self.dimension:]
                    images.append(img)
                    affines.append(aff)
                    headers.append(head)
                    ffds.append(ffd)
                    modalities.append(os.path.basename(pair_names[i][0]).split('_')[0])
                return {'images': np.stack(images), 'ffds': np.stack(ffds),
                        'affines': affines, 'headers': headers, 'modalities': modalities}
        else:
            raise NotImplementedError

    def data_collate_fn(self, batch):
        if self.data_type == 'Reg':
            AF = [data.pop('affines') for data in batch]
            HE = [data.pop('headers') for data in batch]
            MD = [data.pop('modalities') for data in batch]
            batch_tensor = dict([(k, torch.stack([torch.from_numpy(data[k]) for data in batch]))
                                 for k in batch[0].keys()])
            batch_tensor['affines'] = AF
            batch_tensor['headers'] = HE
            batch_tensor['modalities'] = MD
        elif self.data_type == 'App':
            AF = [data.pop('aff') for data in batch]
            HE = [data.pop('head') for data in batch]
            MD = [data.pop('mod') for data in batch]
            batch_tensor = dict([(k, torch.stack([torch.from_numpy(data[k]) for data in batch]))
                                 for k in batch[0].keys()])
            batch_tensor['aff'] = AF
            batch_tensor['head'] = HE
            batch_tensor['mod'] = MD
        else:
            raise NotImplementedError
        return batch_tensor


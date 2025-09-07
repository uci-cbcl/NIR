import csv
import glob
import math
import os

import matplotlib.colors as colors
import numpy as np
import nrrd
import nibabel as nib
from numpy.lib.arraysetops import isin
import scipy.io.wavfile as wavfile
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
import torch
from torch import fix, nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.nn import functional as F

def get_mgrid(sidelen, maxlen=None, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if maxlen is None:
        maxlen = sidelen
    if isinstance(maxlen, int) or isinstance(maxlen, float):
        maxlen = dim * (float(maxlen),)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...]
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...]
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    return pixel_coords, maxlen

def normalize_coords(coords, data_shape, maxlen):
    coords = coords.astype(np.float32)
    coords[..., 0] = (coords[..., 0] - (data_shape[0]-1)/2) / (maxlen[0] - 1)
    coords[..., 1] = (coords[..., 1] - (data_shape[1]-1)/2) / (maxlen[1] - 1)
    coords[..., 2] = (coords[..., 2] - (data_shape[2]-1)/2) / (maxlen[2] - 1)
    coords *= 2.
    coords = torch.Tensor(coords)
    return coords

def unnormalize_coords(coords, data_shape, maxlen):
    coords = coords.numpy() / 2.0
    coords[..., 0] = coords[..., 0] * (maxlen[0] - 1) + (data_shape[0] - 1) / 2
    coords[..., 1] = coords[..., 1] * (maxlen[1] - 1) + (data_shape[1] - 1) / 2
    coords[..., 2] = coords[..., 2] * (maxlen[2] - 1) + (data_shape[2] - 1) / 2 
    return coords

def mask_mri(shape, nsamp):
    mean = np.asarray(shape) // 2

    cov = np.eye(len(shape)) * (2*shape[0])
    samps = np.random.multivariate_normal(mean, cov, size=(1,nsamp))[0,...].astype(np.int32)

    mask = np.zeros(shape)

    mask[np.clip(samps[:, 0], 0, shape[0]-1), 
         np.clip(samps[:, 1], 0, shape[1]-1), 
         np.clip(samps[:, 2], 0, shape[2]-1)] = 1
    
    return mask

class ToTensor:
    def __init__(self, **kwargs):
        pass
        
    def __call__(self, img):
        img = torch.from_numpy(img.astype(np.float32))
        
        return img

class Identity:
    def __init__(self, **kwargs):
        pass

    def __call__(self, img):
        return img

class Normalize_CT:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, min_value=-1000, max_value=1000, **kwargs):
        assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        # self.value_range = max_value - min_value

    def __call__(self, img):
        img = np.clip(img, a_min=self.min_value, a_max=self.max_value)
        img = (img - self.min_value) / max(1, (self.max_value - self.min_value))
        # img = np.clip((img - self.min_value), a_min=0, a_max=None) / max(1, self.value_range)
        return img


class Resize:
    def __init__(self, size=[96, 96, 96], **kwargs):
        self.size = size

    def __call__(self, img):
        return skimage.transform.resize(img, self.size)
    

class Single_CT_Brain(Dataset):
    def __init__(self, path, maxlen=256):
        super().__init__()
        self.v = nrrd.read(os.path.join(path, 'img.nrrd'))[0]
        self.shape = self.v.shape
        transform = Compose([
            # Normalize_CT(0, 300)
            Identity()
        ])
        self.v = transform(self.v)
        self.channels = 1
        self.maxlen = maxlen

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.v[..., np.newaxis]


class Single_CT_Brain_TTO(Dataset):
    def __init__(self, img_path, pretrained_path, maxlen=256):
        super().__init__()
        self.v = nrrd.read(os.path.join(img_path, 'img.nrrd'))[0]
        self.shape = self.v.shape
        transform = Compose([
            # Normalize_CT(0, 300)
            Identity()
        ])
        self.v = transform(self.v)
        self.channels = 1
        self.maxlen = maxlen

        self.new_coords = np.load(os.path.join(pretrained_path, 'new_coords.npy'))

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.v[..., np.newaxis]
    

class Single_CT_Brain_Model:
    def __init__(self, volume, maxlen, shape):
        self.volume = volume
        self.maxlen = maxlen
        self.shape = shape

    def renormalized_coords(self, coords):
        coords[..., 0] = (coords[..., 0] * (self.maxlen[0] - 1)) / (self.shape[0] - 1) 
        coords[..., 1] = (coords[..., 1] * (self.maxlen[1] - 1)) / (self.shape[1] - 1) 
        coords[..., 2] = (coords[..., 2] * (self.maxlen[2] - 1)) / (self.shape[2] - 1) 

        return coords 

    def __call__(self, input):
        coords = self.renormalized_coords(input['coords'])
        return {'model_out': (F.grid_sample(self.volume[None, None], 
                                            coords[..., [2,1,0]], 
                                            mode='bilinear', 
                                            padding_mode='zeros').permute((0,2,3,4,1)) - 0.5) * 2}


class Single_MRI_Brain(Dataset):
    def __init__(self, path, maxlen=256):
        super().__init__()
        img = nib.load(path)
        img_orient = nib.as_closest_canonical(img)
        self.v = np.flip(img_orient.get_fdata().transpose(2,1,0), axis=(1,2)).transpose(2,1,0)
        self.shape = self.v.shape
        transform = Compose([
            Normalize_MR()
        ])
        self.v = transform(self.v)
        self.channels = 1
        self.maxlen = maxlen

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.v[..., np.newaxis]
    

class Implicit3DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_method='random', sample_exp=18, fixed_data_model=None):
        self.dataset = dataset
        self.mgrid, self.maxlen = get_mgrid(dataset.shape, dataset.maxlen, 3)
        self.data_shape = np.asarray(self.dataset.shape)
        data = (torch.from_numpy(self.dataset[0]) - 0.5) / 0.5
        self.data = data
        self.sample_exp = sample_exp
        self.sample_method = sample_method
        if self.sample_method == 'downsize':
            self.mask = mask_mri(self.data_shape, 200000)

        self.fixed_data_model = fixed_data_model

    def __len__(self):
        return 8 

    def __getitem__(self, idx):
        if self.sample_method == 'random':
            self.N_samples = int(2 ** self.sample_exp)
            data = self.data.reshape(-1, 1)
            perm = torch.randperm(data.shape[0])
            coord_idx = perm[:self.N_samples]
            data = data[coord_idx, :]
            coords = self.mgrid.reshape(-1, 3)[coord_idx, :]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_dict = {'idx': idx, 'coords': coords}
            gt_dict = {'img': data}

        elif self.sample_method == 'patch':
            assert self.sample_exp % 3 == 0
            patch_len = int(2 ** (self.sample_exp//3))
            random_corner_xyz = np.random.randint(0, self.data_shape-patch_len)
            mgrid = self.mgrid[:, random_corner_xyz[0]:random_corner_xyz[0]+patch_len,
                               random_corner_xyz[1]:random_corner_xyz[1]+patch_len,
                               random_corner_xyz[2]:random_corner_xyz[2]+patch_len]
            # coords = mgrid.reshape(-1, 3)
            coords = mgrid[0]
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_dict = {'idx': idx, 'coords': coords}
            gt_dict = {'img': data, 'win_scale': 1}
            # gt_dict = {'img': data, 'patch_len': patch_len, 'win_scale': 3}

        elif self.sample_method == 'mae':
            patch_len = int(2 ** 5)
            patch_num = int(2 ** self.sample_exp // (2 ** 15))
            random_corner_xyz = np.random.randint(0, self.data_shape-patch_len, [patch_num, 3])
            mgrid = []
            for patch_idx in range(patch_num):
                mgrid += [self.mgrid[:, random_corner_xyz[patch_idx,0]:random_corner_xyz[patch_idx,0]+patch_len,
                                random_corner_xyz[patch_idx,1]:random_corner_xyz[patch_idx,1]+patch_len,
                                random_corner_xyz[patch_idx,2]:random_corner_xyz[patch_idx,2]+patch_len]]

            # coords = np.asarray(mgrid).reshape(-1, 3)
            coords = np.stack(mgrid, axis=-2)[0]
            data = self.data[coords[...,0], coords[...,1], coords[...,2]].reshape(patch_len, patch_len, patch_len, -1)
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_dict = {'idx': idx, 'coords': coords}
            gt_dict = {'img': data, 'win_scale': 1}

        elif self.sample_method == 'downsize_random':
            jitter_size = 3
            mgrid = self.mgrid[:, 0::jitter_size, 0::jitter_size, 0::jitter_size]
            random_jitter = np.random.randint(0, jitter_size, mgrid.shape[-4:])
            coords = (mgrid + random_jitter)[0]
            coords = np.clip(coords, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_dict = {'idx': idx, 'coords': coords}
            gt_dict = {'img': data}

        elif self.sample_method == 'downsize_center':
            jitter_size = 3
            coords = self.mgrid[:, 1::jitter_size, 1::jitter_size, 1::jitter_size][0]
            coords = np.clip(coords, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_dict = {'idx': idx, 'coords': coords}
            gt_dict = {'img': data}

        elif self.sample_method == 'downsize_unified_random':
            jitter_size = 3
            mgrid = self.mgrid[:, 0::jitter_size, 0::jitter_size, 0::jitter_size]
            random_jitter = np.random.randint(0, jitter_size, 3)
            coords = (mgrid + random_jitter[None, None, None, None])[0]
            coords = np.clip(coords, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_dict = {'idx': idx, 'coords': coords}
            gt_dict = {'img': data}

        elif self.sample_method == 'downsize_unified_random_double':
            jitter_size = 3
            mgrid = self.mgrid[:, 0::jitter_size, 0::jitter_size, 0::jitter_size]
            random_jitter = np.random.randint(0, jitter_size, 3)
            coords = (mgrid + random_jitter[None, None, None, None])[0]
            coords = np.clip(coords, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_0_dict = {'idx': idx, 'coords': coords}
            gt_0_dict = {'img': data}
            in_1_dict = {'idx': idx, 'coords': coords}
            gt_1_dict = {'img': data}

            return in_0_dict, in_1_dict, gt_0_dict, gt_1_dict

        elif self.sample_method == 'downsize_center+unified_random':
            jitter_size = 3
            coords = self.mgrid[:, 1::jitter_size, 1::jitter_size, 1::jitter_size][0]
            coords = np.clip(coords, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_0_dict = {'idx': idx, 'coords': coords}
            gt_0_dict = {'img': data}

            mgrid = self.mgrid[:, 0::jitter_size, 0::jitter_size, 0::jitter_size]
            random_jitter = np.random.randint(0, jitter_size, 3)
            coords = (mgrid + random_jitter[None, None, None, None])[0]
            coords = np.clip(coords, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_1_dict = {'idx': idx, 'coords': coords}
            gt_1_dict = {'img': data}

            return in_0_dict, in_1_dict, gt_0_dict, gt_1_dict

        elif self.sample_method == 'downsize_unified_random+patch':
            jitter_size = 3
            mgrid = self.mgrid[:, 0::jitter_size, 0::jitter_size, 0::jitter_size]
            random_jitter = np.random.randint(0, jitter_size, 3)
            coords = (mgrid + random_jitter[None, None, None, None])[0]
            coords = np.clip(coords, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_0_dict = {'idx': idx, 'coords': coords}
            gt_0_dict = {'img': data}

            assert self.sample_exp % 3 == 0
            patch_len = int(2 ** (self.sample_exp//3))
            random_corner_xyz = np.random.randint(0, self.data_shape-patch_len)
            mgrid = self.mgrid[:, random_corner_xyz[0]:random_corner_xyz[0]+patch_len,
                               random_corner_xyz[1]:random_corner_xyz[1]+patch_len,
                               random_corner_xyz[2]:random_corner_xyz[2]+patch_len]
            # coords = mgrid.reshape(-1, 3)
            coords = mgrid[0]
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_1_dict = {'idx': idx, 'coords': coords}
            gt_1_dict = {'img': data, 'win_scale': 1}

            return in_0_dict, in_1_dict, gt_0_dict, gt_1_dict

        elif self.sample_method == 'downsize_unified_random+mae':
            jitter_size = 3
            mgrid = self.mgrid[:, 0::jitter_size, 0::jitter_size, 0::jitter_size]
            random_jitter = np.random.randint(0, jitter_size, 3)
            coords = (mgrid + random_jitter[None, None, None, None])[0]
            coords = np.clip(coords, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_0_dict = {'idx': idx, 'coords': coords}
            gt_0_dict = {'img': data}

            patch_len = int(2 ** 5)
            patch_num = int(2 ** self.sample_exp // (2 ** 15))
            random_corner_xyz = np.random.randint(0, self.data_shape-patch_len, [patch_num, 3])
            mgrid = []
            for patch_idx in range(patch_num):
                mgrid += [self.mgrid[:, random_corner_xyz[patch_idx,0]:random_corner_xyz[patch_idx,0]+patch_len,
                                random_corner_xyz[patch_idx,1]:random_corner_xyz[patch_idx,1]+patch_len,
                                random_corner_xyz[patch_idx,2]:random_corner_xyz[patch_idx,2]+patch_len]]

            # coords = np.asarray(mgrid).reshape(-1, 3)
            coords = np.stack(mgrid, axis=-2)[0]
            data = self.data[coords[...,0], coords[...,1], coords[...,2]].reshape(patch_len, patch_len, patch_len, -1)
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_1_dict = {'idx': idx, 'coords': coords}
            gt_1_dict = {'img': data, 'win_scale': 1}

            return in_0_dict, in_1_dict, gt_0_dict, gt_1_dict

        elif self.sample_method == 'downsize_center+patch':
            jitter_size = 3
            coords = self.mgrid[:, 1::jitter_size, 1::jitter_size, 1::jitter_size][0]
            coords = np.clip(coords, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_0_dict = {'idx': idx, 'coords': coords}
            gt_0_dict = {'img': data}

            assert self.sample_exp % 3 == 0
            patch_len = int(2 ** (self.sample_exp//3))
            random_corner_xyz = np.random.randint(0, self.data_shape-patch_len)
            mgrid = self.mgrid[:, random_corner_xyz[0]:random_corner_xyz[0]+patch_len,
                               random_corner_xyz[1]:random_corner_xyz[1]+patch_len,
                               random_corner_xyz[2]:random_corner_xyz[2]+patch_len]
            # coords = mgrid.reshape(-1, 3)
            coords = mgrid[0]
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_1_dict = {'idx': idx, 'coords': coords}
            gt_1_dict = {'img': data, 'win_scale': 1}

            return in_0_dict, in_1_dict, gt_0_dict, gt_1_dict

        elif self.sample_method == 'downsize_center+random':
            jitter_size = 3
            coords_center = self.mgrid[:, 1::jitter_size, 1::jitter_size, 1::jitter_size][0]
            mgrid = self.mgrid[:, 0::jitter_size, 0::jitter_size, 0::jitter_size]
            random_jitter = np.random.randint(0, jitter_size, mgrid.shape[-4:])
            coords_random = (mgrid + random_jitter)[0]
            coords_center = np.clip(coords_center, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            coords_random = np.clip(coords_random, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            data_center = self.data[coords_center[...,0], coords_center[...,1], coords_center[...,2]]
            data_random = self.data[coords_random[...,0], coords_random[...,1], coords_random[...,2]]
            coords_center = normalize_coords(coords_center, self.data_shape, self.maxlen)
            coords_random = normalize_coords(coords_random, self.data_shape, self.maxlen)

            in_dict = {'idx': idx, 'coords': np.concatenate([coords_center, coords_random], axis=0)}
            gt_dict = {'img': np.concatenate([data_center, data_random], axis=-1)}

        elif self.sample_method == 'downsize_random_syn':
            jitter_size = 3
            mgrid = self.mgrid[:, 0::jitter_size, 0::jitter_size, 0::jitter_size]
            random_jitter = np.random.random_sample(mgrid.shape[-4:]) * jitter_size
            coords = (mgrid + random_jitter)[0]
            coords = np.clip(coords, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            coords = normalize_coords(coords, self.data_shape, self.maxlen)
            with torch.no_grad():
                data = self.fixed_data_model({'coords':coords[None].cuda()})['model_out'][0].cpu()
            torch.cuda.empty_cache()

            in_dict = {'idx': idx, 'coords': coords}
            gt_dict = {'img': data}

        elif self.sample_method == 'downsize':
            step_size = 3
            min_len = min(self.data_shape)
            side_len = min_len // 3
            pad_xyz = np.clip((self.data_shape - side_len * 3) // 2, a_min=1, a_max=None)
            random_start_xyz = np.random.randint(0, pad_xyz)
            random_start_xyz[(self.data_shape - side_len * 3) == 0] = 0
            mgrid = self.mgrid[:, random_start_xyz[0]::step_size,
                               random_start_xyz[1]::step_size,
                               random_start_xyz[2]::step_size]
            # coords = mgrid.reshape(-1, 3)
            coords = mgrid[0]
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            mask = self.mask[random_start_xyz[0]::step_size,
                             random_start_xyz[1]::step_size,
                             random_start_xyz[2]::step_size]
            mask = np.fft.fftshift(mask).astype(np.complex64)
            mask = torch.from_numpy(mask)

            in_dict = {'idx': idx, 'coords': coords}
            gt_dict = {'img': data, 'mask': mask}

        elif self.sample_method == 'whole':
            coords = normalize_coords(self.mgrid[0], self.data_shape, self.maxlen)
            in_dict = {'coords': coords}
            gt_dict = {'img': self.data}
        else:
            pass

        return in_dict, gt_dict


class Implicit3DWrapper_TTO(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_method='random', sample_exp=18, fixed_data_model=None):
        self.dataset = dataset
        self.mgrid, self.maxlen = get_mgrid(dataset.shape, dataset.maxlen, 3)
        self.data_shape = np.asarray(self.dataset.shape)

        data = (torch.from_numpy(self.dataset[0]) - 0.5) / 0.5
        self.data = data

        new_coords = torch.from_numpy(self.dataset.new_coords)[None]
        self.new_coords = new_coords

        self.sample_exp = sample_exp
        self.sample_method = sample_method
        if self.sample_method == 'downsize':
            self.mask = mask_mri(self.data_shape, 200000)

        self.fixed_data_model = fixed_data_model

    def __len__(self):
        return 8 

    def __getitem__(self, idx):
        if self.sample_method == 'random':
            self.N_samples = int(2 ** self.sample_exp)
            data = self.data.reshape(-1, 1)
            perm = torch.randperm(data.shape[0])
            coord_idx = perm[:self.N_samples]
            data = data[coord_idx, :]
            coords = self.mgrid.reshape(-1, 3)[coord_idx, :]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_dict = {'idx': idx, 'coords': coords}
            gt_dict = {'img': data}

        elif self.sample_method == 'patch':
            assert self.sample_exp % 3 == 0
            patch_len = int(2 ** (self.sample_exp//3))
            random_corner_xyz = np.random.randint(0, self.data_shape-patch_len)
            mgrid = self.mgrid[:, random_corner_xyz[0]:random_corner_xyz[0]+patch_len,
                               random_corner_xyz[1]:random_corner_xyz[1]+patch_len,
                               random_corner_xyz[2]:random_corner_xyz[2]+patch_len]
            # coords = mgrid.reshape(-1, 3)
            coords = mgrid[0]
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_dict = {'idx': idx, 'coords': coords}
            gt_dict = {'img': data, 'win_scale': 1}
            # gt_dict = {'img': data, 'patch_len': patch_len, 'win_scale': 3}

        elif self.sample_method == 'mae':
            patch_len = int(2 ** 5)
            patch_num = int(2 ** self.sample_exp // (2 ** 15))
            random_corner_xyz = np.random.randint(0, self.data_shape-patch_len, [patch_num, 3])
            mgrid = []
            for patch_idx in range(patch_num):
                mgrid += [self.mgrid[:, random_corner_xyz[patch_idx,0]:random_corner_xyz[patch_idx,0]+patch_len,
                                random_corner_xyz[patch_idx,1]:random_corner_xyz[patch_idx,1]+patch_len,
                                random_corner_xyz[patch_idx,2]:random_corner_xyz[patch_idx,2]+patch_len]]

            # coords = np.asarray(mgrid).reshape(-1, 3)
            coords = np.stack(mgrid, axis=-2)[0]
            data = self.data[coords[...,0], coords[...,1], coords[...,2]].reshape(patch_len, patch_len, patch_len, -1)
            coords = normalize_coords(coords, self.data_shape, self.maxlen)

            in_dict = {'idx': idx, 'coords': coords}
            gt_dict = {'img': data, 'win_scale': 1}

        elif self.sample_method == 'downsize_unified_random':
            jitter_size = 3
            
            mgrid = self.mgrid[:, 0::jitter_size, 0::jitter_size, 0::jitter_size]
            random_jitter = np.random.randint(0, jitter_size, 3)
            coords = (mgrid + random_jitter[None, None, None, None])[0]
            coords = mgrid[0]
            coords = np.clip(coords, a_min=0, a_max=(self.data_shape-1)[np.newaxis, np.newaxis, np.newaxis, :])
            data = self.data[coords[...,0], coords[...,1], coords[...,2]]

            new_coords = self.new_coords[0, coords[...,0], coords[...,1], coords[...,2]]

            in_dict = {'idx': idx, 'coords': new_coords}
            gt_dict = {'img': data}

        elif self.sample_method == 'whole':
            new_coords = self.new_coords[0]
            in_dict = {'coords': new_coords}
            gt_dict = {'img': self.data}
        else:
            pass

        return in_dict, gt_dict



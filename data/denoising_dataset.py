# -*- coding:utf-8 -*-
#@Time : 2023/9/8 16:10
#@Author: pc
#@File : denoising_dataset.py

import os
import numpy as np

from torch.utils import data as data
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from scipy.io import loadmat
from util.image_util import random_augmentation


##############################SIDD dataset##############################
@DATASET_REGISTRY.register()
class SIDDData(data.Dataset):
    def __init__(self, opt):
        '''
        Args:
            h5_path (str): path of the hdf5 file
            length (int): length of Datasets
        '''
        super(SIDDData, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.input_folder = opt['dataroot_input']
        self.gt_folder = opt['dataroot_gt']

        self.input_files = [os.path.join(self.input_folder, filename)for filename in list(scandir(self.input_folder))]
        self.noisy_files = [os.path.join(self.gt_folder, filename)for filename in list(scandir(self.gt_folder))]


    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        input_path = self.file_client.get(self.input_files[index], 'lq')
        gt_path = self.file_client.get(self.noisy_files[index], 'gt')
        noisy_img = imfrombytes(input_path, float32=True)
        gt_img = imfrombytes(gt_path, float32=True)

        gt_img, noisy_img = paired_random_crop(gt_img, noisy_img, self.opt['patch_size'], 1)

        gt_img, noisy_img = random_augmentation(gt_img, noisy_img)

        gt_img, noisy_img = img2tensor([gt_img, noisy_img], bgr2rgb=True, float32=True)

        return {'lq': noisy_img, 'gt': gt_img}

@DATASET_REGISTRY.register()
class SIDDValData(data.Dataset):
    def __init__(self, opt):

        self.opt = opt
        path = self.opt['dataroot']
        val_data_dict = loadmat(os.path.join(path, 'ValidationNoisyBlocksSrgb.mat'))
        val_data_noisy = val_data_dict['ValidationNoisyBlocksSrgb']
        val_data_dict = loadmat(os.path.join(path, 'ValidationGtBlocksSrgb.mat'))
        val_data_gt = val_data_dict['ValidationGtBlocksSrgb']
        self.num_img, self.num_block, h_, w_, c_ = val_data_gt.shape
        self.val_data_noisy = np.reshape(val_data_noisy, (-1, h_, w_, c_))
        self.val_data_gt = np.reshape(val_data_gt, (-1, h_, w_, c_))


    def __len__(self):
        return self.num_img * self.num_block

    def __getitem__(self, index):

        noisy_img, gt_img = self.val_data_noisy[index], self.val_data_gt[index]
        noisy_img = noisy_img/255
        gt_img = gt_img/255
        gt_img, noisy_img = img2tensor([gt_img, noisy_img], bgr2rgb=False, float32=True)
        return {'lq': noisy_img, 'gt': gt_img, 'lq_path': self.opt['dataroot']}
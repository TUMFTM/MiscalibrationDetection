#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
from .bin2depth import get_depth


class Kittiloader(object):
    """
    param kittiDir: KITTI dataset root path, e.g. ~/data/kitti/
    param mode: 'train', 'test' or 'val'
    param cam: camera id. 2 represents the left cam, 3 represents the right one
    """
    def __init__(self, kittiDir, mode, perturb_filenames, cam=2, augmentation=None, extrinsic=False):
        self.mode = mode
        self.cam = cam
        self.files = []
        self.kitti_root = kittiDir
        self.perturb_filenames = perturb_filenames
        self.augmentation = augmentation
        self.extrinsic = extrinsic
        # read filenames files
        dir_name = os.path.dirname(os.path.realpath(__file__))
        # currpath = os.path.join(dir_name, '..')
        filepath = os.path.join(dir_name, 'filenames', 'eigen_{}_files.txt'.format(self.mode))
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist. Please check the path and try again.")
        
        with open(filepath, 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                data_info = data.split(' ')

                self.files.append({
                    "l_rgb": data_info[0],
                    "r_rgb": data_info[1],
                    "cam_intrin": data_info[2],
                    "depth": data_info[3],
                    "name": data_info[4],
                    "perturb_filename": self.perturb_filenames
                })

    def data_length(self):
        return len(self.files)

    def _check_path(self, filename, err_info):
        file_path = os.path.join(self.kitti_root, filename)
        assert os.path.exists(file_path), err_info
        return file_path

    def _read_data(self, item_files):
        l_rgb_path = self._check_path(item_files['l_rgb'], err_info="Panic::Cannot find Left Image. Filename: {}".format(item_files['l_rgb']))
        cam_path = self._check_path(item_files['cam_intrin'], err_info="Panic::Cannot find Camera Infos. Filename: {}".format(item_files['cam_intrin']))
        depth_path = self._check_path(item_files['depth'], err_info="Panic::Cannot find depth file. Filename: {}".format(item_files['depth']))
        perturb_path = self._check_path(item_files['perturb_filename'], err_info="Panic::Cannot find perturbation file. Filename: {}".format(item_files['perturb_filename']))

        l_rgb = Image.open(l_rgb_path).convert('RGB')
        w, h = l_rgb.size
        depth, depth_neg = get_depth(cam_path, depth_path, [h,w], perturb_path, item_files['name'], cam=self.cam, vel_depth=True, augmentation=self.augmentation, extrinsic=self.extrinsic) 

        data = {}
        data['left_img'] = l_rgb
        data['depth'] = depth.astype(np.float32)
        data['depth_neg'] = depth_neg.astype(np.float32)
        data['name'] = item_files['name']
        return data

    def load_item(self, idx):
        """
        load an item for training or test
        interp_method can be selected from [linear', 'nyu']
        """
        item_files = self.files[idx]
        data_item = self._read_data(item_files)

        return data_item

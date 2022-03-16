# -*- coding: UTF-8 -*-
import os
import torch
import torch.utils.data as data
import torchvision
import random
from PIL import Image, ImageOps
import utils.preprocess as preprocess
import utils.readpfm as rp
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
import cv2
import imageio


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    img_BGR = cv2.imread(path)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    return img_RGB

def edge_loader(path):
    gt = np.array(Image.open(path), dtype=np.float32)

    gt /= 255.
    gt[gt >= 0.5] = 1
    gt[gt < 0.05] = 0
    return gt

def load_LFdata(LF_dir, loader, edloader):
    for i in range(81):
        if i == 40:
            img_name = os.path.join(LF_dir, 'input_Cam0%.2d.png' % i)
            try:
                img_rgb = loader(img_name)
            except:
                print(LF_dir + '/input_Cam0%.2d.png..does not exist' % i)

    edge_name = os.path.join(LF_dir, 'BinaryEdge.png')
    edge_array = edloader(edge_name)

    return img_rgb, edge_array

def save_disparity_jet(disparity, filename):
    max_disp = np.nanmax(disparity[disparity != np.inf])
    min_disp = np.nanmin(disparity[disparity != np.inf])
    disparity = (disparity - min_disp) / (max_disp - min_disp)
    disparity = (disparity * 255.0).astype(np.uint8)            # normalized to [0,1] and then multiplied by 255
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_WINTER)
    cv2.imwrite(filename, disparity)


class myImageFloder(data.Dataset):
    def __init__(self, im_all, cropsize, id_start, nums, iftrain, loader=default_loader, ed_loader=edge_loader):
        self.im_all = im_all
        self.cropsize = cropsize
        self.iftrain = iftrain
        self.loader = loader
        self.ed_loader = ed_loader

        self.id_start = id_start
        self.nums = nums

        self.processed = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        img_dir = self.im_all[index]
        center_im, edge_binary = load_LFdata(img_dir, loader=self.loader, edloader=self.ed_loader)

        edge_binary = np.ascontiguousarray(edge_binary, dtype=np.float32)

        if self.iftrain == True:
            input_im, edge_binary_label = preprocess.data_crop(center_im, edge_binary, self.cropsize)
            input_im, edge_binary_label = preprocess.data_augment(input_im, edge_binary_label)

        else:
            input_im, edge_binary_label = preprocess.test_dataprocess(center_im, edge_binary, self.cropsize, self.id_start)

        input_im = np.ascontiguousarray(input_im, dtype=np.float32)

        center_input = self.processed(
                                 np.expand_dims(input_im, 2))

        edge_binary_label = np.ascontiguousarray(edge_binary_label, dtype=np.float32)

        return center_input, edge_binary_label

    def __len__(self):
        return len(self.im_all)

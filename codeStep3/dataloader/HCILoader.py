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

def disparity_loader(path):
    return np.float32(rp.readPFM(path))

def edge_loader(path):
    gt = np.array(Image.open(path), dtype=np.float32)

    gt /= 255.
    gt[gt >= 0.5] = 1
    gt[gt < 0.05] = 0
    return gt

def load_LFdata(LF_dir, loader, dploader, edloader):
    img_0d = np.zeros((512, 512, 9, 3), np.uint8)
    img_90d = np.zeros((512, 512, 9, 3), np.uint8)
    for i in range(81):
        (r, c) = divmod(i, 9)
        if r == 4 or c == 4:
            img_name = os.path.join(LF_dir, 'input_Cam0%.2d.png' % i)
            try:
                img_rgb = loader(img_name)
            except:
                print(LF_dir + '/input_Cam0%.2d.png..does not exist' % i)
            if r == 4:
                img_0d[:, :, c, :] = img_rgb
            if c == 4:
                img_90d[:, :, 8-r, :] = img_rgb    # the direction of '90' is from bottom to top

    disp_name = os.path.join(LF_dir, 'gt_disp_lowres.pfm')
    try:
        data_disp = dploader(disp_name)
    except:
        print(LF_dir + '/gt_disp_lowres.pfm..does not exist')

    edge_name = os.path.join(LF_dir, 'BinaryEdge.png')
    if 'Inria_syn_lf_datasets' in edge_name:
        edge_array = np.zeros((512,512), dtype=np.float32)
    else:
        edge_array = edloader(edge_name)

    return img_0d, img_90d, data_disp, edge_array

def save_disparity_jet(disparity, filename):
    max_disp = np.nanmax(disparity[disparity != np.inf])
    min_disp = np.nanmin(disparity[disparity != np.inf])
    disparity = (disparity - min_disp) / (max_disp - min_disp)
    disparity = (disparity * 255.0).astype(np.uint8)            # normalized to [0,1] and then multiplied by 255
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_WINTER)
    cv2.imwrite(filename, disparity)


class myImageFloder(data.Dataset):
    def __init__(self, im_all, cropsize, id_start, nums, boolmask_img4, boolmask_img6, boolmask_img15, iftrain,
                                                        loader=default_loader, dploader=disparity_loader, ed_loader=edge_loader):
        self.im_all = im_all
        self.cropsize = cropsize
        self.iftrain = iftrain
        self.loader = loader
        self.dploader = dploader
        self.ed_loader = ed_loader
        self.boolmask_img4 = boolmask_img4
        self.boolmask_img6 = boolmask_img6
        self.boolmask_img15 = boolmask_img15

        self.id_start = id_start
        self.nums = nums
        self.center_posi = nums // 2

        self.processed = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        img_dir = self.im_all[index]
        class_name = os.path.basename(img_dir)
        img_0d, img_90d, disp, edge_binary = load_LFdata(img_dir, loader=self.loader, dploader=self.dploader, edloader=self.ed_loader)

        disp = np.ascontiguousarray(disp, dtype=np.float32)
        edge_binary = np.ascontiguousarray(edge_binary, dtype=np.float32)

        if self.iftrain == True:
            input_0d, input_90d,\
            disp_batch_label, edge_binary_label = preprocess.data_crop(img_0d, img_90d, disp, edge_binary, self.cropsize, class_name,
                                                      self.boolmask_img4, self.boolmask_img6, self.boolmask_img15)
            input_0d, input_90d,\
            disp_batch_label, edge_binary_label = preprocess.data_augment(input_0d, input_90d, disp_batch_label, edge_binary_label)

        else:
            input_0d, input_90d,\
            disp_batch_label, edge_binary_label = preprocess.test_dataprocess(img_0d, img_90d, disp, edge_binary, self.cropsize, self.id_start)

        input_0d = np.ascontiguousarray(input_0d, dtype=np.float32)
        input_90d = np.ascontiguousarray(input_90d, dtype=np.float32)

        center_input = self.processed(
                                 np.expand_dims(input_0d[:, :, self.center_posi], 2))

        view_input = [[], [], [], []]
        for i in range(self.nums):
            if i != (self.center_posi):
                view_posi = abs(self.nums // 2 - i) - 1

                temp_arr_0d = np.expand_dims(input_0d[:, :, i], 2)
                temp_tensor_0d = self.processed(temp_arr_0d)
                view_input[view_posi].append(temp_tensor_0d)

                temp_arr_90d = np.expand_dims(input_90d[:, :, i], 2)
                temp_tensor_90d = self.processed(temp_arr_90d)
                view_input[view_posi].append(temp_tensor_90d)

        view1_input = torch.cat(view_input[0], dim=0)
        view2_input = torch.cat(view_input[1], dim=0)
        view3_input = torch.cat(view_input[2], dim=0)
        view4_input = torch.cat(view_input[3], dim=0)

        disp_batch_label = np.ascontiguousarray(disp_batch_label, dtype=np.float32)
        edge_binary_label = np.ascontiguousarray(edge_binary_label, dtype=np.float32)

        return center_input, view1_input, view2_input, view3_input, view4_input, disp_batch_label, edge_binary_label

    def __len__(self):
        return len(self.im_all)


def load_Sub_LFdata(LF_dir, loader):
    img_0d = np.zeros((512, 512, 9, 3), np.uint8)
    img_90d = np.zeros((512, 512, 9, 3), np.uint8)
    for i in range(81):
        (r, c) = divmod(i, 9)
        if r == 4 or c == 4:
            img_name = os.path.join(LF_dir, 'input_Cam0%.2d.png' % i)
            try:
                img_rgb = loader(img_name)
            except:
                print(LF_dir + '/input_Cam0%.2d.png..does not exist' % i)
            if r == 4:
                img_0d[:, :, c, :] = img_rgb
            if c == 4:
                img_90d[:, :, 8 - r, :] = img_rgb  # the direction of '90' is from bottom to top

    return img_0d, img_90d

class submissonFloder(data.Dataset):
    def __init__(self, im_all, nums, loader=default_loader):
        self.im_all = im_all
        self.loader = loader

        self.Setting_AngualrViews = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.nums = nums
        self.center_posi = nums // 2
        self.processed = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        img_dir = self.im_all[index]
        class_name = os.path.basename(img_dir)
        img_0d, img_90d = load_Sub_LFdata(img_dir, loader=self.loader)

        R = 0.299  ### 0,1,2,3 = R, G, B, Gray // 0.299 0.587 0.114
        G = 0.587
        B = 0.114
        input_batch_0d = np.squeeze(
            R * img_0d[:, :, :, 0].astype('float32') +
            G * img_0d[:, :, :, 1].astype('float32') +
            B * img_0d[:, :, :, 2].astype('float32'))

        input_batch_90d = np.squeeze(
            R * img_90d[:, :, :, 0].astype('float32') +
            G * img_90d[:, :, :, 1].astype('float32') +
            B * img_90d[:, :, :, 2].astype('float32'))

        input_batch_0d = np.float32((1 / 255.) * input_batch_0d)
        input_batch_90d = np.float32((1 / 255.) * input_batch_90d)

        input_batch_0d = np.clip(input_batch_0d, 0.0, 1.0)
        input_batch_90d = np.clip(input_batch_90d, 0.0, 1.0)

        input_batch_0d = np.ascontiguousarray(input_batch_0d, dtype=np.float32)
        input_batch_90d = np.ascontiguousarray(input_batch_90d, dtype=np.float32)

        center_input = self.processed(
                                 np.expand_dims(input_batch_0d[:, :, self.center_posi], 2))

        view_input = [[], [], [], []]
        for i in range(self.nums):
            if i != (self.center_posi):
                view_posi = abs(self.nums // 2 - i) - 1

                temp_arr_0d = np.expand_dims(input_batch_0d[:, :, i], 2)
                temp_tensor_0d = self.processed(temp_arr_0d)
                view_input[view_posi].append(temp_tensor_0d)

                temp_arr_90d = np.expand_dims(input_batch_90d[:, :, i], 2)
                temp_tensor_90d = self.processed(temp_arr_90d)
                view_input[view_posi].append(temp_tensor_90d)

        view1_input = torch.cat(view_input[0], dim=0)
        view2_input = torch.cat(view_input[1], dim=0)
        view3_input = torch.cat(view_input[2], dim=0)
        view4_input = torch.cat(view_input[3], dim=0)

        return center_input, view1_input, view2_input, view3_input, view4_input, class_name

    def __len__(self):
        return len(self.im_all)
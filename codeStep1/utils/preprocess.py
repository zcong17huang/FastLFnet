# -*- coding: UTF-8 -*-
import random
import numpy as np


def data_crop(img_0d, img_90d, disp, input_size, class_name, boolmask_img4, boolmask_img6, boolmask_img15):
    sum_diff = 0
    valid = 0

    # when it is an untextured or reflective region：
    while (sum_diff < 0.01 * input_size * input_size or valid < 1):
        """//Variable for gray conversion//"""
        # randomly set the R/G/B values for converting to grayscale images
        rand_3color = 0.05 + np.array([random.random() for _ in range(3)])
        rand_3color = rand_3color / np.sum(rand_3color)
        R = rand_3color[0]
        G = rand_3color[1]
        B = rand_3color[2]

        # randomly generate scale ranges
        kk = random.randint(0,16)
        if kk < 8:
            scale = 1
        elif kk < 14:
            scale = 2
        else:
            scale = 3

        idx_start = random.randint(0, 512 - scale * input_size - 1)
        idy_start = random.randint(0, 512 - scale * input_size - 1)
        valid = 1

        """
        boolmask: reflection masks for images(4,6,15)
        """
        if class_name == 'kitchen' or 'museum' or 'vinyl':
            if class_name == 'kitchen':
                a_tmp = boolmask_img4
            elif class_name == 'museum':
                a_tmp = boolmask_img6
            else:
                a_tmp = boolmask_img15
            if np.sum(a_tmp[idx_start: idx_start + scale * input_size:scale,
                              idy_start: idy_start + scale * input_size:scale]) > 0:
                valid = 0

        if valid > 0:
            # discrimination of textureless areas
            image_center = (1 / 255) * np.squeeze(
                R * img_0d[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 4, 0].astype('float32') +
                G * img_0d[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 4, 1].astype('float32') +
                B * img_0d[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 4, 2].astype('float32'))
            sum_diff = np.sum(np.abs(image_center - np.squeeze(image_center[int(0.5 * input_size), int(0.5 * input_size)])))

            # convert to grayscale image
            input_0d = np.squeeze(
                R * img_0d[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, :, 0].astype('float32') +
                G * img_0d[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, :, 1].astype('float32') +
                B * img_0d[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, :, 2].astype('float32'))

            input_90d = np.squeeze(
                R * img_90d[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, :, 0].astype('float32') +
                G * img_90d[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, :, 1].astype('float32') +
                B * img_90d[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, :, 2].astype('float32'))

            disp_batch_label = (1.0 / scale) * disp[idx_start: idx_start + scale * input_size:scale,
                                                                    idy_start: idy_start + scale * input_size:scale]
    input_0d = np.float32((1 / 255.) * input_0d)
    input_90d = np.float32((1 / 255.) * input_90d)

    return input_0d, input_90d, disp_batch_label  # shape = [input_size, input_size, 9]


def data_augment(input_0d, input_90d, disp_batch_label):
    """
        For Data augmentation
        (rotation, transpose and gamma)
    """
    # gamma
    gray_rand = 0.4 * random.random() + 0.8  # gamma value[0.8, 1.2)
    input_0d = pow(input_0d, gray_rand)
    input_90d = pow(input_90d, gray_rand)

    """ transpose """
    transp_rand = random.randint(0, 1)
    if transp_rand == 1:
        input_tmp_0d = np.copy(np.transpose(input_0d, (1, 0, 2)))
        input_tmp_90d = np.copy(np.transpose(input_90d, (1, 0, 2)))

        input_90d = input_tmp_0d[:, :, ::-1]
        input_0d = input_tmp_90d[:, :, ::-1]

        disp_batch_label = np.copy(np.transpose(disp_batch_label, (1, 0)))

    """ rotation """
    # 0 means unchanged; 1, 2, 3 means three rotation angles
    rotation_rand = random.randint(0, 3)
    """ 90 """
    if rotation_rand == 1:
        input_0d_tmp = np.copy(np.rot90(input_0d, 1))
        input_90d_tmp = np.copy(np.rot90(input_90d, 1))

        input_90d = input_0d_tmp
        input_0d = input_90d_tmp[:, :, ::-1]

        disp_batch_label = np.copy(np.rot90(disp_batch_label, 1))
    """ 180 """
    if rotation_rand == 2:
        input_0d_tmp = np.copy(np.rot90(input_0d, 2))
        input_90d_tmp = np.copy(np.rot90(input_90d, 2))

        input_0d = input_0d_tmp[:, :, ::-1]
        input_90d = input_90d_tmp[:, :, ::-1]

        disp_batch_label = np.copy(np.rot90(disp_batch_label, 2))
    """ 270 """
    if rotation_rand == 3:
        input_0d_tmp = np.copy(np.rot90(input_0d, 3))
        input_90d_tmp = np.copy(np.rot90(input_90d, 3))

        input_90d = input_0d_tmp[:, :, ::-1]
        input_0d = input_90d_tmp

        disp_batch_label = np.copy(np.rot90(disp_batch_label, 3))

    """ gaussian noise """
    noise_rand = random.randint(0, 11)
    if noise_rand == 0:
        Sigma = random.uniform(0.0, 1.0) * 0.2
        size_all = input_0d.shape[0] * input_0d.shape[1] * input_0d.shape[2]

        gauss = np.array([random.gauss(0.0, Sigma) for _ in range(size_all)]
                         ).reshape((input_0d.shape[0], input_0d.shape[1], input_0d.shape[2]))

        input_0d = np.float32(np.clip(input_0d + gauss, 0.0, 1.0))
        input_90d = np.float32(np.clip(input_90d + gauss, 0.0, 1.0))

    return input_0d, input_90d, disp_batch_label


def test_dataprocess(img_0d, img_90d, disp, input_size, id_start):
    R = 0.299  ### 0,1,2,3 = R, G, B, Gray // 0.299 0.587 0.114
    G = 0.587
    B = 0.114

    input_batch_0d = np.squeeze(
            R * img_0d[id_start: id_start + input_size, id_start: id_start + input_size, :, 0].astype('float32') +
            G * img_0d[id_start: id_start + input_size, id_start: id_start + input_size, :, 1].astype('float32') +
            B * img_0d[id_start: id_start + input_size, id_start: id_start + input_size, :, 2].astype('float32'))

    input_batch_90d = np.squeeze(
            R * img_90d[id_start: id_start + input_size, id_start: id_start + input_size, :, 0].astype('float32') +
            G * img_90d[id_start: id_start + input_size, id_start: id_start + input_size, :, 1].astype('float32') +
            B * img_90d[id_start: id_start + input_size, id_start: id_start + input_size, :, 2].astype('float32'))

    disp_batch_label = disp[id_start: id_start + input_size, id_start: id_start + input_size]

    input_batch_0d = np.float32((1 / 255.) * input_batch_0d)
    input_batch_90d = np.float32((1 / 255.) * input_batch_90d)

    input_batch_0d = np.clip(input_batch_0d, 0.0, 1.0)
    input_batch_90d = np.clip(input_batch_90d, 0.0, 1.0)

    return input_batch_0d, input_batch_90d, disp_batch_label





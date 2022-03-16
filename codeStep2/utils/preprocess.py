# -*- coding: UTF-8 -*-
import random
import numpy as np


def data_crop(input_im, disp, input_size):
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

        if valid > 0:
            # discrimination of textureless areas
            image_center = (1 / 255) * np.squeeze(
                R * input_im[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 0].astype('float32') +
                G * input_im[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 1].astype('float32') +
                B * input_im[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 2].astype('float32'))
            sum_diff = np.sum(np.abs(image_center - np.squeeze(image_center[int(0.5 * input_size), int(0.5 * input_size)])))

            # convert to grayscale image
            input = np.squeeze(
                R * input_im[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 0].astype('float32') +
                G * input_im[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 1].astype('float32') +
                B * input_im[idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 2].astype('float32'))

            disp_batch_label = (1.0 / scale) * disp[idx_start: idx_start + scale * input_size:scale,
                                                                    idy_start: idy_start + scale * input_size:scale]
    input = np.float32((1 / 255.) * input)

    return input, disp_batch_label  # shape = [input_size, input_size]


def data_augment(input, disp_batch_label):
    """
        For Data augmentation
        (rotation, transpose and gamma)
    """
    # gamma
    gray_rand = 0.4 * random.random() + 0.8  # gamma value[0.8, 1.2)
    input = pow(input, gray_rand)

    """ transpose """
    transp_rand = random.randint(0, 1)
    if transp_rand == 1:
        input = np.copy(np.transpose(input, (1, 0)))

        disp_batch_label = np.copy(np.transpose(disp_batch_label, (1, 0)))

    """ rotation """
    # 0 means unchanged; 1, 2, 3 means three rotation angles
    rotation_rand = random.randint(0, 3)
    """ 90 """
    if rotation_rand == 1:
        input = np.copy(np.rot90(input, 1))

        disp_batch_label = np.copy(np.rot90(disp_batch_label, 1))
    """ 180 """
    if rotation_rand == 2:
        input = np.copy(np.rot90(input, 2))

        disp_batch_label = np.copy(np.rot90(disp_batch_label, 2))
    """ 270 """
    if rotation_rand == 3:
        input = np.copy(np.rot90(input, 3))

        disp_batch_label = np.copy(np.rot90(disp_batch_label, 3))

    """ gaussian noise """
    noise_rand = random.randint(0, 11)
    if noise_rand == 0:
        Sigma = random.uniform(0.0, 1.0) * 0.2
        size_all = input.shape[0] * input.shape[1]

        gauss = np.array([random.gauss(0.0, Sigma) for _ in range(size_all)]
                         ).reshape((input.shape[0], input.shape[1]))

        input = np.float32(np.clip(input + gauss, 0.0, 1.0))

    return input, disp_batch_label


def test_dataprocess(center_im, disp, input_size, id_start):
    R = 0.299  ### 0,1,2,3 = R, G, B, Gray // 0.299 0.587 0.114
    G = 0.587
    B = 0.114

    input = np.squeeze(
            R * center_im[id_start: id_start + input_size, id_start: id_start + input_size, 0].astype('float32') +
            G * center_im[id_start: id_start + input_size, id_start: id_start + input_size, 1].astype('float32') +
            B * center_im[id_start: id_start + input_size, id_start: id_start + input_size, 2].astype('float32'))

    disp_batch_label = disp[id_start: id_start + input_size, id_start: id_start + input_size]

    input = np.float32((1 / 255.) * input)

    input = np.clip(input, 0.0, 1.0)

    return input, disp_batch_label





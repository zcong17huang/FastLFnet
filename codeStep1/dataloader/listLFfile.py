# -*- coding: UTF-8 -*-
from PIL import Image
import os
import os.path

def Traindataloader(filepath):
    filepath = os.path.normcase(filepath)
    imgs = []
    full_data_path = os.path.join(filepath, 'full_data')
    train_class = ['additional']

    for cla in train_class:
        class_path = os.path.join(full_data_path, cla)
        subclass_path = os.listdir(class_path)
        for subcla in subclass_path:
            img_dir = os.path.join(class_path, subcla)
            if os.path.isdir(img_dir):
                imgs.append(img_dir)

    return imgs

def Testdataloader(filepath):
    filepath = os.path.normcase(filepath)
    imgs = []
    full_data_path = os.path.join(filepath, 'full_data')
    train_class = ['stratified', 'training']

    for cla in train_class:
        class_path = os.path.join(full_data_path, cla)
        subclass_path = os.listdir(class_path)
        for subcla in subclass_path:
            img_dir = os.path.join(class_path, subcla)
            if os.path.isdir(img_dir):
                imgs.append(img_dir)

    return imgs

def submission_dataloader(filepath):
    filepath = os.path.normcase(filepath)
    imgs = []
    full_data_path = os.path.join(filepath, 'full_data')
    train_class = ['stratified', 'training', 'test']

    for cla in train_class:
        class_path = os.path.join(full_data_path, cla)
        subclass_path = os.listdir(class_path)
        for subcla in subclass_path:
            img_dir = os.path.join(class_path, subcla)
            if os.path.isdir(img_dir):
                imgs.append(img_dir)

    return imgs
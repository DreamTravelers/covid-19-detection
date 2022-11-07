# -*- coding: utf-8 -*-
"""
@author: liulin
@time: 2021/4/6 13:52
@subscribe: 

"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, files, process, size):
        super(ImageDataset, self).__init__()

        with open(files, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split('&&&'), f))

        if process == 'train':
            self.transform = get_train_transform(size=size)
        else:
            self.transform = get_test_transform(size=size)

    def __getitem__(self, item):
        label, path, desc = self.imgs[item]
        if label.strip() == "Normal":
            label = 0
        elif label.strip() == "Pneumonia":
            label = 1
        else:
            label = 2

        home = os.path.expanduser('~')
        path = home + "/luolijuan/dataset/images/" + path.strip()
        img = Image.open(path).convert('RGB')
        img = img.resize((32, 32), Image.BILINEAR)
        img = self.transform(img)

        return img, torch.from_numpy(np.array(int(label))).float(),desc

    def __len__(self):
        return len(self.imgs)


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1 * self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img


def get_train_transform(size=0):
    train_transform = transforms.Compose([
        # Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    return train_transform


def get_test_transform(size=0):
    return transforms.Compose([
        # Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

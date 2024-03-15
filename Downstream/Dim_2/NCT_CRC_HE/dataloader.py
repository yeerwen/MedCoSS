import os
import pickle
import re
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import ImageFilter
import cv2
class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class NCT_CRC_HE_Dataset(data.Dataset):
    def __init__(self, data_path, split="train", crop_size=(224, 224)):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")
        # print("input size", crop_size)
        if split == "train":
            self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize(size=crop_size),
                        transforms.RandomApply(
                            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ]
                )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=crop_size),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )
        self.data_path = data_path
        self.classes = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}

        self.train_samples = []
        self.train_labels = []

        if split == "train":
            train_path = os.path.join(data_path, "NCT-CRC-HE-100K")
            for class_name, index in self.classes.items():
                train_sub_class_path = os.path.join(train_path, class_name)
                class_path_list = sorted(os.listdir(train_sub_class_path))
                class_path_list = [os.path.join(train_sub_class_path, name) for name in class_path_list]
                class_len = len(class_path_list)
                print(train_path, class_name, class_len)
                self.train_samples.extend(class_path_list[: int(class_len*0.8)])
                self.train_labels.extend([index]*int(class_len*0.8))
        elif split == "val":
            train_path = os.path.join(data_path, "NCT-CRC-HE-100K")
            for class_name, index in self.classes.items():
                train_sub_class_path = os.path.join(train_path, class_name)
                class_path_list = sorted(os.listdir(train_sub_class_path))
                class_path_list = [os.path.join(train_sub_class_path, name) for name in class_path_list]
                class_len = len(class_path_list)
                print(train_path, class_name, class_len)
                self.train_samples.extend(class_path_list[int(class_len * 0.8):])
                self.train_labels.extend([index]*int(class_len-int(class_len * 0.8)))
        elif split == "test":
            train_path = os.path.join(data_path, "CRC-VAL-HE-7K")
            for class_name, index in self.classes.items():
                train_sub_class_path = os.path.join(train_path, class_name)
                class_path_list = sorted(os.listdir(train_sub_class_path))
                class_path_list = [os.path.join(train_sub_class_path, name) for name in class_path_list]
                print(train_path, class_name, len(class_path_list))
                self.train_samples.extend(class_path_list)
                self.train_labels.extend([index]*len(class_path_list))

        print(split, f'dataset samples: {len(self.train_samples)}, labels: {len(self.train_labels)}')


    def __len__(self):
        return len(self.train_samples)


    def __getitem__(self, index):
        img_path = self.train_samples[index]
        label = float(self.train_labels[index])
        label = torch.tensor([label])
        image2D = cv2.imread(img_path)
        image2D = self.transform(image2D)
        return image2D, label


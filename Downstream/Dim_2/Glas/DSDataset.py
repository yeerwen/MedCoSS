import os
import os.path as osp
import numpy as np
import random
import cv2 as cv
from torch.utils import data
from PIL import ImageFilter, Image
import torchvision.transforms as transforms
import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Glas_Dataset(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(224, 224), scale=True,
                 mirror=True, ignore_label=255, ratio_labels=1, split="train"):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.ratio_labels = ratio_labels

        csv_lines = open(os.path.join(root, "Grade.csv"), "r").readlines()
        # print(csv_lines)
        self.img_ids = {"benign": [], "malignant": []}
        data_split = split if split != "val" else "train"
        for line in csv_lines:
            name, grade = line.split(",")[0].strip(), line.split(",")[2].strip()
            # print(name)
            if data_split in name:
                self.img_ids[grade].append(name)
        print(split, len(self.img_ids["benign"]), len(self.img_ids["malignant"]))

        print("Start preprocessing....")
        # self.img_ids = self.img_ids * 10
        self.files = []
        self.img_ids["benign"] = sorted(self.img_ids["benign"])
        self.img_ids["malignant"] = sorted(self.img_ids["malignant"])
        if split == "train":
            for item in self.img_ids["benign"][:int(0.8*len(self.img_ids["benign"]))]:
                self.files.append({
                    "image": os.path.join(self.root, "train", "images", item+".png"),
                    "name": item,
                    "label": os.path.join(self.root, "train", "labels", item+".png")
                })
            for item in self.img_ids["malignant"][:int(0.8*len(self.img_ids["malignant"]))]:
                self.files.append({
                    "image": os.path.join(self.root, "train", "images", item+".png"),
                    "name": item,
                    "label": os.path.join(self.root, "train", "labels", item+".png")
                })
        elif split == "val":
            for item in self.img_ids["benign"][int(0.8*len(self.img_ids["benign"])):]:
                self.files.append({
                    "image": os.path.join(self.root, "train", "images", item+".png"),
                    "name": item,
                    "label": os.path.join(self.root, "train", "labels", item+".png")
                })
            for item in self.img_ids["malignant"][int(0.8*len(self.img_ids["malignant"])):]:
                self.files.append({
                    "image": os.path.join(self.root, "train", "images", item+".png"),
                    "name": item,
                    "label": os.path.join(self.root, "train", "labels", item+".png")
                })

        elif split == "test":
            for item in self.img_ids["benign"]:
                # print(item)
                self.files.append({
                    "image": os.path.join(self.root, "test", "images", item + ".png"),
                    "name": item,
                    "label": os.path.join(self.root, "test", "labels", item + ".png")
                })
            for item in self.img_ids["malignant"]:
                # print(item)
                self.files.append({
                    "image": os.path.join(self.root, "test", "images", item + ".png"),
                    "name": item,
                    "label": os.path.join(self.root, "test", "labels", item + ".png")
                })

        if not max_iters == None and split == "train":
            print('{} images are loaded!'.format(len(self.files)))
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))
        else:
            for file in self.files:
                print(file["name"], end=" ")
            print()
        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        name =  datafiles["name"]
        # read npz file
        image2D =  cv.imread(datafiles["image"])
        label = cv.imread(datafiles["label"], flags=0)
        image2D = image2D.transpose(2, 0, 1).astype(np.float32)
        for i in range(image2D.shape[0]):
            image2D[i] = (image2D[i] - image2D[i].mean()) / image2D[i].std()
        label[label == 255] = 1
        label_one_hot = label[np.newaxis]
        return image2D.copy(), label_one_hot.copy(), name



def get_train_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1,
            data_key="image", label_key="label"
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2), data_key="image", label_key="label"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15, data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15, data_key="image"))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.15, data_key="image"))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def collate_fn_tr(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    # print(image.shape, label.shape)
    data_dict = {'image': image, 'label': label, 'name': name}
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    return data_dict


def collate_fn_ts(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'image': image, 'label': label, 'name': name}
    return data_dict
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

class QaTa_CoV19_Dataset(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(224, 224), scale=True,
                 mirror=True, ignore_label=255, ratio_labels=1, split="train"):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.ratio_labels = ratio_labels

        data_lines = open(os.path.join(root, split+".txt"), "r").readlines()
        print("Start preprocessing....", os.path.join(root, split+".txt"))

        if self.ratio_labels != 1 and split == "train":
            print("original number: ", len(data_lines), end=" ")
            data_lines = data_lines[:int(self.ratio_labels*len(data_lines))]
            print("now number: ", len(data_lines))
        # self.img_ids = self.img_ids * 10
        self.files = []

        if split == "train" or split == "val":
            sub_plane_name = "Train Set"
        elif split == "test":
            sub_plane_name = "Test Set"

        for name in data_lines:
            name = name.strip()
            self.files.append({
                "image": os.path.join(self.root, sub_plane_name, "Images", name),
                "name": name,
                "label": os.path.join(self.root, sub_plane_name, "Ground-truths", "mask_"+name)
            })

        if not max_iters == None and split == "train":
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

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
        assert np.max(label) == 1
        label_one_hot = label[np.newaxis]
        return image2D.copy(), label_one_hot.copy(), name



def get_train_transform(patch_size=(224, 224)):
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
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
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
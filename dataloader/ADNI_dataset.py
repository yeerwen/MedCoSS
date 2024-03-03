
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import os
import math
from torch.utils.data import Dataset
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform

class ADNI_dataset(Dataset):
    def __init__(self, data_path, crop_size=(16, 192, 192), is_sort=False):

        self.data_path = data_path
        self.global_crop3D_d, self.global_crop3D_h, self.global_crop3D_w = crop_size
        if not is_sort:
            self.img_ids = [i_id.strip().split() for i_id in open(os.path.join(data_path, "SSL_data_ADNI.txt"))]
        else:
            self.img_ids = sorted([i_id.strip().split() for i_id in open(os.path.join(data_path, "SSL_data_ADNI.txt"))])
            print("sorted dataset")

        self.files = []
        for nii_name in self.img_ids:
            img_file = os.path.join(self.data_path, nii_name[0])
            self.files.append({
                "img": img_file,
                "name": nii_name[0]
            })
        print('SSL: {} images are loaded!'.format(len(self.files)))

        self.transformer = MirrorTransform(axes=(0, 1, 2), data_key="image", label_key="label")

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(datafiles["img"])
        image = imageNII.get_fdata()
        image = image[np.newaxis, :][np.newaxis, :]
        image = image.transpose((0, 1, 4, 2, 3))
        data_dict = {'image': image, 'label': None}

        if random.randint(0, 1) == 0:
            data_dict = self.transformer(**data_dict)

        return data_dict["image"][0]



class ADNI_dataset_name(Dataset):
    def __init__(self, data_path, crop_size=(16, 192, 192), is_sort=False):

        self.data_path = data_path
        self.global_crop3D_d, self.global_crop3D_h, self.global_crop3D_w = crop_size
        if not is_sort:
            self.img_ids = [i_id.strip().split() for i_id in open(os.path.join(data_path, "SSL_data_ADNI.txt"))]
        else:
            self.img_ids = sorted([i_id.strip().split() for i_id in open(os.path.join(data_path, "SSL_data_ADNI.txt"))])
            print("sorted dataset")

        self.files = []
        for nii_name in self.img_ids:
            img_file = os.path.join(self.data_path, nii_name[0])
            self.files.append({
                "img": img_file,
                "name": nii_name[0]
            })
        print('SSL: {} images are loaded!'.format(len(self.files)))


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(datafiles["img"])
        image = imageNII.get_fdata()
        image = image[np.newaxis, :][np.newaxis, :]
        image = image.transpose((0, 1, 4, 2, 3))
        data_dict = {'image': image, 'label': None}
        return torch.from_numpy(data_dict["image"][0]).float(), datafiles["img"]
    
def my_collate_MR(batch):
    image3D, key = zip(*batch)
    image3D = torch.stack(image3D, 0)
    
    return [image3D, key]


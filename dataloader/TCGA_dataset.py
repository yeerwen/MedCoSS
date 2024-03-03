import os
import numpy as np
import torch
import torch.utils.data as data
import json
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

class TCGA_Image_Dataset(data.Dataset):
    def __init__(self, data_path, imsize=(224, 224), is_sort=False):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        #find all images
        self.image_path = []
        if os.path.exists(os.path.join(data_path, "pretrain_data_list.json")):
            self.image_path = load_json(os.path.join(data_path, "pretrain_data_list.json"))["path"]
        else:
            for root, dirs, files in tqdm(os.walk(data_path)):
                for file in files:
                    if ".jpg" in file:
                        self.image_path.append(os.path.join(root, file))
            data_list = {"path": self.image_path}
            save_json(data_list, os.path.join(data_path, "pretrain_data_list.json"))

        self.tr_transforms2D = get_train_transform2D(imsize)

        if is_sort:
            self.img_ids = sorted(self.image_path)
            print("sorted dataset")

        print("image sample number: ", len(self.image_path))

    def __len__(self):
        return len(self.image_path)


    def __getitem__(self, index):
        img_path = self.image_path[index]
        image2D = cv2.imread(img_path)
        image2D = Image.fromarray(image2D)
        image2D_trans = self.tr_transforms2D(image2D)


        return image2D_trans


class TCGA_Image_Dataset_name(data.Dataset):
    def __init__(self, data_path, imsize=(224, 224), is_sort=False):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        #find all images
        self.image_path = []
        if os.path.exists(os.path.join(data_path, "pretrain_data_list.json")):
            self.image_path = load_json(os.path.join(data_path, "pretrain_data_list.json"))["path"]
        else:
            for root, dirs, files in tqdm(os.walk(data_path)):
                for file in files:
                    if ".jpg" in file:
                        self.image_path.append(os.path.join(root, file))
            data_list = {"path": self.image_path}
            save_json(data_list, os.path.join(data_path, "pretrain_data_list.json"))

        self.tr_transforms2D = transforms.ToTensor()

        if is_sort:
            self.img_ids = sorted(self.image_path)
            print("sorted dataset")

        print("image sample number: ", len(self.image_path))

    def __len__(self):
        return len(self.image_path)


    def __getitem__(self, index):
        img_path = self.image_path[index]
        image2D = cv2.imread(img_path)
        image2D = Image.fromarray(image2D)
        image2D = self.tr_transforms2D(image2D)
        return image2D, img_path
    
def my_collate_path(batch):
    image2D, key = zip(*batch)
    image2D = torch.stack(image2D, 0)
    
    return [image2D, key]

def get_train_transform2D(crop_size):

    tr_transforms = transforms.Compose(
        [
        # transforms.ToPILImage(),
         transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0), interpolation=3),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()
         ])

    return tr_transforms






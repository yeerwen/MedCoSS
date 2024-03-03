import os
import pickle
import re
import shutil

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import json
import cv2
import pydicom
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import random

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def processing(path, out_path):
    img = Image.open(path).resize((224, 224))
    # print(os.path.join(out_path, path.split("/")[-1]))
    img.save(os.path.join(out_path, path.split("/")[-1]))


image_path = []
sub_dataset_name = ["TCGA-THYM", "TCGA-THCA", "TCGA-BRCA", "TCGA-UCEC", "TCGA-UVM", "TCGA-OV", "TCGA-MESO"]
root_path = "/data/userdisk0/ywye/Pretrained_dataset/2D/TCGA-Processed/"
out_path = "/erwen_SSD/2T/continual_pretraining/2D/TCGA/"
if os.path.exists(out_path):
    shutil.rmtree(out_path)
os.makedirs(out_path)
os.makedirs(os.path.join(out_path, "TCGA_resized"))
np.random.seed(0)

for sub_data in sub_dataset_name:
    dataset_path = os.path.join(root_path, sub_data)
    patient_list = os.listdir(dataset_path)
    print(sub_data, len(patient_list))
    # select_patient_list = np.random.choice(patient_list, patient_each_dataset, replace=False)
    for patient_name in patient_list:
        patient_path = os.path.join(dataset_path, patient_name)
        patient_slice_list = os.listdir(patient_path)
        # print(patient_name, len(patient_slice_list))
        select_patient = np.random.choice(patient_slice_list, min(100, len(patient_slice_list) ), replace=False)
        for name in select_patient:
            image_path.append(os.path.join(patient_path, name))
print(len(image_path))
print(len(np.unique(image_path))==len(image_path))

pool = Pool(processes=16, maxtasksperchild=1000)

for path in tqdm(image_path):

    pool.apply_async(func=processing, args=(path, os.path.join(out_path, "TCGA_resized")))

pool.close()
pool.join()
resize_list = os.listdir(os.path.join(out_path, "TCGA_resized"))
resize_list = [os.path.join(out_path, "TCGA_resized", path) for path in resize_list]
data_list = {"path": resize_list}
print("availabel data: ", len(resize_list))
save_json(data_list, os.path.join(out_path, "pretrain_data_list.json"))

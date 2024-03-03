
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
import random
from multiprocessing import Pool
def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)
        

def processing(path, out_path):
    img = Image.open(path).resize((224, 224))
    # print(os.path.join(out_path, path.split("/")[-1]))
    img.save(os.path.join(out_path, path.split("/")[-1]))

image_path = []
for root, dirs, files in tqdm(os.walk("/erwen_SSD/2T/continual_pretraining/2D/MIMIC-CXR-2.0-JPG/")):
    for file in files:
        if ".jpg" in file:
            image_path.append(os.path.join(root, file))
        if len(image_path) % 10000 == 0:
            print(len(image_path))
print("origion len", len(image_path))
image_path = list(set(image_path))
print("origion len", len(image_path))
#
print(len(image_path))
out_path = "/erwen_SSD/2T/continual_pretraining/2D/MIMIC-CXR-2.0-JPG/image_resize_224"
if os.path.exists(out_path):
    shutil.rmtree(out_path)
os.makedirs(out_path)
resize_path = []

pool = Pool(processes=16, maxtasksperchild=1000)

for path in tqdm(image_path):
    try:
        pool.apply_async(func=processing, args=(path, out_path))

    except:
        pass
pool.close()
pool.join()
resize_list = os.listdir(out_path)
resize_list = [os.path.join(out_path, path) for path in resize_list if path.endswith(".jpg")]
data_list = {"path": resize_list}
print("availabel data: ", len(resize_list))
save_json(data_list, "/erwen_SSD/2T/continual/2D/MIMIC-CXR-2.0-JPG/pretrain_data_list.json")


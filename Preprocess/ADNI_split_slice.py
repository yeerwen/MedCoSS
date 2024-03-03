import shutil

import numpy as np
import os
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk
from multiprocessing import Pool
from math import ceil

ori_path = '/data/userdisk0/ywye/Pretrained_dataset/3D/MRI/ADNI_nii_v0_resize'
save_path = '/erwen_SSD/2T/contiual_pretraining/3D/ADNI/DL_patches_v2'
file_path = '/erwen_SSD/2T/contiual_pretraining/3D/ADNI/SSL_data_ADNI.txt'

if os.path.exists(save_path):
    shutil.rmtree(save_path)
def processing(root, i_files):
    img_path = os.path.join(root, i_files)
    imageITK = sitk.ReadImage(img_path)
    image = sitk.GetArrayFromImage(imageITK)
    ori_spacing = np.array(imageITK.GetSpacing())[[2, 1, 0]]
    ori_origin = imageITK.GetOrigin()
    ori_direction = imageITK.GetDirection()

    # print(image.shape)
    d, w, h = image.shape
    image = image[int(0.1*d):int(0.9*d)]
    d, w, h = image.shape
    tile_size = 16
    # strideD = ceil(tile_size * (1 - overlap))
    strideD = 8 #4 12
    tile_deps = int(max(0, ceil((d - tile_size) / strideD)) + 1)
    # print("strideD is %d" % (strideD))
    print("tile_deps is %d" % (tile_deps))
    for dep in range(tile_deps):
        path_dep = os.path.join(save_path, i_files[:-4]+'_dep'+str(dep)+'.nii.gz')
        if os.path.isfile(path_dep):
            continue
        else:
            d1 = int(dep * strideD)
            d2 = min(d1 + tile_size, d)
            if d2-d1 < tile_size:
                d1 = d2-tile_size

            if w > 320:
                img = image[np.maximum(d1, 0):d2, int(w * 0.1):int(w * 0.9), int(h * 0.1):int(h * 0.9)]
            else:
                img = image[np.maximum(d1, 0):d2]

            # img = img.astype(np.int16)
            print(img.shape, dep, path_dep)
            lower = np.percentile(img, 0.2)
            upper = np.percentile(img, 99.8)

            img[img < lower] = lower
            img[img > upper] = upper

            img = (img - img.mean()) / img.std()
            assert img.shape == (16, 192, 192)
            #save it
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            saveITK = sitk.GetImageFromArray(img)
            saveITK.SetSpacing(ori_spacing[[2, 1, 0]])
            saveITK.SetOrigin(ori_origin)
            saveITK.SetDirection(ori_direction)
            sitk.WriteImage(saveITK, path_dep)


count = -1

pool = Pool(processes=16, maxtasksperchild=1000)
for root, dirs, files in os.walk(ori_path):
    for i_files in tqdm(sorted(files)):
        if i_files[0]=='.':
            continue

        # read img
        # print("Processing %s" % (i_files))

        pool.apply_async(func=processing, args=(root, i_files))

pool.close()
pool.join()

file = open(file_path, "w")
for name in sorted(os.listdir(save_path)):
    if name.endswith("nii.gz"):
        file.write("DL_patches_v2/"+name+"\n")
file.close()
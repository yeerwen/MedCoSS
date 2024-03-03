import numpy as np
import os
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk
from multiprocessing import Pool
from math import ceil

ori_path = '../../Images_nifti_spacing'
save_path = '../../DL_patches_v2'
file_path = '../../SSL_data_deeplesion.txt'

def processing(root, i_files):
    img_path = os.path.join(root, i_files)
    imageITK = sitk.ReadImage(img_path)
    image = sitk.GetArrayFromImage(imageITK)
    ori_spacing = np.array(imageITK.GetSpacing())[[2, 1, 0]]
    ori_origin = imageITK.GetOrigin()
    ori_direction = imageITK.GetDirection()

    # print(image.shape)
    d, w, h = image.shape
    tile_size = 24
    # strideD = ceil(tile_size * (1 - overlap))
    strideD = 12  #4 12
    tile_deps = int(ceil((d - tile_size) / strideD) + 1)
    print("strideD is %d" % (strideD))
    print("tile_deps is %d" % (tile_deps))
    for dep in tqdm(range(tile_deps)):
        path_dep = os.path.join(save_path, i_files[:-7]+'_dep'+str(dep)+i_files[-7:-3])
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

            img = img.astype(np.int16)
            print(img.shape, dep, path_dep)

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
        print("Processing %s" % (i_files))

        pool.apply_async(func=processing, args=(root, i_files))

pool.close()
pool.join()

file = open(file_path, "w")
for name in sorted(os.listdir(save_path)):
    if name.endswith("nii.gz"):
        file.write("DL_patches_v2/"+name+"\n")
file.close()
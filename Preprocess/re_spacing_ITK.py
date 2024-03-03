import numpy as np
import os
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk
from multiprocessing import Pool

def truncate(CT):
    # truncate
    min_HU = -1024
    max_HU = 1024
    CT[np.where(CT <= min_HU)] = min_HU
    CT[np.where(CT >= max_HU)] = max_HU
    # CT = CT - 158.58
    # CT = CT / 324.70
    return CT

# rate = 1.5

spacing = {
    0: [1.0, 1.0, 1.0],
}

ori_path = '../Images_nifti'
save_path = '../Images_nifti_spacing'

count = -1


def processing(root, i_files):
    img_path = os.path.join(root, i_files)
    imageITK = sitk.ReadImage(img_path)
    image = sitk.GetArrayFromImage(imageITK)
    ori_size = np.array(imageITK.GetSize())[[2, 1, 0]]
    ori_spacing = np.array(imageITK.GetSpacing())[[2, 1, 0]]
    ori_origin = imageITK.GetOrigin()
    ori_direction = imageITK.GetDirection()

    task_id = 0
    target_spacing = np.array(spacing[task_id])
    spc_ratio = ori_spacing / target_spacing
    spc_ratio = np.round(spc_ratio, 4)

    data_type = image.dtype
    order = 3

    image = image.astype(np.float)
    image = truncate(image)

    image_resize = resize(image, (
    int(np.round(ori_size[0] * spc_ratio[0])), int(ori_size[1] * spc_ratio[1]), int(ori_size[2] * spc_ratio[2])),
                          order=order, cval=0, clip=True, preserve_range=True)
    image_resize = np.round(image_resize).astype(data_type)

    # save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saveITK = sitk.GetImageFromArray(image_resize)
    saveITK.SetSpacing(target_spacing[[2, 1, 0]])
    saveITK.SetOrigin(ori_origin)
    saveITK.SetDirection(ori_direction)
    sitk.WriteImage(saveITK, os.path.join(save_path, i_files))



pool = Pool(processes=8, maxtasksperchild=1000)

for root, dirs, files in os.walk(ori_path):
    for i_files in tqdm(sorted(files)):
        if i_files[0]=='.':
            continue

        if os.path.isfile(os.path.join(save_path, i_files)):
            continue

        # read img
        print("Processing %s" % (i_files))

        pool.apply_async(func=processing, args=(root, i_files))

pool.close()
pool.join()

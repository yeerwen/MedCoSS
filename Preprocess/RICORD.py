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
    return CT

ori_path = '/media/userdisk0/ywye/Contiual_learning/3D/RICORD/RICORD_nii'
save_path = '/media/userdisk0/ywye/Contiual_learning/3D/RICORD/RICORD_nii_resize'
sub_path = ["MIDRC-RICORD-1A", "MIDRC-RICORD-1B"]
count = -1

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkLinear):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

def processing(sub_path_name, i_files):
    img_path = os.path.join(ori_path, sub_path_name, i_files)
    imageITK = sitk.ReadImage(img_path)
    imageITK = resize_image_itk(imageITK, (192, 192, 64))
    target_spacing = imageITK.GetSpacing()
    ori_origin = imageITK.GetOrigin()
    ori_direction = imageITK.GetDirection()
    image = sitk.GetArrayFromImage(imageITK)

    # save
    if not os.path.exists(os.path.join(save_path, sub_path_name)):
        os.makedirs(os.path.join(save_path, sub_path_name))
    saveITK = sitk.GetImageFromArray(image)
    saveITK.SetSpacing(target_spacing)
    saveITK.SetOrigin(ori_origin)
    saveITK.SetDirection(ori_direction)
    sitk.WriteImage(saveITK, os.path.join(save_path, sub_path_name, i_files))



pool = Pool(processes=16, maxtasksperchild=1000)

for sub_path_name in sub_path:
    for name in os.listdir(os.path.join(ori_path, sub_path_name)):

        if name.endswith("nii.gz"):
            # read img
            print("Processing %s" % (name))
            pool.apply_async(func=processing, args=(sub_path_name, name))

pool.close()
pool.join()

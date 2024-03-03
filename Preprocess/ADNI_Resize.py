import numpy as np
import os
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk
from multiprocessing import Pool


ori_path = '/data/userdisk0/ywye/Pretrained_dataset/3D/MRI/ADNI_nii_v0/'
save_path = '/data/userdisk0/ywye/Pretrained_dataset/3D/MRI/ADNI_nii_v0_resize'

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

def processing(root, i_files):
    img_path = os.path.join(root, i_files)
    imageITK = sitk.ReadImage(img_path)
    depth = imageITK.GetSize()[-1]
    imageITK = resize_image_itk(imageITK, (192, 192, depth))
    target_spacing = imageITK.GetSpacing()
    ori_origin = imageITK.GetOrigin()
    ori_direction = imageITK.GetDirection()
    image = sitk.GetArrayFromImage(imageITK)

    # save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saveITK = sitk.GetImageFromArray(image)
    saveITK.SetSpacing(target_spacing)
    saveITK.SetOrigin(ori_origin)
    saveITK.SetDirection(ori_direction)
    sitk.WriteImage(saveITK, os.path.join(save_path, i_files))



pool = Pool(processes=8, maxtasksperchild=1000)

for root, dirs, files in os.walk(ori_path):
    for i_files in tqdm(sorted(files)):
        if i_files[0]=='.':
            continue

        pool.apply_async(func=processing, args=(root, i_files))

pool.close()
pool.join()


slice_sum, sample_sum = 0, 0
for root, dirs, files in os.walk(ori_path):
    for i_files in tqdm(sorted(files)):
        if i_files[0]=='.':
            continue
        img_path = os.path.join(root, i_files)
        imageITK = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(imageITK)
        slice_sum += image.shape[0]
        sample_sum += 1
print(f"slice sum: {slice_sum}, sample sum: {sample_sum}")
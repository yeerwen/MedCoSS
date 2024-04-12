# MedCoSS
This is the official Pytorch implementation of our CVPR 2024 paper (Highlight) "[Continual Self-supervised Learning: Towards Universal Multi-modal Medical Data Representation Learning](https://arxiv.org/abs/2311.17597)".

<div align="center">
  <img width="100%" alt="MedCoSS illustration" src="github/Overview.png">
</div>

## Requirements
CUDA 11.5<br />
Python 3.8<br /> 
Pytorch 1.11.0<br />
CuDNN 8.3.2.44

### Data Preparation
* Pre-training data
  * Download the [MIMIC-CXR dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
  * Download the [DeepLesion dataset](https://nihcc.app.box.com/v/DeepLesion).
  * Download the [ADNI dataset](https://adni.loni.usc.edu/).
  * Download [seven TCGA datasets](https://portal.gdc.cancer.gov/) (TCGA-THYM, TCGA-THCA, TCGA-BRCA, TCGA-UCEC, TCGA-UVM, TCGA-OV, and TCGA-MESO).
* Fine-tuning data
  * PudMed20k dataset: Download the [PudMed20k dataset](https://github.com/Franck-Dernoncourt/pubmed-rct/tree/master).
  * ChestXR dataset: Download the [ChestXR dataset](https://cxr-covid19.grand-challenge.org/Download/).
  * QaTav2 dataset: Download the [QaTav2 dataset](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset).
  * RICORD dataset: Download the [MIDRC-RICORD-1A dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969742#80969742171ba531fc374829b21d3647e95f532c) and [MIDRC-RICORD-1B dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969771).
    The folder structure of the dataset should be like

        dataset/RICORD_nii/
          ├── MIDRC-RICORD-1A
          ├── MIDRC-RICORD-1B
  * LiTS dataset: Download the [LiTS dataset](https://competitions.codalab.org/competitions/17094).
  * VS dataset: Download the [VS dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053).
  * LA dataset: Download the [LA dataset](https://github.com/yulequan/UA-MT/tree/master).
  * NCH dataset: Download the [NCT-CRC-HE-100K and CRC-VAL-HE-7K datasets](https://zenodo.org/records/1214456#.YwH6HOpBy5f).
  * GlaS dataset: Download the [GlaS dataset](https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation).


### Pre-processing
* Pre-training data
  * Report: Following [MGCA's procedure](https://github.com/HKU-MedAI/MGCA/blob/main/mgca/preprocess/mimic_cxr.py) to pre-process the MIMIC-CXR dataset.
  * X-ray: Using `Preprocess/MIMIC_CXR_JPG_Preprocess.py` to pre-process the MIMC-CXR dataset.
  * CT: 
    * Using `Preprocess/DL_save_nifti.py` (from downloaded files) to transfer the PNG image to the nii.gz form.
    * Using `Preprocess/re_spacing_ITK.py` to resample CT volumes.
    * Using `Preprocess/splitting_to_patches.py` to extract about 125k sub-volumes, and the pre-processed dataset will be saved in `DL_patches_v2/`.
    * Using `Preprocess/DeepLesion_Resize.py` to resize images.
  * MRI: 
    * Using `Preprocess/ADNI_Resize.py` to resize images.
    * Using `Preprocess/ADNI_split_slice.py` to extract about 59k sub-volumes.
  * Pathological imaging: Using `Preprocess/TCGA_Preprocess.py` to pre-process seven TCGA datasets.
* Fine-tuning data
  * PudMed20k dataset: None.
  * ChestXR dataset: None.
  * QaTav2 dataset: Using `Preprocess/QaTav2.py` to pre-process.
  * RICORD dataset: Using `Preprocess/RICORD.py` to pre-process. Data Splits can be obtained from `/Downstream/Dim_3/RICORD/data_split`.
  * LiTS dataset: 
    * (1) Resampling all data to the same spacing of 1.5mm × 0.8mm × 0.8mm; 
    * (2) Using the [nnUNet v1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) framework to pre-process.
  * VS dataset: 
    * (1) Run [Convert_VSseg_to_nnUNet_dataset.py](https://github.com/yeerwen/UniSeg/blob/main/Downstream/Convert_VSseg_to_nnUNet_dataset.py);
    * (2) Using the [nnUNet v1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) framework to pre-process;
    * (3) Using `Preprocess/VSeg.py` to pre-process.
  * LA dataset: None.
  * NCH dataset: None.
  * GlaS dataset: Using `Preprocess/GlaS.py` to pre-process.


### Pre-training
* Download [uni-perceiver-base-L12-H768-224size-torch-pretrained.pth](https://github.com/fundamentalvision/Uni-Perceiver/blob/main/data/checkpoints.md).
* Run `sh run_ssl.sh` for pre-training (4 GPUs with 24G. Before running it, you need to modify some addresses.)

### Pre-trained Model
* Pre-trained model is available in [MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05](https://drive.google.com/drive/folders/15RlO7l8njzqCt4ccXIwBIjPXZo94Z118?usp=sharing).

### Fine-tuning
* Run `sh run_ds.sh` for fine-tuning. (one GPU with 11G. Before running it, you need to modify some addresses.)


## To do
- [x] Dataset Links
- [x] Pre-processing Code
- [x] Pre-training Code Release
- [x] Pre-trained Model
- [x] Fine-tuning Code Release
- [ ] Continual pre-training on new data

## Citation
If this code is helpful for your study, please cite:

```
@article{ye2024medcoss,
  title={Continual Self-supervised Learning: Towards Universal Multi-modal Medical Data Representation Learning},
  author={Ye, Yiwen and Xie, Yutong and Zhang, Jianpeng and Chen, Ziyang and Wu, Qi and Xia, Yong},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={},
  year={2024},
}
```

## Acknowledgements
The whole framework is based on [MAE](https://github.com/facebookresearch/mae), [Uni-Perceiver](https://github.com/fundamentalvision/Uni-Perceiver), and [MGCA](https://github.com/HKU-MedAI/MGCA/tree/main).

## Contact
Yiwen Ye (ywye@mail.nwpu.edu.cn)

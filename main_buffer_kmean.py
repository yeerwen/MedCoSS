# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import shutil

import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from dataloader.mimic_cxr_report_dataset import MIMIC_CXR_Report_Dataset_name, my_collate_text
from dataloader.mimic_cxr_image_dataset import MIMIC_CXR_Image_Dataset_name, my_collate_xray
from dataloader.DeepLesion_dataset import DeepLesion_dataset_name, my_collate_CT
from dataloader.ADNI_dataset import ADNI_dataset_name, my_collate_MR
from dataloader.TCGA_dataset import TCGA_Image_Dataset_name, my_collate_path
from model.Unimodel import Unified_Model
from typing import Iterable

import tqdm
import pandas as pd
from sklearn.cluster import KMeans
import json

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='unified_vit', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--mask_ratio', default=0.00, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    # task 1D 2D 3D
    parser.add_argument('--task_modality', default="1D_text",
                        choices=["1D_text", "2D_xray", "3D_CT", "3D_MR", "2D_path"], type=str,
                        help='pre-training data')
    parser.add_argument('--load_current_pretrained_weight', default="", type=str)
    parser.add_argument('--num_center', type=float, required=True)
    parser.add_argument('--buffer_ratio', type=float, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    return parser


def main(args):

    os.environ["OMP_NUM_THREADS"] = "1"
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # no augmentation   
    if args.task_modality == "1D_text":
        dataset_train = MIMIC_CXR_Report_Dataset_name(data_path=args.data_path, split="train", transform=None, max_words=112)
        my_collate = my_collate_text
    elif args.task_modality == "2D_xray":
        dataset_train = MIMIC_CXR_Image_Dataset_name(data_path=args.data_path, imsize=args.input_size)
        my_collate = my_collate_xray
    elif args.task_modality == "3D_CT":
        dataset_train = DeepLesion_dataset_name(data_path=args.data_path, crop_size=(16, 192, 192))
        my_collate = my_collate_CT
    elif args.task_modality == "3D_MR":
        dataset_train = ADNI_dataset_name(data_path=args.data_path, crop_size=(16, 192, 192))
        my_collate = my_collate_MR
    elif args.task_modality == "2D_path":
        dataset_train = TCGA_Image_Dataset_name(data_path=args.data_path, imsize=args.input_size)
        my_collate = my_collate_path
    else:
        exit()


    model = Unified_Model(now_1D_input_size=(112,  1), now_2D_input_size=(224, 224), now_3D_input_size=(16, 192, 192), norm_pix_loss=args.norm_pix_loss)
    print("load pretrained parameter from ", args.load_current_pretrained_weight)
    pretrained_weight = torch.load(args.load_current_pretrained_weight, map_location='cpu')
    model.load_state_dict(pretrained_weight["model"], strict=False)
    print("load pre-trained model success!")
    model.to(device)

   
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=128,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False, collate_fn=my_collate
    )
    task_list = ["1D_text", "2D_xray", "3D_CT", "3D_MR", "2D_path"]
    for task_index, task_name in enumerate(task_list):
        if task_name == args.task_modality:
            task_id = task_index
            break
    save_path = args.load_current_pretrained_weight.split("/checkpoint")[0]
    estimate_kmean(save_path, task_id, model, data_loader_train, device=device, args=args, center_num=args.num_center, buffer_ratio=args.buffer_ratio, exp_name=args.exp_name)



def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def estimate_kmean(save_path, task_id, model: torch.nn.Module,
                    data_loader: Iterable,
                    device: torch.device,
                    args=None, buffer_ratio=0.01, center_num=2250, exp_name=None):
    print("task id:", task_id)
    print("save path", save_path)
    img_feature, img_path_list, image_loss = [], [], []
    for data_iter_step, samples in tqdm.tqdm(enumerate(data_loader)):
        img_path = samples[-1]

        if args.task_modality == "1D_text":
            samples = {"data": samples[0].to(device, non_blocking=True), "text_labels": samples[1].to(device, non_blocking=True),
                    "mask_attention": samples[2].to(device, non_blocking=True), "modality": "text", "task": "1D_text"}
        elif args.task_modality == "2D_xray":
            samples = {"data": samples[0].to(device, non_blocking=True), "modality": "2D image", "task": "2D_xray"}
        elif args.task_modality == "3D_CT":
            samples = {"data": samples[0].float().to(device, non_blocking=True), "modality": "3D image", "task": "3D_CT"}
        elif args.task_modality == "3D_MR":
            samples = {"data": samples[0].float().to(device, non_blocking=True), "modality": "3D image", "task": "3D_MR"}
        elif args.task_modality == "2D_path":
            samples = {"data": samples[0].to(device, non_blocking=True), "modality": "2D image", "task": "2D_path"}


        with torch.cuda.amp.autocast():
            with torch.no_grad():
                feature, _ = model(samples.copy(), mask_ratio=0.0, feature=True)
                feature = feature.mean(1)

        img_feature.append(feature.cpu().detach().numpy())
        img_path_list.append(img_path)

        # test only
        # if len(img_feature) == 20:
        #     break
        

    assert len(img_feature) == len(img_path_list)
    if args.exp_name == "kmean":
        img_feature = np.concatenate(img_feature, axis=0)
        img_path_list = np.concatenate(img_path_list, axis=0)
        print(img_feature.shape, img_path_list.shape)
        num_clusters = int(img_feature.shape[0] * center_num)
        sample_num_each_clusters = int(img_feature.shape[0] * buffer_ratio // num_clusters)
        print("center number: {}, sample_number_each_center: {}".format(num_clusters, sample_num_each_clusters))

        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(img_feature)
  
        distances_to_cluster_centers = np.linalg.norm(img_feature - kmeans.cluster_centers_[kmeans.labels_], axis=1)
        # print(img_feature.shape, kmeans.cluster_centers_[kmeans.labels_].shape, distances_to_cluster_centers.shape)

        def get_top_k_sample_names_for_each_center(k):
            closest_sample_names = {}
            
            for i in range(kmeans.n_clusters):
                cluster_distances = distances_to_cluster_centers[kmeans.labels_ == i]
                cluster_indices = np.where(kmeans.labels_ == i)[0]
                top_k_indices = cluster_distances.argsort()[:k]
                top_k_names = [img_path_list[idx] for idx in cluster_indices[top_k_indices]]
                # print([distances_to_cluster_centers[idx] for idx in cluster_indices[top_k_indices]])
                closest_sample_names[i] = top_k_names
            
            return closest_sample_names

        top_k_samples_name = get_top_k_sample_names_for_each_center(sample_num_each_clusters)

    else:
        exit()

    
    if args.task_modality == "1D_text":
        file_path = f'{args.task_modality}_{center_num}_{buffer_ratio}_{exp_name}.csv'

        if os.path.exists(os.path.join(save_path, file_path)):
            os.remove(os.path.join(save_path, file_path))
        
        df = pd.read_csv(os.path.join(args.data_path, "master.csv"))
        for itr in tqdm.tqdm(range(num_clusters)):
            top_k_name = top_k_samples_name[itr]
            top_k_name = [spare_name.replace("/data/userdisk0/ywye/Pretrained_dataset/1D/2019.MIMIC-CXR-JPG/", "") for spare_name in top_k_name] #Need to modify
            filtered_df = df[df['Path'].isin(top_k_name)]
            assert len(top_k_name) == len(filtered_df)
            write_header = not pd.io.common.file_exists(os.path.join(save_path, file_path))
            filtered_df.to_csv(os.path.join(save_path, file_path), index=False, mode='a', header=write_header)
        
    elif args.task_modality == "2D_xray":
        file_path = f'{args.task_modality}_{center_num}_{buffer_ratio}_{exp_name}.json'
        if os.path.exists(os.path.join(save_path, file_path)):
            os.remove(os.path.join(save_path, file_path))
        data_list = {"path": []}
        for itr in tqdm.tqdm(range(num_clusters)):
            top_k_name = top_k_samples_name[itr]

            data_list["path"].extend(top_k_name)
        print("availabel data: ", len(data_list["path"]))
        save_json(data_list, os.path.join(save_path, file_path))
            
    elif args.task_modality == "3D_CT":
        file_path = f'{args.task_modality}_{center_num}_{buffer_ratio}_{exp_name}.txt'
        if os.path.exists(os.path.join(save_path, file_path)):
            os.remove(os.path.join(save_path, file_path))
        with open(os.path.join(save_path, file_path), "w") as fp:

            for itr in tqdm.tqdm(range(num_clusters)):
                top_k_name = top_k_samples_name[itr]
                for name in top_k_name:
                    fp.write(name.replace("/data1/ywye/continual_pretrainingl/3D/DeepLesion/DL_patches_v2_resize/", "DL_patches_v2/")+"\n") #Need to modify

    elif args.task_modality == "3D_MR":
        file_path = f'{args.task_modality}_{center_num}_{buffer_ratio}_{exp_name}.txt'

        if os.path.exists(os.path.join(save_path, file_path)):
            os.remove(os.path.join(save_path, file_path))

        with open(os.path.join(save_path, file_path), "w") as fp:

            for itr in tqdm.tqdm(range(num_clusters)):
                top_k_name = top_k_samples_name[itr]
                for name in top_k_name:
                    fp.write(name.replace("/data1/ywye/continual_pretraining/3D/ADNI/", "")+"\n") #Need to modify

    elif args.task_modality == "2D_path":
        file_path = f'{args.task_modality}_{center_num}_{buffer_ratio}_{exp_name}.json'
        if os.path.exists(os.path.join(save_path, file_path)):
            os.remove(os.path.join(save_path, file_path))
        data_list = {"path": []}
        for itr in tqdm.tqdm(range(num_clusters)):
            top_k_name = top_k_samples_name[itr]
            data_list["path"].extend(top_k_name)
        print("availabel data: ", len(data_list["path"]))

        save_json(data_list, os.path.join(save_path, file_path))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

#!/bin/bash
output_dir="./output_dir/1D_text_300"
CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' main_pretrain_single_modal.py \
--model "unified_vit" \
--batch_size 128 \
--num_workers 10 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 300 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
--data_path "/data/userdisk0/ywye/Pretrained_dataset/1D/2019.MIMIC-CXR-JPG/2.0.0/" \
--task_modality "1D_text" \
--load_current_pretrained_weight "/data/userdisk0/ywye/Pretrained_dataset/pretrained_weigth/uni-perceiver-base-L12-H768-224size-torch-pretrained.pth" \
--output_dir=$output_dir \
--log_dir=$output_dir

CUDA_VISIBLE_DEVICES=3 python main_buffer_kmean.py \
--model "unified_vit" \
--num_workers 10 \
--norm_pix_loss \
--data_path "/data/userdisk0/ywye/Pretrained_dataset/1D/2019.MIMIC-CXR-JPG/2.0.0/" \
--task_modality "1D_text" \
--load_current_pretrained_weight "./output_dir/1D_text_300/checkpoint-299.pth" \
--num_center 0.01 \
--buffer_ratio 0.05 \
--exp_name "kmean"



output_dir="./output_dir/MedCoSS_Report_Xray_CT_MR_Path_buff_0.05_cen_0.01_2D_Xray_300"

CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29361' main_pretrain_medcoss.py \
--model "unified_vit" \
--batch_size 128 \
--num_workers 10 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 300 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
--task_modality "2D_xray" \
--load_current_pretrained_weight "./output_dir/1D_text_300/checkpoint-299.pth" \
--data_path_1D_text "/data/userdisk0/ywye/Pretrained_dataset/1D/2019.MIMIC-CXR-JPG/2.0.0/" \
--data_path_2D_xray "/data1/ywye/continual_pretraining/2D/MIMIC-CXR-2.0-JPG/" \
--output_dir=$output_dir \
--log_dir=$output_dir \
--num_center 0.01 \
--buffer_ratio 0.05 \
--exp_name "kmean" \
--mix_up 1


CUDA_VISIBLE_DEVICES=3 python main_buffer_kmean.py \
--model "unified_vit" \
--num_workers 10 \
--norm_pix_loss \
--data_path "/data1/ywye/continual_pretraining/2D/MIMIC-CXR-2.0-JPG/" \
--task_modality "2D_xray" \
--load_current_pretrained_weight "./output_dir/MedCoSS_Report_Xray_CT_MR_Path_buff_0.05_cen_0.01_2D_Xray_300/checkpoint-299.pth" \
--num_center 0.01 \
--buffer_ratio 0.05 \
--exp_name "kmean"


output_dir="./output_dir/MedCoSS_Report_Xray_CT_MR_Path_buff_0.05_cen_0.01_3D_CT_300"

CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29361' main_pretrain_medcoss.py \
--model "unified_vit" \
--batch_size 128 \
--num_workers 10 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 300 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
--task_modality "3D_CT" \
--load_current_pretrained_weight "./output_dir/MedCoSS_Report_Xray_CT_MR_Path_buff_0.05_cen_0.01_2D_Xray_300/checkpoint-299.pth" \
--data_path_1D_text "/data/userdisk0/ywye/Pretrained_dataset/1D/2019.MIMIC-CXR-JPG/2.0.0/" \
--data_path_2D_xray "/data1/ywye/continual_pretraining/2D/MIMIC-CXR-2.0-JPG/" \
--data_path_3D_CT "/data1/ywye/continual_pretraining/3D/DeepLesion/" \
--output_dir=$output_dir \
--log_dir=$output_dir \
--num_center 0.01 \
--buffer_ratio 0.05 \
--exp_name "kmean" \
--mix_up 1


CUDA_VISIBLE_DEVICES=3 python main_buffer_kmean.py \
--model "unified_vit" \
--num_workers 10 \
--norm_pix_loss \
--data_path "/data1/ywye/ccontinual_pretraining/3D/DeepLesion/" \
--task_modality "3D_CT" \
--load_current_pretrained_weight "./output_dir/MedCoSS_Report_Xray_CT_MR_Path_buff_0.05_cen_0.01_3D_CT_300/checkpoint-299.pth" \
--num_center 0.01 \
--buffer_ratio 0.05 \
--exp_name "kmean"



output_dir="./output_dir/MedCoSS_Report_Xray_CT_MR_Path_buff_0.05_cen_0.01_3D_MR_300"

CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29361' main_pretrain_medcoss.py \
--model "unified_vit" \
--batch_size 128 \
--num_workers 10 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 300 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
--task_modality "3D_MR" \
--load_current_pretrained_weight "./output_dir/MedCoSS_Report_Xray_CT_MR_Path_buff_0.05_cen_0.01_3D_CT_300/checkpoint-299.pth" \
--data_path_1D_text "/data/userdisk0/ywye/Pretrained_dataset/1D/2019.MIMIC-CXR-JPG/2.0.0/" \
--data_path_2D_xray "/data1/ywye/continual_pretraining/2D/MIMIC-CXR-2.0-JPG/" \
--data_path_3D_CT "/data1/ywye/continual_pretraining/3D/DeepLesion/" \
--data_path_3D_MR "/data1/ywye/continual_pretraining/3D/ADNI/" \
--output_dir=$output_dir \
--log_dir=$output_dir \
--num_center 0.01 \
--buffer_ratio 0.05 \
--exp_name "kmean" \
--mix_up 1


CUDA_VISIBLE_DEVICES=3 python main_buffer_kmean.py \
--model "unified_vit" \
--num_workers 10 \
--norm_pix_loss \
--data_path "/data1/ywye/continual_pretraining/3D/ADNI/" \
--task_modality "3D_MR" \
--load_current_pretrained_weight "./output_dir/MedCoSS_Report_Xray_CT_MR_Path_buff_0.05_cen_0.01_3D_MR_300/checkpoint-299.pth" \
--num_center 0.01 \
--buffer_ratio 0.05 \
--exp_name "kmean"



output_dir="./output_dir/MedCoSS_Report_Xray_CT_MR_Path_buff_0.05_cen_0.01_2D_Path_300"

CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29361' main_pretrain_medcoss.py \
--model "unified_vit" \
--batch_size 128 \
--num_workers 10 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 300 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
--task_modality "2D_path" \
--load_current_pretrained_weight "./output_dir/MedCoSS_Report_Xray_CT_MR_Path_buff_0.05_cen_0.01_3D_MR_300/checkpoint-299.pth" \
--data_path_1D_text "/data/userdisk0/ywye/Pretrained_dataset/1D/2019.MIMIC-CXR-JPG/2.0.0/" \
--data_path_2D_xray "/data1/ywye/continual_pretraining/2D/MIMIC-CXR-2.0-JPG/" \
--data_path_3D_CT "/data1/ywye/continual_pretraining/3D/DeepLesion/" \
--data_path_3D_MR "/data1/ywye/continual_pretraining/3D/ADNI/" \
--data_path_2D_path "/data1/ywye/continual_pretraining/2D/TCGA/" \
--output_dir=$output_dir \
--log_dir=$output_dir \
--num_center 0.01 \
--buffer_ratio 0.05 \
--exp_name "kmean" \
--mix_up 1
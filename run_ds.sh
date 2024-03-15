#!/bin/bash

gpu_id=1

task_id='1D_PubMed'

reload_from_pretrained=True
pretrained_path='/media/userdisk0/ywye/MAE/Pretrained_Model/MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05/checkpoint-299.pth'
exp_name='MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05'
model_name='model'
data_path='/media/userdisk1/ywye/Continual_learning/1D/pubmed-rct/PubMed_20k_RCT/'

lr=0.0002


seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_1/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_1/PudMed20k/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size=112 \
--batch_size=64 \
--num_gpus=1 \
--num_epochs=5 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=5 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed \
--model_name=$model_name 


seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_1/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_1/PudMed20k/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size=112 \
--batch_size=64 \
--num_gpus=1 \
--num_epochs=5 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=5 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed \
--model_name=$model_name 


seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_1/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_1/PudMed20k/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size=112 \
--batch_size=64 \
--num_gpus=1 \
--num_epochs=5 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=5 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed \
--model_name=$model_name 



########################################################################################################################################

gpu_id=1

task_id='2D_ChestXR'


reload_from_pretrained=True
pretrained_path='/media/userdisk0/ywye/MAE/Pretrained_Model/MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05/checkpoint-299.pth'
exp_name='MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05'


data_path='/media/erwen_SSD/Continual_learning/2D/Chest_XR/'
lr=0.00005



seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_2/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Chest_XR/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='224,224' \
--batch_size=32 \
--num_gpus=1 \
--num_epochs=80 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=3 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed 

seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_2/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Chest_XR/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='224,224' \
--batch_size=32 \
--num_gpus=1 \
--num_epochs=80 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=3 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed 


seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_2/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Chest_XR/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='224,224' \
--batch_size=32 \
--num_gpus=1 \
--num_epochs=80 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=3 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed 


# ################################################################################################

nnudata='/media/erwen_SSD/Continual_learning/2D/QaTa_CoV19/QaTa-COV19-v2/'

gpu_id=1

task_id='QaTa_CoV19'

reload_from_pretrained=True
pretrained_path='/media/userdisk0/ywye/MAE/Pretrained_Model/MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05/checkpoint-299.pth'
exp_name='MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05'
model_name="model"
lr=0.0001
ratio_labels=1
num_epochs=100

seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/ratio_'$ratio_labels'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_2/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/QaTa_CoV19/train.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='224,224' \
--batch_size=16 \
--num_gpus=1 \
--num_epochs=$num_epochs \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--model_name=$model_name \
--ratio_labels=$ratio_labels

echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_2/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/QaTa_CoV19/evaluate.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--nnUNet_preprocessed=$nnudata \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='224,224' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True 

seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/ratio_'$ratio_labels'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_2/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/QaTa_CoV19/train.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='224,224' \
--batch_size=16 \
--num_gpus=1 \
--num_epochs=$num_epochs \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--model_name=$model_name \
--ratio_labels=$ratio_labels

echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_2/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/QaTa_CoV19/evaluate.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--nnUNet_preprocessed=$nnudata \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='224,224' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True



seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/ratio_'$ratio_labels'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_2/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/QaTa_CoV19/train.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='224,224' \
--batch_size=16 \
--num_gpus=1 \
--num_epochs=$num_epochs \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--model_name=$model_name \
--ratio_labels=$ratio_labels

echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_2/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/QaTa_CoV19/evaluate.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--nnUNet_preprocessed=$nnudata \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='224,224' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True

# #################################################################################################


gpu_id=0

task_id='3D_RICORD'


reload_from_pretrained=True
pretrained_path='/media/userdisk0/ywye/MAE/Pretrained_Model/MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05/checkpoint-299.pth'
exp_name='MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05'

data_path='/media/erwen_SSD/Continual_learning/3D/RICORD/'
epoch=200

lr=0.00001

seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/RICORD/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='64,192,192' \
--batch_size=8 \
--num_gpus=1 \
--num_epochs=$epoch \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=2 \
--num_workers=10 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed

seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/RICORD/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='64,192,192' \
--batch_size=8 \
--num_gpus=1 \
--num_epochs=$epoch \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=2 \
--num_workers=10 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed


seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/RICORD/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='64,192,192' \
--batch_size=8 \
--num_gpus=1 \
--num_epochs=$epoch \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=2 \
--num_workers=10 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed



# #################################################################################################

nnudata='/media/new_userdisk0/Continual_pretraining/3D/' #the location of downstream datasets (nnudata/Task100_MOTS/)

gpu_id=1

task_id='0Liver'

reload_from_pretrained=True
pretrained_path="/media/userdisk0/ywye/Contiual_pretraining/MAE/pretrained_model/MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05/checkpoint-299.pth"
exp_name='MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05'


lr=0.0001


seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/MOTS/train.py \
--arch='unified_vit' \
--train_list='Downstream/Dim_3/MOTS/data_list/'$task_id'/'$task_id'_train.txt' \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='64,192,192' \
--batch_size=2 \
--num_gpus=1 \
--num_epochs=100 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=2 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \


echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_3/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/MOTS/evaluate.py \
--arch='unified_vit' \
--nnUNet_preprocessed=$nnudata \
--val_list='Downstream/Dim_3/MOTS/data_list/'$task_id'/'$task_id'_test.txt' \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='64,192,192' \
--batch_size=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True \



seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/MOTS/train.py \
--arch='unified_vit' \
--train_list='Downstream/Dim_3/MOTS/data_list/'$task_id'/'$task_id'_train.txt' \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='64,192,192' \
--batch_size=2 \
--num_gpus=1 \
--num_epochs=100 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=2 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \


echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_3/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/MOTS/evaluate.py \
--arch='unified_vit' \
--nnUNet_preprocessed=$nnudata \
--val_list='Downstream/Dim_3/MOTS/data_list/'$task_id'/'$task_id'_test.txt' \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='64,192,192' \
--batch_size=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True \



seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/MOTS/train.py \
--arch='unified_vit' \
--train_list='Downstream/Dim_3/MOTS/data_list/'$task_id'/'$task_id'_train.txt' \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='64,192,192' \
--batch_size=2 \
--num_gpus=1 \
--num_epochs=100 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=2 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path 

echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_3/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/MOTS/evaluate.py \
--arch='unified_vit' \
--nnUNet_preprocessed=$nnudata \
--val_list='Downstream/Dim_3/MOTS/data_list/'$task_id'/'$task_id'_test.txt' \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='64,192,192' \
--batch_size=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True 


#############################################################################################################

nnudata='/media/new_userdisk0/Continual_pretraining/3D/Task061_VSseg/' 
gpu_id=3
task_id='VSseg'
reload_from_pretrained=True
pretrained_path='/media/userdisk0/ywye/Contiual_pretraining/MAE/pretrained_model/MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05/checkpoint-299.pth'
exp_name='MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05'

lr=0.0001


seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/VSseg/train.py \
--arch='unified_vit' \
--train_list=$nnudata'VSseg_train.txt' \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='64,192,192' \
--batch_size=2 \
--num_gpus=1 \
--num_epochs=100 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path 

echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_3/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/VSseg/evaluate.py \
--arch='unified_vit' \
--nnUNet_preprocessed=$nnudata \
--val_list=$nnudata'/VSseg_test.txt' \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='64,192,192' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True


seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/VSseg/train.py \
--arch='unified_vit' \
--train_list=$nnudata'VSseg_train.txt' \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='64,192,192' \
--batch_size=2 \
--num_gpus=1 \
--num_epochs=100 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path 

echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_3/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/VSseg/evaluate.py \
--arch='unified_vit' \
--nnUNet_preprocessed=$nnudata \
--val_list=$nnudata'/VSseg_test.txt' \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='64,192,192' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True 


seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/VSseg/train.py \
--arch='unified_vit' \
--train_list=$nnudata'VSseg_train.txt' \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='64,192,192' \
--batch_size=2 \
--num_gpus=1 \
--num_epochs=100 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path 

echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_3/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/VSseg/evaluate.py \
--arch='unified_vit' \
--nnUNet_preprocessed=$nnudata \
--val_list=$nnudata'/VSseg_test.txt' \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='64,192,192' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True

###############################################################################################################

nnudata='/media/erwen_SSD/Continual_learning/3D/LASeg/'
gpu_id=2

task_id='LASeg'

reload_from_pretrained=True
pretrained_path='/media/userdisk0/ywye/MAE/Pretrained_Model/MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05/checkpoint-299.pth'
exp_name='MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05'

lr=0.00005


seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/LASeg/train.py \
--arch='unified_vit' \
--train_list=$nnudata'train.list' \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='64,192,192' \
--batch_size=2 \
--num_gpus=1 \
--num_epochs=80 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path 

echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_3/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/LASeg/evaluate.py \
--arch='unified_vit' \
--nnUNet_preprocessed=$nnudata \
--val_list=$nnudata'test.list' \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='64,192,192' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True 



seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/LASeg/train.py \
--arch='unified_vit' \
--train_list=$nnudata'train.list' \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='64,192,192' \
--batch_size=2 \
--num_gpus=1 \
--num_epochs=80 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path


echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_3/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/LASeg/evaluate.py \
--arch='unified_vit' \
--nnUNet_preprocessed=$nnudata \
--val_list=$nnudata'test.list' \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='64,192,192' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True 



seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/LASeg/train.py \
--arch='unified_vit' \
--train_list=$nnudata'train.list' \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='64,192,192' \
--batch_size=2 \
--num_gpus=1 \
--num_epochs=80 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path 

echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_3/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/LASeg/evaluate.py \
--arch='unified_vit' \
--nnUNet_preprocessed=$nnudata \
--val_list=$nnudata'test.list' \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='64,192,192' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True 


##############################################################################################################

gpu_id=1

task_id='2D_NCT_CRC_HE'

reload_from_pretrained=True
pretrained_path='/media/userdisk0/ywye/MAE/Pretrained_Model/MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05/checkpoint-299.pth'
exp_name='MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05'

data_path='/media/erwen_SSD/Continual_learning/2D/NCT_CRC_HE/'

lr=0.0001

seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_2/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/NCT_CRC_HE/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='224,224' \
--batch_size=32 \
--num_gpus=1 \
--num_epochs=10 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=9 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed 

seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_2/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/NCT_CRC_HE/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='224,224' \
--batch_size=32 \
--num_gpus=1 \
--num_epochs=10 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=9 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed 

seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_2/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/NCT_CRC_HE/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='224,224' \
--batch_size=32 \
--num_gpus=1 \
--num_epochs=10 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=9 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed 

######################################################################################################

nnudata='/media/erwen_SSD/Continual_learning/2D/Glas/'

gpu_id=2

task_id='Glas'

reload_from_pretrained=True
pretrained_path='/media/userdisk0/ywye/MAE/Pretrained_Model/MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05/checkpoint-299.pth'
exp_name='MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05'

lr=0.0001


seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_2/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Glas/train.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='512,512' \
--batch_size=4 \
--num_gpus=1 \
--num_epochs=100 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path 

echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_2/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Glas/evaluate.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--nnUNet_preprocessed=$nnudata \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='512,512' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True 


seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_2/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Glas/train.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='512,512' \
--batch_size=4 \
--num_gpus=1 \
--num_epochs=100 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path

echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_2/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Glas/evaluate.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--nnUNet_preprocessed=$nnudata \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='512,512' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True 

seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_2/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Glas/train.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='512,512' \
--batch_size=4 \
--num_gpus=1 \
--num_epochs=100 \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path

echo $task_id" Evaluating"
output_dir='snapshots/downstream/dim_2/'$path_id'prediction/'
mkdir $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Glas/evaluate.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--nnUNet_preprocessed=$nnudata \
--reload_from_checkpoint=True \
--checkpoint_path=$snapshot_dir'checkpoint.pth' \
--save_path=$output_dir \
--input_size='512,512' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True





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
import util
from dataloader.Jointly_Dataset import Buffer_Dataset
from model.Unimodel import Unified_Model
from model.Unimodel_Teacher import Teacher_Unified_Model
from engine_pretrain_er import jointly_train_one_epoch_with_teacher


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='unified_vit', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # task 1D 2D 3D
    parser.add_argument('--task_modality', default="1D_text", choices=["1D_text", "2D_xray", "3D_CT", "3D_MR", "2D_path"], type=str, help='current task (modality)')
    parser.add_argument('--load_current_pretrained_weight', default="", type=str, help='pre-training path')
    parser.add_argument('--load_teacher_weight', default="", type=str, help='pre-training path')
    parser.add_argument('--num_center', type=float)
    parser.add_argument('--buffer_ratio', type=float)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--mix_up', type=int, default=1)

    # Dataset parameters
    parser.add_argument('--data_path_1D_text', default='', type=str, help='dataset path')
    parser.add_argument('--data_path_2D_xray', default='', type=str, help='dataset path')
    parser.add_argument('--data_path_3D_CT', default='', type=str, help='dataset path')
    parser.add_argument('--data_path_3D_MR', default='', type=str, help='dataset path')
    parser.add_argument('--data_path_2D_path', default='', type=str, help='dataset path')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    task_data = [args.task_modality]
    buffer_data = []
    if args.data_path_1D_text != "":
        buffer_data.append("1D_text")
    if args.data_path_2D_xray != "":
        buffer_data.append("2D_xray")
    if args.data_path_3D_CT != "":
        buffer_data.append("3D_CT")
    if args.data_path_3D_MR != "":
        buffer_data.append("3D_MR")
    if args.data_path_2D_path != "":
        buffer_data.append("2D_path")

    buffer_data.remove(args.task_modality) #exclude current modality data
    print("buffer_data", buffer_data)
    buffer_file_path = args.load_current_pretrained_weight.split("/checkpoint")[0]
    dataset_train = Buffer_Dataset(data_path_text=args.data_path_1D_text, data_path_xray=args.data_path_2D_xray,
                                          data_path_ct=args.data_path_3D_CT,
                                          data_path_mr=args.data_path_3D_MR, data_path_path=args.data_path_2D_path,
                                          max_words=112, crop_size=(16, 192, 192), imsize=224, #1D, 3D, and 2D input sizes
                                          batch_size=args.batch_size,
                                          buffer_data=buffer_data, task_data=task_data, num_center=args.num_center, buffer_ratio=args.buffer_ratio, exp_name=args.exp_name,
                                          buffer_file_path=buffer_file_path, file_copy_path=args.output_dir)
    
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )


    model = Unified_Model(now_1D_input_size=(112,1), now_2D_input_size=(224, 224), now_3D_input_size=(16, 192, 192), norm_pix_loss=args.norm_pix_loss)
    print("load student pretrained parameter from ", args.load_current_pretrained_weight)
    pretrained_weight = torch.load(args.load_current_pretrained_weight, map_location='cpu')
    pre_dict = pretrained_weight["model"]
    model_dict = model.state_dict()
    # print(model_dict)
    for k, v in pre_dict.items():
        if k in model_dict:
            if v.shape != model_dict[k].shape:
                print(v.shape, model_dict[k].shape)
    update_module = ["fused_encoder", "token_embed", "cls_token", "video_embed"]  # drop pre-trained weights of the decoder and projectors

    pre_dict_update = {k: v for k, v in pre_dict.items() if
                       (k in model_dict and sum([module in k for module in update_module]))}
    print(pre_dict_update.keys())

    pre_dict_no_update = [k for k in pre_dict.keys() if (
            k not in model_dict or sum([module not in k for module in update_module]) == len(update_module))]
    print("no update: ", pre_dict_no_update, len(pre_dict_no_update))
    print("[pre_%d/mod_%d]: %d shared layers" % (len(pre_dict), len(model_dict), len(pre_dict_update)))
    model_dict.update(pre_dict_update)
    model.load_state_dict(model_dict)
    print("load pre-trained model success!")
    del pre_dict, pretrained_weight, model_dict
    model.to(device)


    teacher_model = Teacher_Unified_Model(now_1D_input_size=(112, 1), norm_pix_loss=args.norm_pix_loss)
    model_dict = teacher_model.state_dict()
    if args.load_teacher_weight != "":
        print("load teacher pretrained parameters from ", args.load_teacher_weight)
        pretrained_weight = torch.load(args.load_teacher_weight, map_location='cpu')
    else:
        print("load teacher pretrained parameters from ", args.load_current_pretrained_weight)
        pretrained_weight = torch.load(args.load_current_pretrained_weight, map_location='cpu')
    pre_dict = pretrained_weight["model"]
    updated_keys = []
    not_found_keys = []

    for k in pre_dict.keys():
        if k in model_dict:
            updated_keys.append(k)
        else:
            not_found_keys.append(k)
    print("total teacher layers:", len(model_dict))
    print("Updated parameters:", updated_keys, len(updated_keys))
    print("Parameters not found in the model:", not_found_keys, len(not_found_keys))
    print("[pre_%d/mod_%d]: %d shared layers" % (len(pre_dict), len(model_dict), len(updated_keys)))
    teacher_model.load_state_dict(pre_dict, strict=False)
    print("load pre-trained model success!")
    del pre_dict, pretrained_weight, model_dict
    teacher_model.to(device)

 
    # print("Model = %s" % str(model_without_ddp))
    #
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu], find_unused_parameters=True)
        teacher_model_without_ddp = teacher_model.module
   
    for p in teacher_model.parameters():
        p.requires_grad = False

        
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed: 
            data_loader_train.sampler.set_epoch(epoch)

        
            
        train_stats = jointly_train_one_epoch_with_teacher(
            model, teacher_model, teacher_model_without_ddp, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args, current_task=args.task_modality
        )
        if args.output_dir and (epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            
        elif args.output_dir and ((epoch + 1) % 100  == 0):
            misc.save_model_every_epoch(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if misc.is_main_process():
        os.makedirs(os.path.join(args.output_dir, "code"), exist_ok=True)
        shutil.copytree("dataloader", os.path.join(args.output_dir, "code", "dataloader"), dirs_exist_ok=True)
        shutil.copytree("model", os.path.join(args.output_dir, "code", "model"), dirs_exist_ok=True)
        shutil.copytree("util", os.path.join(args.output_dir, "code", "util"), dirs_exist_ok=True)
        shutil.copyfile("run_ssl.sh", os.path.join(args.output_dir, "code", "run_ssl.sh"))
        shutil.copyfile("engine_pretrain_er.py", os.path.join(args.output_dir, "code", "engine_pretrain_er.py"))
        shutil.copyfile("main_pretrain_medcoss.py", os.path.join(args.output_dir, "code", "main_pretrain_medcoss.py"))
        shutil.copyfile("main_buffer_kmean.py", os.path.join(args.output_dir, "code", "main_buffer_kmean.py"))
    main(args)

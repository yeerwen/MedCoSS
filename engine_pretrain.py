# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
import sys
from typing import Iterable

import torch
import numpy as np
import util.misc as misc
import util.lr_sched as lr_sched
import cv2
import SimpleITK as sitk

def NiiDataWrite(path, prediction_final, spacing, origin, direction):
    # prediction_final = prediction_final.astype(as_type)
    img = sitk.GetImageFromArray(prediction_final)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    sitk.WriteImage(img, path)

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt="{global_avg:.6f}"))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir), "task_name", args.task_modality)

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.task_modality == "1D_text":
            samples = {"data": samples[0].to(device, non_blocking=True), "text_labels": samples[1].to(device, non_blocking=True),
                    "mask_attention": samples[2].to(device, non_blocking=True), "modality": "text", "task": "1D_text"}
        elif args.task_modality == "2D_xray":
            samples = {"data": samples.to(device, non_blocking=True), "modality": "2D image", "task": "2D_xray"}
        elif args.task_modality == "3D_CT":
            samples = {"data": samples.float().to(device, non_blocking=True), "modality": "3D image", "task": "3D_CT"}
        elif args.task_modality == "3D_MR":
            samples = {"data": samples.float().to(device, non_blocking=True), "modality": "3D image", "task": "3D_MR"}
        elif args.task_modality == "2D_path":
            samples = {"data": samples.to(device, non_blocking=True), "modality": "2D image", "task": "2D_path"}

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            (loss, _), _, _, _ = model(samples.copy(), mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update("loss", loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update("lr", lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    if misc.is_main_process() and (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            if "2D" in args.task_modality:
                loss, y, mask, (mean, var) = model(samples.copy(), mask_ratio=args.mask_ratio)
                y = y * (var + 1.e-6) ** .5 + mean
                y = model.module.unpatchify_2D(y)
                y = torch.einsum('nchw->nhwc', y).detach().cpu()

                # visualize the mask
                mask = mask.detach()
                mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_size ** 2*3)  # (N, H*W, p*p*3)
                mask = model.module.unpatchify_2D(mask)  # 1 is removing, 0 is keeping
                mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

                x = torch.einsum('nchw->nhwc', samples["data"].detach().cpu())
                # masked image
                im_masked = x * (1 - mask)

                # MAE reconstruction pasted with visible patches
                im_paste = x * (1 - mask) + y * mask
                x[0] = torch.clip((x[0]) * 255, 0, 255).int()
                im_masked[0] = torch.clip((im_masked[0]) * 255, 0, 255).int()
                y[0] = torch.clip((y[0]) * 255, 0, 255).int()
                im_paste[0] = torch.clip((im_paste[0]) * 255, 0, 255).int()
                image_all = torch.cat([x[0], im_masked[0], y[0], im_paste[0]], dim=1).cpu().numpy()

                cv2.imwrite(os.path.join(args.output_dir, f'epoch_{epoch}_{args.task_modality}_visual.png'), image_all)
            elif "3D" in args.task_modality:
                loss, y, mask, (mean, var) = model(samples.copy(), mask_ratio=args.mask_ratio)
                y = y * (var + 1.e-6) ** .5 + mean
                y = model.module.unpatchify_3D(y)
                y = torch.einsum('ncdhw->ndhwc', y).detach().cpu()

                # visualize the mask
                mask = mask.detach()
                mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_size ** 3)  # (N, H*W, p*p*3)
                mask = model.module.unpatchify_3D(mask)  # 1 is removing, 0 is keeping
                mask = torch.einsum('ncdhw->ndhwc', mask).detach().cpu()

                x = torch.einsum('ncdhw->ndhwc', samples["data"].detach().cpu())
                # masked image
                im_masked = x * (1 - mask)

                # MAE reconstruction pasted with visible patches
                im_paste = x * (1 - mask) + y * mask


                NiiDataWrite(os.path.join(args.output_dir, f'epoch_{epoch}_{args.task_modality}_img_ori.nii.gz'), x[0]*1024., (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                             (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                NiiDataWrite(os.path.join(args.output_dir, f'epoch_{epoch}_{args.task_modality}_mask.nii.gz'), im_masked[0]*1024., (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                             (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                NiiDataWrite(os.path.join(args.output_dir, f'epoch_{epoch}_{args.task_modality}_pred.nii.gz'), y[0]*1024., (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                             (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                NiiDataWrite(os.path.join(args.output_dir, f'epoch_{epoch}_{args.task_modality}_pred_mask.nii.gz'), im_paste[0]*1024., (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                             (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        model.train()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    global_avg_print = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", global_avg_print)
    return global_avg_print

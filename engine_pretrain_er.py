
import math
import os
import sys
from typing import Iterable
import torch
import util.misc as misc
import util.lr_sched as lr_sched


def jointly_train_one_epoch_with_teacher(model: torch.nn.Module, teacher_model, teacher_model_without_ddp,
                            data_loader: Iterable, optimizer: torch.optim.Optimizer,
                            device: torch.device, epoch: int, loss_scaler,
                            log_writer=None,
                            args=None, current_task=None):
    class_name = ["1D_text", "2D_xray", "3D_CT", "3D_MR", "2D_path"]
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt="{global_avg:.6f}"))
    metric_logger.add_meter('mse', misc.SmoothedValue(window_size=1))
    metric_logger.add_meter('1D_text', misc.SmoothedValue(window_size=1))
    metric_logger.add_meter('2D_xray', misc.SmoothedValue(window_size=1))
    metric_logger.add_meter('3D_CT', misc.SmoothedValue(window_size=1))
    metric_logger.add_meter('3D_MR', misc.SmoothedValue(window_size=1))
    metric_logger.add_meter('2D_path', misc.SmoothedValue(window_size=1))


    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir), "task_name", args.task_modality)

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # print("data", text.size(), "label", label.size(), "mask_attention", attention_mask.size())
        # print(samples)
        now_task_modality = class_name[samples[-1][0].long().item()]
        it = len(data_loader) * epoch + data_iter_step  # global training iteration

        if now_task_modality == "1D_text":
            samples = {"data": samples[0][0].long().to(device, non_blocking=True),
                       "text_labels": samples[1][0].long().to(device, non_blocking=True),
                       "mask_attention": samples[2][0].long().to(device, non_blocking=True), "modality": "text", "task": "1D_text"}

        elif now_task_modality == "2D_xray":
            samples = {"data": samples[0][0].float().to(device, non_blocking=True), "modality": "2D image", "task": "2D_xray"}
        elif now_task_modality == "3D_CT":
            samples = {"data": samples[0][0].float().to(device, non_blocking=True), "modality": "3D image", "task": "3D_CT"}
        elif now_task_modality == "3D_MR":
            samples = {"data": samples[0][0].float().to(device, non_blocking=True), "modality": "3D image", "task": "3D_MR"}
        elif now_task_modality == "2D_path":
            samples = {"data": samples[0][0].float().to(device, non_blocking=True), "modality": "2D image", "task": "2D_path"}
      
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            # torch.cuda.empty_cache()
            if args.task_modality != now_task_modality:
                if args.mix_up == 1:
                    if now_task_modality != "1D_text":
                        perm = torch.randperm(samples["data"].size()[0])
                        data_shuffled = samples["data"][perm]
                        if "2D" in samples["modality"]:
                            lambda_value = torch.rand(samples["data"].size()[0], 1, 1, 1).cuda()
                        elif  "3D" in samples["modality"]:
                            lambda_value = torch.rand(samples["data"].size()[0], 1, 1, 1, 1).cuda()
                        else:
                            exit()
                        samples["data"] = lambda_value * samples["data"] + (1 - lambda_value) * data_shuffled
                    else:
                        N, L = samples["data"].size()
                        perm = torch.randperm(N)
                        data_shuffled = samples["data"][perm]
                        attention_shuffled = samples["mask_attention"][perm]
                        mixup_ratio = torch.rand(1).item()
                        binary_mask =  (torch.rand(N, L) < mixup_ratio).long().cuda()
                        samples["data"] = binary_mask * samples["data"] + (1 - binary_mask) * data_shuffled
                        samples["mask_attention"] = binary_mask * samples["mask_attention"] + (1 - binary_mask) * attention_shuffled
                        
                #todo: Verify that noise can standardize input consistency
                latent_out, noise = model(samples.copy(), mask_ratio=args.mask_ratio, feature=True)
                with torch.no_grad():
                    target_out = teacher_model(samples.copy(), mask_ratio=args.mask_ratio, feature=True, noise=noise)
                loss_mse = ((target_out.detach() - latent_out) ** 2).mean()
                loss = loss_mse
                loss_mse_value = loss_mse.item()
                metric_logger.update("mse", loss_mse_value)
                
            else:
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
        metric_logger.update(now_task_modality, loss_value)


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

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    global_avg_print = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", global_avg_print)
    return global_avg_print


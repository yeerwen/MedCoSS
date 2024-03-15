import argparse
import os, sys
import numpy as np
from dataloader import ChestXR_Dataset
import os.path as osp
from model.Unimodel import Unified_Model
import timeit, time
from utils.ParaFlop import print_model_parm_nums
from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
from torch.cuda.amp import GradScaler, autocast
import shutil
import torch
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
start = timeit.default_timer()
import torch.backends.cudnn as cudnn

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Downstream PudMed20k tasks")

    parser.add_argument("--data_path", type=str, default='./data_list/')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/tmp/')

    parser.add_argument("--reload_from_pretrained", type=str2bool, default=False)
    parser.add_argument("--pretrained_path", type=str, default='../snapshots/xx/checkpoint.pth')

    parser.add_argument("--input_size", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--val_only", type=int, default=0)

    parser.add_argument("--power", type=float, default=0.9)

    # others
    parser.add_argument("--gpu", type=str, default='None')
    parser.add_argument("--arch", type=str, default='unified_vit')
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    if i_iter < 0:
        lr = 1e-2 * lr + i_iter * (lr - 1e-2 * lr) / 10.
    else:
        lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def adjust_alpha(i_iter, num_stemps):
    alpha_begin = 1
    alpha_end = 0.01
    decay = (alpha_begin - alpha_end) / num_stemps
    alpha = alpha_begin - decay * i_iter
    return alpha


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)
    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)

    return result




def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ["OMP_NUM_THREADS"] = "1"

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        # os.environ['nnUNet_preprocessed'] = args.nnUNet_preprocessed
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)


        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        cudnn.benchmark = True

        if os.path.exists(os.path.join(args.snapshot_dir, "code")):
            shutil.rmtree(os.path.join(args.snapshot_dir, "code"))
        shutil.copytree("Downstream/Dim_2/Chest_XR", os.path.join(args.snapshot_dir, "code"))
        shutil.copyfile("run_ds.sh", os.path.join(args.snapshot_dir, "code", "run_ds.sh"))
        print("code copy!")

        if args.arch == "unified_vit":
            model = Unified_Model(now_2D_input_size=[224, 224], num_classes=args.num_classes, pre_trained=args.reload_from_pretrained, pre_trained_weight=args.pretrained_path)
            print("unified_vit")
        
        else:
            exit()

        print_model_parm_nums(model)

        model.train()

        if args.num_gpus > 1:
            model = convert_syncbn_model(model)

        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)


        optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=0.0001)

        if args.FP16:
            print("Note: Using FP16 during training************")
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        if args.FP16:
            print("Using FP16 for training!!!")
            scaler = torch.cuda.amp.GradScaler()

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        # load checkpoint...
        to_restore = {"epoch": 0}
        restart_from_checkpoint(
            os.path.join(args.snapshot_dir, "checkpoint.pth"),
            run_variables=to_restore,
            model=model,
            optimizer=optimizer,
        )
        start_epoch = to_restore["epoch"]
        # print(args.input_size)
        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)
        trainloader, train_sampler = engine.get_train_loader(
            ChestXR_Dataset(args.data_path, split="train", crop_size=input_size),
            drop_last=True)

        valloader, val_sampler = engine.get_test_loader(
            ChestXR_Dataset(args.data_path, split="val", crop_size=input_size), batch_size=1)

        testloader, test_sampler = engine.get_test_loader(
            ChestXR_Dataset(args.data_path, split="test", crop_size=input_size), batch_size=1)

        print("train dataset len: {}, val dataset len: {}".format(len(trainloader), len(valloader)))
        all_tr_loss = []
        best_acc = -1
        for epoch in range(start_epoch, args.num_epochs):

            if args.val_only == 1:
                break

            time_t1 = time.time()

            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_loss = []
            epoch_acc = []
            adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)

            for iter, (input_ids, labels) in tqdm(enumerate(trainloader)):

                input_ids = input_ids.cuda(non_blocking=True)
                labels = labels.squeeze(1).long().cuda(non_blocking=True)

                data = {"data": input_ids, "labels": labels, "modality": "2D image"}
                optimizer.zero_grad()

                if args.FP16:
                    with autocast():
                        term_all = model(data)
                        del data

                        term_all = engine.all_reduce_tensor(term_all)
                    scaler.scale(term_all).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    term_all = model(data)
                    del data
                    reduce_all = engine.all_reduce_tensor(term_all)
                    term_all.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    optimizer.step()


                epoch_loss.append(float(reduce_all))

            epoch_loss = np.mean(epoch_loss)

            all_tr_loss.append(epoch_loss)

            time_t2 = time.time()

            if (args.local_rank == 0):
                print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}, time_cost = {}s'.format(epoch,
                                                                                           optimizer.param_groups[0][
                                                                                               'lr'],
                                                                                           epoch_loss.item(),
                                                                                           int(time_t2 - time_t1)))
            if args.local_rank == 0:
                model.eval()
                model.cal_acc = True
                pre_score = []
                label_val = []
                with torch.no_grad():
                    for iter, (input_ids, labels) in tqdm(enumerate(valloader)):
                        input_ids = input_ids.cuda(non_blocking=True)
                        labels = labels.squeeze(1).long().cuda(non_blocking=True)
                        data = {"data": input_ids, "labels": labels, "modality": "2D image"}
                        term_acc, pred_softmax = model(data)
                        epoch_acc.append(term_acc)
                        pre_score.append(pred_softmax.cpu().numpy())
                        label_val.append(labels.cpu().numpy())
                pre_score = np.concatenate(pre_score, 0)
                label_val = np.concatenate(label_val, 0)
                val_auc = metrics.roc_auc_score(label_val, pre_score, average='macro', multi_class="ovo")
                val_f1 = metrics.f1_score(label_val, np.argmax(pre_score, axis=-1), average='macro')

                model.train()
                model.cal_acc = False
                epoch_acc_mean = np.mean(epoch_acc)
                if best_acc < (epoch_acc_mean + val_auc + val_f1):
                    best_acc = epoch_acc_mean + val_auc + val_f1
                    print(f"save best weight: acc:{epoch_acc_mean}, auc: {val_auc}, f1: {val_f1}")
                    save_dict = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1,
                    }
                    torch.save(save_dict, osp.join(args.snapshot_dir, 'checkpoint.pth'))

        model.eval()
        print("load best weight from", osp.join(args.snapshot_dir, 'checkpoint.pth'))
        best_performance_weight = torch.load(osp.join(args.snapshot_dir, 'checkpoint.pth'))['model']
        model.load_state_dict(best_performance_weight, strict=True)
        model.cal_acc = True
        test_acc = []
        pre_score = []
        label_test = []
        with torch.no_grad():
            for iter, (input_ids, labels) in tqdm(enumerate(testloader)):
                input_ids = input_ids.cuda(non_blocking=True)
                labels = labels.squeeze(1).long().cuda(non_blocking=True)
                data = {"data": input_ids, "labels": labels, "modality": "2D image"}
                term_acc, pred_softmax = model(data)
                test_acc.append(term_acc)
                pre_score.append(pred_softmax.cpu().numpy())
                label_test.append(labels.cpu().numpy())

        pre_score = np.concatenate(pre_score, 0)
        label_test = np.concatenate(label_test, 0)
        test_auc = metrics.roc_auc_score(label_test, pre_score, average='macro', multi_class="ovo")
        test_f1 = metrics.f1_score(label_test, np.argmax(pre_score, axis=-1), average='macro')
        test_acc_mean = np.mean(test_acc)
        print("test dataset acc: {}, auc: {}, f1: {}".format(test_acc_mean, test_auc, test_f1))
        with open(os.path.join(args.snapshot_dir, "result.txt"), "w") as fp:
            fp.write("test dataset acc: {}, auc: {}, f1: {}".format(test_acc_mean, test_auc, test_f1))
        end = timeit.default_timer()
        print(end - start, 'seconds')


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    checkpoint = torch.load(ckp_path, map_location="cpu")
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded {} from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))
        else:
            print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))

    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


if __name__ == '__main__':
    main()

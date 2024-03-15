import argparse
import os, sys
sys.path.append("..")
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import os.path as osp
from DSDataset import Glas_Dataset, collate_fn_tr, collate_fn_ts
import random
import timeit, time
from tensorboardX import SummaryWriter
import loss_Single as loss
from utils.ParaFlop import print_model_parm_nums
from model.Unimodel import Unified_Model
from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
from torch.cuda.amp import GradScaler, autocast
import shutil
import torch.nn.functional as F
start = timeit.default_timer()

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
    parser = argparse.ArgumentParser(description="Downstream segmentation tasks")

    parser.add_argument("--data_dir", type=str, default='/media/new_userdisk0/JSRT/')
    parser.add_argument("--train_list", type=str, default='0Liver/0Liver_train5f_1.txt')
    parser.add_argument("--val_list", type=str, default='0Liver/0Liver_val5f_1.txt')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/tmp/')
    parser.add_argument("--nnUNet_preprocessed", type=str)

    parser.add_argument("--reload_from_pretrained", type=str2bool, default=False)
    parser.add_argument("--pretrained_path", type=str, default='../snapshots/xx/checkpoint.pth')

    parser.add_argument("--input_size", type=str, default='64,128,128')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=10)

    parser.add_argument("--weight_std", type=str2bool, default=False)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.00003)

    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--not_restore_last", action="store_true")
    parser.add_argument("--save_num_images", type=int, default=2)

    # data aug.
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)

    # others
    parser.add_argument("--gpu", type=str, default='None')
    parser.add_argument("--recurrence", type=int, default=1)
    parser.add_argument("--ft", type=str2bool, default=False)
    parser.add_argument("--ohem", type=str2bool, default='False')
    parser.add_argument("--ohem_thres", type=float, default=0.6)
    parser.add_argument("--ohem_keep", type=int, default=200000)

    parser.add_argument("--arch", type=str, default='res18')
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    if i_iter<0:
        lr = 1e-2*lr + i_iter*(lr - 1e-2*lr)/10.
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
    if torch.sum(preds) == 0 and torch.sum(labels) == 0:
        return torch.Tensor([1.0])
    else:
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
        os.environ['nnUNet_preprocessed'] = args.nnUNet_preprocessed
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        cudnn.benchmark = True

        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)
        #

        if os.path.exists(os.path.join(args.snapshot_dir, "code")):
            shutil.rmtree(os.path.join(args.snapshot_dir, "code"))
        # print(os.getcwd())
        shutil.copytree("Downstream/Dim_2/Glas", os.path.join(args.snapshot_dir, "code"))
        shutil.copyfile("run_ds.sh", os.path.join(args.snapshot_dir, "code", "run_ds.sh"))
        print("code copy!")

        if args.arch == "unified_vit":
            model = Unified_Model(now_2D_input_size=(512, 512), in_chans=3, num_classes=args.num_classes, pre_trained=args.reload_from_pretrained, pre_trained_weight=args.pretrained_path)
            print("unified_vit")

        else:
            exit()


        print_model_parm_nums(model)

        model.train()

        if args.num_gpus > 1:
            model = convert_syncbn_model(model)

        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        # print(model)
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

        loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).to(device)
        loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255).to(device)

        trainloader, train_sampler = engine.get_train_loader(
            Glas_Dataset(args.data_dir, args.train_list, max_iters=args.itrs_each_epoch * args.batch_size,
                        crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror, split="train"),
            drop_last=True, collate_fn=collate_fn_tr)

        valloader, val_sampler = engine.get_test_loader(
            Glas_Dataset(args.data_dir, args.val_list,
                        crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror, split="val"), collate_fn=collate_fn_ts)

        all_tr_loss = []
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None

        val_best_loss = 999999
        best_dice = 0
        for epoch in range(start_epoch, args.num_epochs):

            time_t1 = time.time()

            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_loss = []
            adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)
            model.train()
            for iter, batch in enumerate(trainloader):
                #
                if iter>=args.itrs_each_epoch:
                    break
                images = torch.from_numpy(batch['image']).cuda(non_blocking=True)
                labels = torch.from_numpy(batch['label']).cuda(non_blocking=True)

                data = {"data": images, "labels": labels, "modality": "2D image"}
                optimizer.zero_grad()
                
                if args.FP16:
                    with autocast():
                        preds = model(data)
                        del images
                        term_seg_Dice = loss_seg_DICE.forward(preds, labels)
                        term_seg_BCE = loss_seg_CE.forward(preds, labels)
                        term_all = term_seg_Dice + term_seg_BCE

                        reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                        reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                        reduce_all = engine.all_reduce_tensor(term_all)

                    scaler.scale(term_all).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    preds = model(data)
                    # print(iter, torch.mean(preds))
                    del images
                    # print(preds.size(), labels.size())
                    term_seg_Dice = loss_seg_DICE.forward(preds, labels)
                    term_seg_BCE = loss_seg_CE.forward(preds, labels)
                    term_all = term_seg_Dice + term_seg_BCE

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)
                    
                    term_all.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    optimizer.step()

                epoch_loss.append(float(reduce_all))

            epoch_loss = np.mean(epoch_loss)

            all_tr_loss.append(epoch_loss)

            time_t2 = time.time()

            if (args.local_rank == 0):
                print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}, time_cost = {}s'.format(epoch, optimizer.param_groups[0]['lr'],
                                                                          epoch_loss.item(), int(time_t2 - time_t1)))
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Train_loss', epoch_loss.item(), epoch)
            model.eval()
            with torch.no_grad():
                class_0 = []
                for iter, batch in enumerate(valloader):
                    #
                    images = torch.from_numpy(batch['image']).cuda(non_blocking=True)
                    labels = torch.from_numpy(batch['label']).cuda(non_blocking=True)
                    data = {"data": images, "labels": labels, "modality": "2D image"}
                    if args.FP16:
                        with autocast():
                            preds = model(data)
                            del images
                    else:
                        preds = model(data)
                        preds = F.sigmoid(preds)
                    preds[preds >= 0.5] = 1
                    preds[preds < 0.5] = 0
                    class_0.append(dice_score(preds, labels).item())

            total_mean_dice = np.mean(class_0)
            if total_mean_dice > best_dice:
                best_dice = total_mean_dice
                print("dice 0: {}".format(np.mean(class_0)))

                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                }
                if args.local_rank == 0:
                    torch.save(save_dict, osp.join(args.snapshot_dir, 'checkpoint.pth'))

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


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch['image'] = torch.from_numpy(self.next_batch['image']).cuda(non_blocking=True)
            self.next_batch['label'] = torch.from_numpy(self.next_batch['label']).cuda(non_blocking=True)
            self.next_batch['task_id'] = torch.from_numpy(self.next_batch['task_id']).cuda(non_blocking=True)
            self.next_batch['image'] = self.next_batch['image'].float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            batch['image'].record_stream(torch.cuda.current_stream())
            batch['label'].record_stream(torch.cuda.current_stream())
            batch['task_id'].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch

if __name__ == '__main__':
    main()

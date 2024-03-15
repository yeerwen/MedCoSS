import argparse
import os, sys

sys.path.append("..")

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import math
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from model.Unimodel import Unified_Model
from DSDataset import QaTa_CoV19_Dataset, collate_fn_ts
import random
import timeit
from utils.ParaFlop import print_model_parm_nums, print_model_parm_flops, torch_summarize_df
from sklearn import metrics
import nibabel as nib
from math import ceil
from engine import Engine
# from torch.cuda.amp import GradScaler, autocast
from skimage.measure import label as LAB
import SimpleITK as sitk
from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg
from medpy.metric import hd95
from collections import OrderedDict
import json

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
    parser = argparse.ArgumentParser(description="DS evaluation!")

    parser.add_argument("--data_dir", type=str, default='/media/new_userdisk0/JSRT/')
    parser.add_argument("--val_list", type=str, default='xx.txt')
    parser.add_argument("--dataset_path", type=str, default='/media/new_userdisk0/JSRT/')

    parser.add_argument("--nnUNet_preprocessed", type=str)
    parser.add_argument("--save_path", type=str, default='outputs/')

    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--checkpoint_path", type=str, default='xx.pth')

    parser.add_argument("--reload_from_pretrained", type=str2bool, default=False)
    parser.add_argument("--pretrained_path", type=str, default='xx/checkpoint.pth')

    parser.add_argument("--input_size", type=str, default='64,192,192')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--weight_std", type=str2bool, default=False)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    parser.add_argument("--isHD", type=str2bool, default=False)
    parser.add_argument("--arch", type=str, default='res50')
    return parser


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003



def dice_score(preds, labels):
    preds = preds[np.newaxis, :]
    labels = labels[np.newaxis, :]
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    if np.sum(target) == 0 and np.sum(predict) == 0:
        return 1.0
    else:
        num = np.sum(np.multiply(predict, target), axis=1)
        den = np.sum(predict, axis=1) + np.sum(target, axis=1)

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


def compute_dice_score(preds, labels):
    # preds: 1x4x128x128x128
    # labels: 1x128x128x128

    preds = torch.sigmoid(preds)

    pred_pa = preds[:, 0, :, :, :]
    label_pa = labels[:, 0, :, :, :]
    dice_pa = dice_score(pred_pa, label_pa)

    return dice_pa

def compute_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 500.
    elif num_pred == 0 and num_ref != 0:
        return 500.
    else:
        return hd95(pred, ref, (1, 1))


def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def multi_net(net_list, img, args):
    # img = torch.from_numpy(img).cuda()
    data = {"data": img, "labels": None, "modality": "2D image"}
    padded_prediction = net_list[0](data)

    padded_prediction = torch.sigmoid(padded_prediction)
    return padded_prediction  # .cpu().data.numpy()


def predict_sliding(args, net_list, image, tile_size, classes, gaussian_importance_map):  # tile_size:32x256x256

    flag_padding = False
    rows_missing = math.ceil(tile_size[0] - image.shape[2])
    cols_missing = math.ceil(tile_size[1] - image.shape[3])
    if rows_missing < 0:
        rows_missing = 0
    if cols_missing < 0:
        cols_missing = 0

    image = np.pad(image, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), constant_values = (-1,-1))

    image_size = image.shape
    overlap = 1 / 2

    strideHW = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    full_probs = np.zeros(
        (image_size[0], classes, image_size[2], image_size[3]))  # .astype(np.float32)  # 1x4x155x240x240
    count_predictions = np.zeros(
        (image_size[0], classes, image_size[2], image_size[3]))  # .astype(np.float32)
    full_probs = torch.from_numpy(full_probs)
    count_predictions = torch.from_numpy(count_predictions)


    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * strideHW)
            y1 = int(row * strideHW)
            x2 = min(x1 + tile_size[0], image_size[2])
            y2 = min(y1 + tile_size[1], image_size[3])
            x1 = max(int(x2 - tile_size[0]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[1]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            img = torch.from_numpy(img).cuda()

            prediction1 = multi_net(net_list, img, args)
            prediction2 = torch.flip(multi_net(net_list, torch.flip(img, [2]), args), [2])
            prediction3 = torch.flip(multi_net(net_list, torch.flip(img, [3]), args), [3])
            prediction4 = torch.flip(multi_net(net_list, torch.flip(img, [2, 3]), args), [2, 3])

            prediction = (prediction1 + prediction2 + prediction3 + prediction4) / 4.
            prediction = prediction.cpu()

            prediction[:, :] *= gaussian_importance_map

            if isinstance(prediction, list):
                shape = np.array(prediction[0].shape)
                shape[0] = prediction[0].shape[0] * len(prediction)
                shape = tuple(shape)
                preds = torch.zeros(shape).cuda()
                bs_singlegpu = prediction[0].shape[0]
                for i in range(len(prediction)):
                    preds[i * bs_singlegpu: (i + 1) * bs_singlegpu] = prediction[i]
                count_predictions[:, :, y1:y2, x1:x2] += 1
                full_probs[:, :, y1:y2, x1:x2] += preds

            else:
                count_predictions[:, :, y1:y2, x1:x2] += gaussian_importance_map
                full_probs[:, :, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs[:,:, :(image_size[2]-rows_missing), :(image_size[3]-cols_missing)]


def save_nii(args, pred, name):  # bs, c, WHD
    label_2_gray = {0: 0, 1: 255}
    # segmentation = pred.transpose((1, 2, 0))  # bsx240x240x155
    segmentation = pred

    for lab, gray in label_2_gray.items():
        segmentation[segmentation == lab] = gray
    # save
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    cv2.imwrite(os.path.join(args.save_path, name[0]), segmentation)
    return None


def continues_region_extract_organ(label, keep_region_nums):  # keep_region_nums=1
    mask = False*np.zeros_like(label)
    regions = np.where(label>=1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, background=0, connectivity=2, return_num=True)

    #
    ary_num = np.zeros(shape=(n+1,1))
    # print(n)
    for i in range(0, n+1):
        # print(i)
        ary_num[i] = np.sum(L==i)
    max_index = np.argsort(-ary_num, axis=0)
    count=1
    for i in range(1, n+1):
        if count<=keep_region_nums: # keep
            mask = np.where(L == max_index[i][0], label, mask)
            count+=1
        # else: # remove
        #     label = np.where(L == max_index[i][0], np.zeros_like(label), label)
    label = np.where(mask==True, label, np.zeros_like(label))
    return label

def continues_region_extract_tumor(label):  #

    regions = np.where(label>=1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, background=0, connectivity=2, return_num=True)

    #
    for i in range(1, n+1):
        if np.sum(L==i)<=50: # remove
            label = np.where(L == i, np.zeros_like(label), label)

    return label


def validate(args, input_size, model, ValLoader, num_classes):
    for index, batch in tqdm(enumerate(ValLoader)):
        # print('%d processd' % (index))
        image, label, name = batch['image'], batch["label"], batch["name"]

        # print("Processing %s" % name)
        with torch.no_grad():
            gaussian_importance_map = _get_gaussian(input_size, sigma_scale=1. / 4)
            pred = predict_sliding(args, model, image, input_size, num_classes, gaussian_importance_map)
            # print("pred", pred.shape)
            seg_pred_class = np.asarray(np.around(pred), dtype=np.uint8)

            seg_pred = np.zeros(shape=(pred.shape[2],pred.shape[3]))
            for index in range(num_classes):
                each_class_pre = seg_pred_class[0, index, :, :] # bs should be 1
                seg_pred = np.where(each_class_pre == 1, index+1, seg_pred)

            # save
            save_nii(args, seg_pred, name)
            # print("Saving done.")


    # evaluate metrics
    print("Start to evaluate...")
    val_Dice = [[] for _ in range(num_classes)]
    val_HD = [[] for _ in range(num_classes)]
    gray_2_label = {0: 0, 255: 1}
    for root, dirs, files in os.walk(args.save_path):
        for i in  tqdm(sorted(files)):
            if i[-4:]!='.png':
                continue
            i_file = os.path.join(root, i)
            i2_file = os.path.join(os.environ["nnUNet_preprocessed"], 'Test Set', "Ground-truths", "mask_"+i)
            pred = cv2.imread(i_file, flags=0)
            label = cv2.imread(i2_file, flags=0)

            for gray, lab_num in gray_2_label.items():
                pred[pred == gray] = lab_num
                label[label == gray] = lab_num

            assert np.max(pred) <= 1 and np.max(label) <= 1, [np.max(pred), np.max(label)]
            for num_class in range(1, num_classes+1):
                dice = dice_score(pred == num_class, label  == num_class)
                val_Dice[num_class-1].append(dice)
                if args.isHD:
                    HD = compute_HD95(pred == num_class, label == num_class)
                    val_HD[num_class - 1].append(HD)
                else:
                    HD = 500.
                    val_HD[num_class-1].append(HD)
                # print(i, dice, HD)

    print("Sum results")
    for t in range(num_classes):
        print('Sum: Task%d- len: [%d] Organ:[Dice-%.8f; HD-%.8f]' %
              (t, len(val_Dice[t]), np.mean(val_Dice[t]), np.mean(val_HD[t])))


    return val_Dice, val_HD


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        os.environ['nnUNet_preprocessed'] = args.nnUNet_preprocessed
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        cudnn.benchmark = True

        if args.arch == "unified_vit":
            model = Unified_Model(now_2D_input_size=input_size, in_chans=3, num_classes=args.num_classes, pre_trained=args.reload_from_pretrained, pre_trained_weight=args.pretrained_path, model_name="model")
            print("unified_vit")

        else:
            exit()
        print_model_parm_nums(model)

        model.eval()

        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=0.0001)
        if args.FP16:
            print("Using FP16 for training!!!")
            scaler = torch.cuda.amp.GradScaler()

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.checkpoint_path))
            if os.path.exists(args.checkpoint_path):
                if args.FP16:
                    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
                    pre_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
                    # pre_dict = checkpoint['model']
                    model.load_state_dict(pre_dict)
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scaler.load_state_dict(checkpoint['scaler'])
                else:
                    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
                    pre_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
                    # pre_dict = checkpoint
                    model.load_state_dict(pre_dict)
            else:
                print('File not exists in the reload path: {}'.format(args.checkpoint_path))

        valloader, val_sampler = engine.get_test_loader(
           QaTa_CoV19_Dataset(args.data_dir, args.val_list, split="test"), collate_fn=collate_fn_ts)

        json_dict = OrderedDict()
        json_dict['name'] = "Single"
        json_dict["meanDice"] = OrderedDict()
        json_dict["meanHD"] = OrderedDict()

        print('validate ...')
        val_Dice, val_HD = validate(args, input_size, [model], valloader, args.num_classes)

        with open(os.path.join(args.save_path, "result.txt"), 'w') as f:
            for i, (dice, hd) in enumerate(zip(val_Dice, val_HD)):
                f.write(str(i) + "  Dice: "+str(dice) + "   HD: " + str(hd) + "\n" + "mean: Dice: {}, HD: {}".format(np.mean(dice), np.mean(hd)))

        end = timeit.default_timer()
        print(end - start, 'seconds')


if __name__ == '__main__':
    main()


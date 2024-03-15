import argparse
import os, sys
sys.path.append("..")
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from scipy.ndimage.filters import gaussian_filter
import math
from tqdm import tqdm
from dataloader import VSseg_Dataset_test
import timeit
from model.Unimodel import Unified_Model
from utils.ParaFlop import print_model_parm_nums
import nibabel as nib
from math import ceil
from engine import Engine
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

    parser.add_argument("--data_dir", type=str, default='./data_list/')
    parser.add_argument("--val_list", type=str, default='xx.txt')
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

    parser.add_argument("--weight_std", type=str2bool, default=False)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    parser.add_argument("--isHD", type=str2bool, default=False)
    parser.add_argument("--arch", type=str, default='res50')
    parser.add_argument("--random_seed", type=int, default=1234)
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
    num = np.sum(np.multiply(predict, target), axis=1)
    den = np.sum(predict, axis=1) + np.sum(target, axis=1) + 1
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
        return hd95(pred, ref, (1, 1, 1))


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


def multi_net(net_list, img):
    # img = torch.from_numpy(img).cuda()
    data = {"data": img, "modality": "3D image"}
    padded_prediction = net_list[0](data)
    padded_prediction = torch.sigmoid(padded_prediction)
    for i in range(1, len(net_list)):
        padded_prediction_i = net_list[i](img)
        padded_prediction_i = torch.sigmoid(padded_prediction_i)
        padded_prediction += padded_prediction_i
    padded_prediction /= len(net_list)
    return padded_prediction  # .cpu().data.numpy()


def predict_sliding(args, net_list, image, tile_size, classes, gaussian_importance_map):  # tile_size:32x256x256

    flag_padding = False
    dept_missing = math.ceil(tile_size[0] - image.shape[2])
    rows_missing = math.ceil(tile_size[1] - image.shape[3])
    cols_missing = math.ceil(tile_size[2] - image.shape[4])
    if rows_missing < 0:
        rows_missing = 0
    if cols_missing < 0:
        cols_missing = 0
    if dept_missing < 0:
        dept_missing = 0
    image = np.pad(image, ((0, 0), (0, 0), (0, dept_missing), (0, rows_missing), (0, cols_missing)), constant_values = (-1,-1))

    image_size = image.shape
    overlap = 1 / 2

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = np.zeros(
        (image_size[0], classes, image_size[2], image_size[3], image_size[4]))  # .astype(np.float32)  # 1x4x155x240x240
    count_predictions = np.zeros(
        (image_size[0], classes, image_size[2], image_size[3], image_size[4]))  # .astype(np.float32)
    full_probs = torch.from_numpy(full_probs)
    count_predictions = torch.from_numpy(count_predictions)

    for dep in tqdm(range(tile_deps)):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)  # for portrait images the x1 underflows sometimes
                y1 = max(int(y2 - tile_size[1]), 0)  # for very few rows y1 underflows

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img = torch.from_numpy(img).cuda()

                prediction1 = multi_net(net_list, img)
                prediction2 = torch.flip(multi_net(net_list, torch.flip(img, [2])), [2])
                prediction3 = torch.flip(multi_net(net_list, torch.flip(img, [3])), [3])
                prediction4 = torch.flip(multi_net(net_list, torch.flip(img, [4])), [4])
                prediction5 = torch.flip(multi_net(net_list, torch.flip(img, [2, 3])), [2, 3])
                prediction6 = torch.flip(multi_net(net_list, torch.flip(img, [2, 4])), [2, 4])
                prediction7 = torch.flip(multi_net(net_list, torch.flip(img, [3, 4])), [3, 4])
                prediction8 = torch.flip(multi_net(net_list, torch.flip(img, [2, 3, 4])), [2, 3, 4])
                prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5 + prediction6 + prediction7 + prediction8) / 8.
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
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += preds

                else:
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += gaussian_importance_map
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs[:,:,:(image_size[2]-dept_missing), :(image_size[3]-rows_missing), :(image_size[4]-cols_missing)]


def save_nii(args, pred, name, properties):  # bs, c, WHD

    segmentation = pred

    # save
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    current_shape = segmentation.shape

    shape_original_after_cropping = np.array(properties.get('size_after_cropping'), dtype='int')
    shape_original_before_cropping = properties.get('original_size_of_raw_data')[0].data.numpy()

    order = 0
    force_separate_z = None

    if np.any(np.array(current_shape) != np.array(shape_original_after_cropping)):
        if order == 0:
            seg_old_spacing = resize_segmentation(segmentation, shape_original_after_cropping, 0)
        else:
            if force_separate_z is None:
                if get_do_separate_z(properties.get('original_spacing').data.numpy()[0]):
                    do_separate_z = True
                    lowres_axis = get_lowres_axis(properties.get('original_spacing').data.numpy()[0])
                elif get_do_separate_z(properties.get('spacing_after_resampling').data.numpy()):
                    do_separate_z = True
                    lowres_axis = get_lowres_axis(properties.get('spacing_after_resampling').data.numpy()[0])
                else:
                    do_separate_z = False
                    lowres_axis = None
            else:
                do_separate_z = force_separate_z
                if do_separate_z:
                    lowres_axis = get_lowres_axis(properties.get('original_spacing').data.numpy()[0])
                else:
                    lowres_axis = None

            print("separate z:", do_separate_z, "lowres axis", lowres_axis)
            seg_old_spacing = resample_data_or_seg(segmentation[None], shape_original_after_cropping, is_seg=True,
                                                   axis=lowres_axis, order=order, do_separate_z=do_separate_z, cval=0,
                                                   order_z=0)[0]
    else:
        seg_old_spacing = segmentation

    bbox = properties.get('crop_bbox')
    # print(pred.shape, seg_old_spacing.shape, shape_original_after_cropping, shape_original_before_cropping, bbox)
    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
    seg_resized_itk.SetSpacing(np.array(properties['itk_spacing']).astype(np.float64))
    seg_resized_itk.SetOrigin(np.array(properties['itk_origin']).astype(np.float64))
    seg_resized_itk.SetDirection(np.array(properties['itk_direction']).astype(np.float64))
    sitk.WriteImage(seg_resized_itk, args.save_path + name[0]+'.nii.gz')

    return None


def continues_region_extract_organ(label, keep_region_nums):  # keep_region_nums=1
    mask = False*np.zeros_like(label)
    regions = np.where(label>=1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, background=0, connectivity=2, return_num=True)

    ary_num = np.zeros(shape=(n+1,1))

    for i in range(0, n+1):
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

def validate(args, input_size, model, ValLoader, num_classes, engine, json_dict):
    for index, batch in enumerate(ValLoader):
        # print('%d processd' % (index))

        image, label, name, properties = batch
        print("Processing %s" % name[0])
        if os.path.exists(args.save_path + name[0] + '.nii.gz'):
            continue
        with torch.no_grad():
            gaussian_importance_map = _get_gaussian(input_size, sigma_scale=1. / 8)
            pred = predict_sliding(args, model, image.numpy(), input_size, num_classes, gaussian_importance_map)
            seg_pred_2class = np.asarray(np.around(pred), dtype=np.uint8)
            pred_tumor_1 = seg_pred_2class[0, 0, :, :, :] # bs should be 1
            seg_pred = np.zeros_like(pred_tumor_1)
            pred_all = np.where(pred_tumor_1 == 1, 1, seg_pred)
            # print("pred all shape", pred_all.shape)
            # save
            save_nii(args, pred_all, name, properties)
            print("Saving done.")


    # evaluate metrics
    print("Start to evaluate...")
    val_Dice = torch.zeros(size=(1, 1))
    val_HD = torch.zeros(size=(1, 1))
    count_Dice = torch.zeros(size=(1, 1))
    count_HD = torch.zeros(size=(1, 1))

    for root, dirs, files in os.walk(args.save_path):
        for i in sorted(files):
            if i[-6:]!='nii.gz':
                continue
            i_file = os.path.join(root, i)
            i2_file = os.path.join(os.environ["nnUNet_preprocessed"], 'gt_segmentations', i)
            predNII = nib.load(i_file)
            labelNII = nib.load(i2_file)
            pred = predNII.get_data()
            label = labelNII.get_data()
            task_id = 0
            #
            dice_1 = dice_score(pred == 1, label == 1)
            val_Dice[task_id, 0] += dice_1

            if args.isHD:
                HD_1 = compute_HD95(pred == 1, label == 1)
            else:
                HD_1 = 500.

            val_HD[task_id, 0] += HD_1 
            count_Dice[task_id, 0] += 1
            count_HD[task_id, 0] += 1

            log_i = ("tumor_1-[Dice-%.4f; HD-%.4f]" % (dice_1, HD_1))
            print("%s: %s" % (i,log_i))
            json_dict[i]=log_i


    count_Dice[count_Dice == 0] = 1
    count_HD[count_HD == 0] = 1
    val_Dice = val_Dice / count_Dice
    val_HD = val_HD / count_HD

    print("Sum results")
    for t in range(1):
        print('Sum: Task%d- tumor_1-[Dice-%.8f; HD-%.8f]' %
              (t, val_Dice[t, 0], val_HD[t, 0]))


    return val_Dice.data.numpy(), val_HD.data.numpy()


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        os.environ['nnUNet_preprocessed'] = args.nnUNet_preprocessed
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)

        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        cudnn.benchmark = True

        if args.arch == "unified_vit":
            model = Unified_Model(now_3D_input_size=input_size, num_classes=args.num_classes, pre_trained=args.reload_from_pretrained,
                                  pre_trained_weight=args.pretrained_path, in_chans=1)
            print("unified_vit")
        else:
            print("error net name")
            exit()
        print_model_parm_nums(model)

        model.eval()

        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)


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
                    scaler.load_state_dict(checkpoint['scaler'])
                else:
                    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
                    pre_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
                    model.load_state_dict(pre_dict)
            else:
                print('File not exists in the reload path: {}'.format(args.checkpoint_path))

        valloader, val_sampler = engine.get_test_loader(VSseg_Dataset_test(args.data_dir, args.val_list), 1)

        json_dict = OrderedDict()
        json_dict['name'] = "Single"
        json_dict["meanDice"] = OrderedDict()
        json_dict["meanHD"] = OrderedDict()

        print('validate ...')
        val_Dice, val_HD = validate(args, input_size, [model], valloader, args.num_classes, engine, json_dict)

        json_dict["meanDice"]["tumor_1"] = str(val_Dice[0][0])

        json_dict["meanHD"]["tumor_1"] = str(val_HD[0][0])

        print(json_dict["meanDice"])
        print(json_dict["meanHD"])

        with open(os.path.join(args.save_path, "summary.json"), 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=True)

        end = timeit.default_timer()
        print(end - start, 'seconds')


if __name__ == '__main__':
    main()

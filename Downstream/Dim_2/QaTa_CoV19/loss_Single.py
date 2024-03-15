import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from torch import Tensor, einsum


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth
        dice_score = 2*num / den
        dice_loss = 1 - dice_score
        dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]
        return dice_loss_avg

class DiceLoss4MOTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.smooth = 1

    def forward(self, predict, target):
        predict = F.sigmoid(predict)
        predict2 = predict.contiguous().view(predict.shape[0], predict.shape[1], -1)
        target2 = target.contiguous().view(target.shape[0], target.shape[1], -1)
        num = torch.sum(torch.mul(predict2, target2), dim=-1)
        den = torch.sum(predict2, dim=-1) + torch.sum(target2, dim=-1)
        dice_score = (2 * num) / (den + self.smooth)
        dice_score = dice_score[target[:,:,0,0]!=-1]
        dice_loss = 1 - dice_score.mean()

        return dice_loss

class CELoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(CELoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(2):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

        return weights

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        ce_loss = self.criterion(predict, target)
        ce_loss = torch.mean(ce_loss, dim=[2,3])
        ce_loss = ce_loss[target[:,:,0,0]!=-1]
        return ce_loss.mean()

import numpy as np

import torch
from torch import nn


def one_hot(net_output, target):
    shp_x = net_output.shape  # (batch size,class_num,x,y,z)
    shp_y = target.shape  # (batch size,1,x,y,z)
    # one hot code for gt
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            target = target.view((shp_y[0], 1, *shp_y[1:]))
        if all([i == j for i, j in zip(net_output.shape, target.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = target
        else:
            gt = target.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
    return y_onehot


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, alpha=2, gamma=2, balance_index=0, smooth=1e-5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        logit = torch.softmax(logit, dim=1)
        num_class = logit.shape[1]

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)
        idx = target.view(-1, 1).cpu().long()
        y_onehot = one_hot(logit, target)
        pt = (y_onehot * torch.softmax(logit, dim=1)).sum(1) + self.smooth
        logpt = pt.log()
        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt
        loss = loss.mean()
        return loss

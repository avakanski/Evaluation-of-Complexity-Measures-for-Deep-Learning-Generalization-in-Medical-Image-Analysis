# This file is used for defining the metrics

import torch.nn as nn

# Dice score coefficient and Dice loss are used for the segmentation task
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth)
        return 1. - dsc

def DiceCoeff(y_pred, y_true):
    smooth = 1.0
    assert y_pred.size() == y_true.size()
    y_pred = y_pred[:, 0].contiguous().view(-1)
    y_true = y_true[:, 0].contiguous().view(-1)
    intersection = (y_pred * y_true).sum().float()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
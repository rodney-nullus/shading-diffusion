import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceBoundaryLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, boundary_weight=1.0, smooth=1e-5):
        super(BCEDiceBoundaryLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def dice_loss(self, probs, targets):
        """ 计算 Dice Loss """
        num = 2 * torch.sum(probs * targets) + self.smooth
        den = torch.sum(probs) + torch.sum(targets) + self.smooth
        return 1 - (num / den)

    def boundary_loss(self, probs, targets):
        """ 计算梯度边界损失 """
        probs_grad_x = torch.abs(probs[:, :, :, :-1] - probs[:, :, :, 1:])
        probs_grad_y = torch.abs(probs[:, :, :-1, :] - probs[:, :, 1:, :])

        targets_grad_x = torch.abs(targets[:, :, :, :-1] - targets[:, :, :, 1:])
        targets_grad_y = torch.abs(targets[:, :, :-1, :] - targets[:, :, 1:, :])

        loss_x = F.l1_loss(probs_grad_x, targets_grad_x)
        loss_y = F.l1_loss(probs_grad_y, targets_grad_y)

        return loss_x + loss_y

    def forward(self, logits, targets):
        """ 计算总损失 """
        probs = torch.sigmoid(logits)

        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(probs, targets)
        boundary = self.boundary_loss(probs, targets)

        total_loss = self.bce_weight * bce + self.dice_weight * dice + self.boundary_weight * boundary
        return total_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight


@LOSSES.register("bceloss")
class BCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None):
        super(BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None

    def forward(self, logits, targets):
        logits = logits[0]
        device = logits.device  # Get device from input

        # It is used to smooth the target
        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # print("loss_m")
        # print(loss_m)

        targets_mask = torch.where(
            targets.detach().cpu() > 0.5,
            torch.ones(1, device=device),
            torch.zeros(1, device=device)
        )

        # print("targets_mask")
        # print(targets_mask)
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask.cpu(), self.sample_weight)
            loss_m = (loss_m * sample_weight.to(device))
            # print("sample_weight")
            # print(sample_weight)
            # print("loss_m")
            # print(loss_m)

        # losses = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()
        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()
        # print("loss")
        # print(loss)

        return [loss], [loss_m]
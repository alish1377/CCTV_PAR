import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight


@LOSSES.register("scaledbceloss")
class ScaledBCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=30, tb_writer=None):
        super(ScaledBCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None
        self.tb_writer = tb_writer

    def forward(self, logits, targets, pos_scale, neg_scale):
        logits = logits[0]
        batch_size = logits.shape[0]
        device = logits.device  # Get device from input


        pos_coef = neg_scale / (pos_scale)
        neg_coef = pos_scale / (neg_scale)
        # pose_coef, neg_coef = 1, 1
        pos_coef = torch.tensor(pos_coef, device=device, dtype=logits.dtype)
        neg_coef = torch.tensor(neg_coef, device=device, dtype=logits.dtype)
        logits = logits * targets * pos_coef + logits * (1 - targets) * neg_coef

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        targets_mask = torch.where(
            targets.detach().cpu() > 0.5,
            torch.ones(1, device=device),
            torch.zeros(1, device=device)
        )

        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask.cpu(), self.sample_weight)
            loss_m = (loss_m * sample_weight.to(device))


        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()

        return [loss], [loss_m]
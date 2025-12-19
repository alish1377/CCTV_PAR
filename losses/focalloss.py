import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight


@LOSSES.register("focalloss")
class Focalloss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=30, tb_writer=None, alpha=0.5, gamma=2):
        super(Focalloss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None
        self.tb_writer = tb_writer
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, pos_scale=1, neg_scale=1):
        logits = logits[0]
        batch_size = logits.shape[0]
        device = logits.device  # Get device from input

        pos_coef = neg_scale / (pos_scale)
        neg_coef = pos_scale / (neg_scale)
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

        # log(pt) = ylog(p) + (1-y)log(1-p)
        # (1-pt) ** gamma
        pt = torch.where(targets == 1, logits, 1 - logits)
        modulating_factor = (1 - pt) ** self.gamma

        # alpha_t = y * a_t + (1 - y) * (1 - a_t)
        alpha_factor = torch.where(targets==1, self.alpha, 1 - self.alpha)

        # compute focal_loss
        focal_loss = alpha_factor * modulating_factor * loss_m

        focal_loss = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()

        return [focal_loss], [loss_m]
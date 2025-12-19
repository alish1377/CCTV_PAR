import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight


def calculate_new_loss(logits, targets):
    """
    Calculate the new loss for the given logits and targets.

    Args:
        logits: Tensor of shape [N, num_classes] containing predicted logits.
        targets: Tensor of shape [N, num_classes] containing target labels.

    Returns:
        new_loss: Scalar tensor representing the calculated loss.
    """
    # Gender target index (assumed to be the 4th index for "Gender_female")
    gender_index = 3

    # Calculate categorical cross-entropy for Age
    loss_age = F.cross_entropy(logits[:, 0:3], torch.argmax(targets[:, 0:3], dim=1))

    # Binary cross-entropy for Gender
    loss_gender = F.binary_cross_entropy_with_logits(logits[:, gender_index], targets[:, gender_index])

    # Mask for male (1 - t[gender])
    male_mask = 1 - targets[:, gender_index]

    # Categorical cross-entropy for Hair (applies only for males)
    loss_hair = F.cross_entropy(
        logits[:, 4:7], torch.argmax(targets[:, 4:7], dim=1), reduction='none'
    )
    loss_hair = (male_mask * loss_hair).mean()

    # Binary cross-entropy for Upper short (applies only for males)
    loss_upper_short = F.binary_cross_entropy_with_logits(
        logits[:, 7], targets[:, 7], reduction='none'
    )
    loss_upper_short = (male_mask * loss_upper_short).mean()

    # Binary cross-entropy for other binary classes (8:32 and 32:37)
    loss_binary_1 = F.binary_cross_entropy_with_logits(
        logits[:, 8:32], targets[:, 8:32], reduction='mean'
    )
    loss_binary_2 = F.binary_cross_entropy_with_logits(
        logits[:, 32:37], targets[:, 32:37], reduction='mean'
    )

    # Combine all losses
    new_loss = (
            loss_age
            + loss_gender
            + loss_hair
            + loss_upper_short
            + loss_binary_1
            + loss_binary_2
    )

    return new_loss

@LOSSES.register("myloss")
class MyLoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=30, tb_writer=None):
        super(MyLoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None
        self.tb_writer = tb_writer


    def forward(self, logits, targets):
        logits = logits[0]
        batch_size = logits.shape[0]

        # pose_coef = neg_scale / (pose_scale)
        # neg_coef = pose_scale / (neg_scale)
        # # pose_coef, neg_coef = 1, 1
        # pose_coef, neg_coef = torch.tensor(pose_coef, device='cuda'), torch.tensor(neg_coef, device='cuda')
        # logits = logits * targets * pose_coef + logits * (1 - targets) * neg_coef
        #
        # if self.smoothing is not None:
        #     targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss = calculate_new_loss(logits, targets)

        # targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        #
        # if self.sample_weight is not None:
        #     sample_weight = ratio2weight(targets_mask, self.sample_weight)
        #
        #     loss_m = (loss_m * sample_weight.cuda())
        #
        # loss = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()

        return [loss], [None]
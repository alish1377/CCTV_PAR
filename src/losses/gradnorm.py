import torch
import torch.nn as nn
import torch.optim as optim

from models.registry import LOSSES
from tools.function import ratio2weight

@LOSSES.register("gradnormloss")
class GradNormLoss(nn.Module):
    def __init__(self, sample_weight, num_of_task=37, alpha=1.5, optimizer=None):
        """
        GradNorm loss implementation for multi-task classification problems.
        Args:
            num_of_task: Total number of tasks (attributes to classify).
            alpha: GradNorm hyperparameter for controlling task asymmetry.
        """
        super(GradNormLoss, self).__init__()
        self.num_of_task = num_of_task  # Total number of tasks
        self.alpha = alpha  # Alpha value for relative task rate adjustment
        self.sample_weight = sample_weight
        self.w = nn.Parameter(torch.ones(num_of_task, dtype=torch.float))  # Task-specific weights
        self.l1_loss = nn.L1Loss()  # L1 Loss for GradNorm regularization
        self.L_0 = None  # Reference for initial task losses

    def forward(self, logits, targets, pos_scale, neg_scale):
        """
        Compute the total weighted loss for multi-task classification.
        Args:
            logits: Predicted logits for each task (shape: [batch_size, num_of_task]).
            targets: Ground truth labels for each task (shape: [batch_size, num_of_task]).
        Returns:
            total_loss: Weighted total loss across all tasks.
        """
        
        logits = logits[0]
        batch_size = logits.shape[0]
        device = logits.device  # Get device from input

        pos_coef = neg_scale / (pos_scale)
        neg_coef = pos_scale / (neg_scale)
        # pose_coef, neg_coef = 1, 1
        pos_coef = torch.tensor(pos_coef, device=device, dtype=logits.dtype)
        neg_coef = torch.tensor(neg_coef, device=device, dtype=logits.dtype)

        logits = logits * targets * pos_coef + logits * (1 - targets) * neg_coef

        # Compute individual binary cross-entropy losses for each task
        task_losses = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )  # Loss per task, averaged across the batch
        # print("task_losses shape1")
        # print(task_losses.shape)

        targets_mask = torch.where(
            targets.detach() > 0.5,
            torch.ones(1, device=device),
            torch.zeros(1, device=device)
        )

        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask.detach().cpu(), self.sample_weight)
            task_losses = (task_losses * sample_weight.to(device))

        task_losses = task_losses.mean(dim=0)
        # print("task_losses shape2")
        # print(task_losses.shape)
        # print("task losses")
        # print(task_losses)

        # Initialize the initial task losses if not already set
        if self.L_0 is None:
            self.L_0 = task_losses.detach()  # Detach to prevent gradient flow

        # Compute the weighted losses: w_i(t) * L_i(t)
        weighted_task_losses = task_losses * self.w
        total_loss = weighted_task_losses.sum()  # Sum of weighted losses

        # Store intermediate variables for GradNorm updates
        self.task_losses = task_losses
        self.weighted_task_losses = weighted_task_losses
        self.total_loss = total_loss
        # print("self.total_loss")
        # print(self.total_loss)
        return [self.total_loss], None

    def additional_forward_and_backward(self, model: nn.Module, optimizer: optim.Optimizer):
        """
        Perform GradNorm-specific updates for task weights.
        Args:
            model: The multitask model to compute gradients for shared parameters.
            optimizer: Optimizer for updating the task weights `w`.
        """
        # Backward pass for the total loss
        self.total_loss.backward(retain_graph=True)

        # Reset gradients for task weights
        self.w.grad.zero_()

        # Compute gradient norms for each task
        # print("weighted_task_losses shape:")
        # print(self.weighted_task_losses.shape)
        # x = [torch.autograd.grad(self.weighted_task_losses[i], model.parameters(), retain_graph=True, create_graph=True) for i in range(self.num_of_task)]
        # print(x)
        GW_t = [
            torch.norm(torch.autograd.grad(
                self.weighted_task_losses[i].sum(), model.parameters(), retain_graph=True, create_graph=True)[0])
            for i in range(self.num_of_task)
        ]
        GW_t = torch.stack(GW_t)  # Gradient norms as a tensor

        # Average gradient norm
        bar_GW_t = GW_t.mean().detach()

        # Compute relative training rates and desired gradient norms
        tilde_L_t = (self.task_losses / self.L_0).detach()
        r_t = tilde_L_t / tilde_L_t.mean()
        desired_GW_t = bar_GW_t * (r_t ** self.alpha)

        # Compute GradNorm loss
        grad_loss = self.l1_loss(GW_t, desired_GW_t)

        # Backward pass for GradNorm loss to update task weights
        self.w.grad = torch.autograd.grad(grad_loss, self.w, only_inputs=True)[0]
        optimizer.step()

        # Re-normalize task weights to ensure their sum equals the number of tasks
        self.w.data = self.w.data / self.w.data.sum() * self.num_of_task

        # Clear intermediate variables to save memory
        del GW_t, bar_GW_t, tilde_L_t, r_t, desired_GW_t
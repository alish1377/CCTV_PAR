import torch

def compute_mean_accuracy(ground_truth: torch.Tensor, prediction: torch.Tensor) -> float:
    """
    Computes the mean accuracy for a binary classification problem with multiple labels.

    Args:
        ground_truth (torch.Tensor): Tensor of shape [1, 11] containing 0 or 1.
        prediction (torch.Tensor): Tensor of shape [1, 11] containing probabilities between 0 and 1.

    Returns:
        float: Mean accuracy (between 0 and 1).
    """
    # Apply threshold to get predicted labels
    predicted_labels = (prediction >= 0.5).int()
    
    # Compute correct predictions
    correct = (predicted_labels == ground_truth.int()).sum()
    
    # Compute mean accuracy
    accuracy = correct.item() / ground_truth.numel()
    
    return accuracy

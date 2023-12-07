import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        """
        Initializes the Dice Loss.
        """
        super().__init__()

    def forward(self, preds, targets):
        """
        Calculates the Dice Loss between predicted and target segmentation masks.

        Args:
            preds (torch.Tensor): Predicted segmentation masks.
            targets (torch.Tensor): Target segmentation masks.

        Returns:
            torch.Tensor: Dice Loss.
        """
        intersection = (preds * targets).sum()  
        dice = (2 * intersection) / (preds.sum() + targets.sum())
        return 1 - dice

def calculate_confidence(preds, threshold=0.8):
    """
    Calculates prediction confidence scores and identifies low-confidence predictions.

    Args:
        preds (torch.Tensor): Model predictions.
        threshold (float): Confidence threshold.

    Returns:
        torch.Tensor: Confidence scores.
        torch.Tensor: Predicted classes.
        torch.Tensor: Indices of low-confidence predictions.
    """
    prob = torch.nn.functional.softmax(preds, dim=1)
    confidence, predictions = torch.max(prob, dim=1)
    low_confidence_idx = torch.where(confidence < threshold)
    
    return confidence, predictions, low_confidence_idx[0]

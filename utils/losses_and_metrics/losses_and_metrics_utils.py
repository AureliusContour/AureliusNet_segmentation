import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff

class DiceLoss(nn.Module):
    def __init__(self):
        """
        Initializes the Dice Loss.
        """
        super(DiceLoss, self).__init__()

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
        # Return Dice Loss
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
	prob = F.softmax(preds, dim=1)
	confidence, predictions = torch.max(prob, dim=1)
	# Identify low-confidence predictions
	low_confidence_idx = torch.where(confidence < threshold)

	return confidence, predictions, low_confidence_idx[0]

def dice_score(predicted_masks, true_masks, epsilon=1e-8):
    """
    Calculate the Dice coefficient for batches of binary masks.

    Args:
    - predicted_masks (torch.Tensor): Predicted masks.
    - true_masks (torch.Tensor): True binary masks.
    - epsilon (float): A small constant to avoid division by zero.

    Returns:
    - torch.Tensor: Dice coefficients for each mask in the batch.
    """
    predicted_masks = torch.clamp(predicted_masks, 0, 1)
    # Calculate intersection and union for Dice coefficient
    intersection = torch.sum(predicted_masks * true_masks, dim=(2, 3))
    union = torch.sum(predicted_masks, dim=(2, 3)) + torch.sum(true_masks, dim=(2, 3)) + epsilon
    # Calculate Dice coefficient for each mask in the batch
    dice = (2.0 * intersection) / union
    return dice

def hausdorff_distance(preds, targets):
    """
    Calculate the Hausdorff distance between predicted and target binary masks.

    Args:
        preds (torch.Tensor): Predicted binary mask.
        targets (torch.Tensor): Target binary mask.

    Returns:
        float: Hausdorff distance between the contours of the predicted and target masks.
    """
    # Get coordinates of non-zero elements in predicted and target masks
    pred_coords = torch.stack(torch.where(preds > 0.5), dim=1).float()
    target_coords = torch.stack(torch.where(targets > 0.5), dim=1).float()

    # Calculate Hausdorff distance in both directions
    hausdorff_dist_forward = directed_hausdorff(pred_coords.numpy(), target_coords.numpy())[0]
    hausdorff_dist_backward = directed_hausdorff(target_coords.numpy(), pred_coords.numpy())[0]

    # Choose the maximum Hausdorff distance
    hausdorff_dist = max(hausdorff_dist_forward, hausdorff_dist_backward)
    return hausdorff_dist


def mean_iou(predicted_masks, true_masks, epsilon=1e-8):
    """
    Calculate the Mean Intersection over Union (mIOU) for batches of binary masks.

    Args:
        predicted_masks (torch.Tensor): Predicted binary masks.
        true_masks (torch.Tensor): True binary masks.
        epsilon (float, optional): A small constant to avoid division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: Mean IOU for the batch.
    """
    # Ensure predicted masks are within the range [0, 1]
    predicted_masks = torch.clamp(predicted_masks, 0, 1)
    # Calculate intersection and union for each mask
    intersection = torch.sum(predicted_masks * true_masks, dim=(2, 3))
    union = torch.sum(predicted_masks, dim=(2, 3)) + torch.sum(true_masks, dim=(2, 3)) - intersection + epsilon
    # Calculate IOU for each mask
    iou = intersection / union
    # Return mean IOU for the batch
    return torch.mean(iou)


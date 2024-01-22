import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff

class DiceSimilarityCoefficient(nn.Module):
	def __init__(self, reduction:str="mean"):
		"""
		Initializes the Dice Score (Dice Similarity Coefficient).
		Params:
			1. reduction:str = ["mean", "sum", "median"]. Defaults to "mean".
		"""
		super(DiceSimilarityCoefficient, self).__init__()
		if reduction == "mean":
			self.reduce = torch.mean
		elif reduction == "sum":
			self.reduce = torch.sum
		elif reduction == "median":
			self.reduce = torch.median
		else:
			print("Invalid reduction, defaulting to mean.")
			self.reduce = torch.mean

	def forward(self, preds, targets):
		"""
		Calculate the Dice coefficient for batches of binary masks.

		Args:
		- predicted_masks (torch.Tensor): Predicted masks.
		- true_masks (torch.Tensor): True binary masks.
		- epsilon (float): A small constant to avoid division by zero.

		Returns:
		- torch.Tensor: Dice coefficients for each mask in the batch.
		"""

		# Calculate intersection and union for Dice coefficient
		intersection = torch.sum(preds * targets, dim=(2, 3))
		union = torch.sum(preds, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))
		# Calculate Dice coefficient for each mask in the batch
		dice = (2.0 * intersection + 1.0) / (union + 1.0)
		return self.reduce(dice)
	
      
class DiceLoss(nn.Module):
	def __init__(self, reduction:str="mean"):
		"""
		Initializes the Dice Loss.		
		Params:
			1. reduction:str = ["mean", "sum", "median"]. Defaults to "mean".
		"""
		super(DiceLoss, self).__init__()
		self.dice = DiceSimilarityCoefficient(reduction=reduction)

	def forward(self, preds, targets):
		"""
		Calculates the Dice Loss between predicted and target segmentation masks.

		Args:
			preds (torch.Tensor): Predicted segmentation masks.
			targets (torch.Tensor): Target segmentation masks.

		Returns:
			torch.Tensor: Dice Loss.
		"""
		return 1.0 - self.dice(preds, targets)


class HausdorffDistance(nn.Module):
	def __init__(self):
		"""
		Initializes the Hausdorff Distance.
		"""
		super(HausdorffDistance, self).__init__()

	def forward(self, preds, targets):
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


class IOU(nn.Module):
	def __init__(self, reduction:str="mean"):
		"""
		Initializes the Intersection over Union (Jaccard Index).
		Params:
			1. reduction:str = ["mean", "sum", "median"]. Defaults to "mean".
		"""
		super(IOU, self).__init__()
		if reduction == "mean":
			self.reduce = torch.mean
		elif reduction == "sum":
			self.reduce = torch.sum
		elif reduction == "median":
			self.reduce = torch.median
		else:
			print("Invalid reduction, defaulting to mean.")
			self.reduce = torch.mean

	def forward(self, preds, targets, epsilon=1e-8):
		"""
		Calculate the Reduced(Mean) Intersection over Union (mIOU) for batches of binary masks.

		Args:
			predicted_masks (torch.Tensor): Predicted binary masks.
			true_masks (torch.Tensor): True binary masks.
			epsilon (float, optional): A small constant to avoid division by zero. Defaults to 1e-8.

		Returns:
			torch.Tensor: Reduced IOU for the batch.
		"""
		# Calculate intersection and union for each mask
		intersection = torch.sum(preds * targets, dim=(2, 3))
		union = torch.sum(preds, dim=(2, 3)) + torch.sum(targets, dim=(2, 3)) - intersection
		# Calculate IOU for each mask
		iou = (intersection + 1.0) / (union + 1.0)
		
		# Return reduced scalar value for the batch
		return self.reduce(iou)
			

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


#active_learning/sampling.py
"""
sampling.py implements the key functionality of using the model to predict a dataset,
analyzing the prediction confidence scores to find uncertain samples that are below the configured confidence threshold
, and returning the indices of those uncertain samples.
"""


import torch
from torchvision import transforms
from .losses_and_metrics.losses_and_metrics_utils import calculate_confidence
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_uncertain_masks(model, dataloader, threshold=0.80):
    uncertain_indices = []
    for i, (inputs, _, _) in enumerate(dataloader):
        inputs = inputs.to(device)  # Assuming device is defined
        outputs = model(inputs)
        confidence, _, low_confidence_idx = calculate_confidence(outputs, threshold)
        uncertain_indices.extend(low_confidence_idx.tolist())
    
    # Log the number of uncertain samples
    logger.info(f"Number of uncertain samples identified: {len(uncertain_indices)}")
    return uncertain_indices

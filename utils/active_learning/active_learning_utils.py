# utils/active_learning/active_learning_utils.py
"""
This code ties together the active_learning_pipeline's different components of the active_learning_pipeline; sampling, intervention, augmenting, and retraining. 
"""

import torch
from utils.losses_and_metrics import calculate_confidence
from .components import sample_uncertain_masks
from .components import initiate_human_intervention
from .components import augment_and_store_data
from .components import retraining_pipeline, automated_model_retraining

def active_learning_pipeline(model, dataloader, dataset, threshold=0.80, num_epochs=5):
    # Step 1: Sampling Uncertain Masks
    uncertain_indices = sample_uncertain_masks(model, dataloader, threshold)

    # Step 2: Initiating Human Intervention
    radiologist_segmentations = initiate_human_intervention(uncertain_indices, dataset)

    # Step 3: Augmenting and Storing Data
    augment_and_store_data(dataset, radiologist_segmentations)

    # Step 4: Retraining Pipeline
    retraining_pipeline(model, dataset)

    # Step 5: Automated Model Retraining
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    automated_model_retraining(model, train_loader, num_epochs=num_epochs)

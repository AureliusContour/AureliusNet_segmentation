# utils/active_learning/intervention.py
"""
This code prompts the radiologists to provide labels for uncertain mask indices. 
"""
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initiate_human_intervention(uncertain_indices, dataset):
    #begin time 
    start_time = time.time() 
    # Implement the logic to prompt radiologist intervention
    # Provide tools for visualization and editing of uncertain sections
    # Return radiologist-provided segmentations

    #end time
    end_time = time.time()
    duration = end_time - start_time
    num_samples = len(uncertain_indices)
    
    
    # Log the intervention initiation details -time spent, number of labels provided
    logger.info("Human intervention complete", extra={
        "duration": duration,  
        "num_samples": num_samples
    })


#code for later log analysis below

"""
import json

for event in logged_events:
    data = json.loads(event.getMessage())
    print(f"Duration: {data['duration']}, Samples: {data['num_samples']}")
"""

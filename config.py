import torch
import os

# Dataset base path
DATASET_PATH = os.path.join("datasets", "train")

# Dataset filenames split csv
DATASET_FILENAMES_SPLIT = os.path.join(DATASET_PATH, "filenames_split.csv")

# Device used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
PIN_MEMORY = (DEVICE == "cuda" or DEVICE == "mps")

# Initial learning rate, number of epoch and batch size
INIT_LR = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 32

# Input Image dimensions
INPUT_IMAGE_WIDTH = 512
INPUT_IMAGE_HEIGHT = 512

# Thresholds
SEGMENTATION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD = 0.75

# Output
BASE_OUTPUT_PATH = "output"

# define the path to the output weights and runs
OUTPUT_WEIGHTS_PATH = os.path.join(BASE_OUTPUT_PATH, "weights")
OUTPUT_RUNS_PATH = os.path.join(BASE_OUTPUT_PATH, "runs")

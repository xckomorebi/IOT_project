import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]

DATA_PATH = os.path.join(PROJECT_ROOT, "data")

DATA_RAW_PATH = os.path.join(DATA_PATH, "raw")

DATA_SUBSAMPLED_PATH = os.path.join(DATA_PATH, "subsampled")

DATA_FEATURE_PATH = os.path.join(DATA_PATH, "feature")

# TODO change to relative path
EVALUATION_VIDEO_PATH = "/content/drive/MyDrive/IoT/IOT Classification Challenge/Evaluation_Dataset/"
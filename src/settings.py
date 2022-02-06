import os

PROJECT_ROOT = os.getenv("IOT_PROJECT_ROOT", "/content/drive/MyDrive/repo/IOT_project")

DATA_PATH = os.path.join(PROJECT_ROOT, "data")

DATA_RAW_PATH = os.path.join(DATA_PATH, "raw")

DATA_SUBSAMPLED_PATH = os.path.join(DATA_PATH, "subsampled")

DATA_FEATURE_PATH = os.path.join(DATA_PATH, "feature")
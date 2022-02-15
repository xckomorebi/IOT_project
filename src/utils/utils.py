import os


def check_path(in_dir):
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)

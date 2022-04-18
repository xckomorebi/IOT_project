import os
import cv2

from glob import glob
from typing import Tuple

from src.settings import *

class BaseSubSample:
    @classmethod
    def initialize_dataset(cls, dataset=DATA_PATH):
        result = [y for x in os.walk(dataset)
                  for y in glob(os.path.join(x[0], '*.mp4'))]
        labels = [r.split('/')[8] for r in result]
        return list(zip(labels, result))

    @classmethod
    def convert(cls, outputs):
        # output_dir = '/content/drive/My Drive/IoT/sd/subsampled videos'
        outputs = outputs
        dataset = cls.initialize_dataset()

        for label, path in dataset:
            cls.atomic_convert(label, path)
            
    @classmethod
    def atomic_convert(cls, label, path, outputs):
        filename = path.split('/')[-1]
        output_path = outputs + '/' + label + '/' + filename

        cap = cv2.VideoCapture(path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                            fps, (frame_width, frame_height))

        subsample_idx = cls.get_index()

        for i in range(length):
            if i in range(subsample_idx[0] - 1, subsample_idx[1]):
                ret, frame = cap.read()
                out.write(frame)
            else:
                continue

        if length < 100:  # for videos shorter than 100 frames, duplicate the last frame
            for i in range(100 - length):
                out.write(frame)

        cap.release()
        out.release()

    @classmethod
    def get_index(cls, frames: int = 100) -> Tuple:
        subsample_idx = (max(1, int(frames / 2) - 49),
                        min(frames, int(frames / 2) + 50))
        return subsample_idx

    @classmethod
    def run(cls, inputs, outputs) -> None:
        raise NotImplementedError
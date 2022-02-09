from glob import glob
import os
import torch
from torch.utils import data
import numpy as np


def extract_trajectories(keypoints, with_index=False):
    """
    :param keypoints: keypoints from npz file
    :param with_index: whether to record indexes of frames that have features
    :return: videos keypoints(length, 3, 17) and indexes of frames(if with_index=True)
    """
    trajectory = []
    index = []
    for i, (_, k) in enumerate(keypoints):
        if len(k) != 0:
            index.append(i)
            three_d_point = k[0,[0,1,3],:]
            trajectory.append(three_d_point)

        if with_index:
            return np.stack(trajectory), index
        else:
            return np.stack(trajectory)

class IOTDataset(data.Dataset):
    def __init__(self, feature_path):
        self.Data, self.Label, self.frame_indexes = self._fetch_feature(feature_path)

    def _fetch_feature(self, feature_path):
        features = []
        frame_indices = []
        labels = []
        self.__label_encoder = {'no_interaction':0,
                                'open_close_fridge':1,
                                'put_back_item':2,
                                'screen_interaction':3,
                                'take_out_item':4}

        feature_result = [y for x in os.walk(feature_path) for y in glob(os.path.join(x[0], "*.npz"))]
        for path in feature_result:
            label = path.split('/')[-2]  #may be hard code
            labels.append(self.__label_encoder[label])
            data = np.load(path, allow_pickle=True)
            if len(data['keypoints'].shape) != 2:
                print(path)
                continue
            traj, index = extract_trajectories(data['keypoints'], with_index=True)
            features.append(traj)
            frame_indices.append(index)

        return features, labels, frame_indices

    def __getitem__(self, index):
        seq = torch.from_numpy(self.Data[index])
        label = torch.tensor(self.Label[index])
        return seq, label

    def __len__(self):
        return len(self.Data)


def collate_fn(batch):
    seq_lists = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    labels = torch.LongTensor(labels)
    return seq_lists, labels
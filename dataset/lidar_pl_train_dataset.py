import torch
import pickle
import random
import numpy as np
from copy import deepcopy
from itertools import chain

from dataset.reference_dataset import ReferenceDataset

class LiDARAndPseudoLabeledTrainingDataset(ReferenceDataset):
    def __init__(self, data_path, data_path_pl, data_path_ref, transform=None, transform_pl=None, transform_ref=None, 
                 split='train', split_ref='train', ratio=1, ratio_ref=1):
        super().__init__(data_path, data_path_ref, transform, transform_ref, split, split_ref, ratio, ratio_ref)

        with open(data_path_pl, 'rb') as f:
            self.all_data_pl = pickle.load(f)

        self.transform_pl = transform_pl
        self.split_pl = self.split_[int(len(self.split_) * ratio):]
        self.data_pl = [self.all_data_pl['sequences'][i] for i in self.split_pl]
        self.seq_lens_pl = [len(seq['keypoints']) for seq in self.data_pl]
        self.len_pl = int(np.sum(self.seq_lens_pl))

    def __getitem__(self, idx):
        sample, sample_ref = super().__getitem__(idx)

        idx_pl = random.randint(0, self.len_pl - 1)
        seq_idx_pl = 0
        global_idx_pl = idx_pl
        while idx_pl >= self.seq_lens_pl[seq_idx_pl]:
            idx_pl -= self.seq_lens_pl[seq_idx_pl]
            seq_idx_pl += 1
        sample_pl = deepcopy(self.data_pl[seq_idx_pl])

        sample_pl['dataset_name'] = self.data_path.split('/')[-1].split('.')[0]
        sample_pl['sequence_index'] = seq_idx_pl
        sample_pl['global_index'] = global_idx_pl
        sample_pl['index'] = idx_pl
        sample_pl['centroid'] = np.array([0.,0.,0.])
        sample_pl['radius'] = 1.
        sample_pl['scale'] = 1.
        sample_pl['translate'] = np.array([0.,0.,0.])
        sample_pl['rotation_matrix'] = np.eye(3)

        sample_pl = self.transform_pl(sample_pl)

        return sample, sample_pl, sample_ref
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        keys = ['point_clouds', 'keypoints', 'centroid', 'radius', 'sequence_index', 'index', 'global_index']
        keys_pl = keys.copy()
        keys_ref = keys.copy()
        
        for key in keys:
            batch_data[key] = torch.stack([sample[0][key] for sample in batch], dim=0)
        for key in keys_pl:
            batch_data[key+'_pl'] = torch.stack([sample[1][key] for sample in batch], dim=0)
        for key in keys_ref:
            batch_data[key+'_ref'] = torch.stack([sample[2][key] for sample in batch], dim=0)

        return batch_data
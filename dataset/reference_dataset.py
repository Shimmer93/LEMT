import torch
import pickle
import random
import numpy as np
from copy import deepcopy
from itertools import chain

from dataset.base_dataset import BaseDataset

class ReferenceDataset(BaseDataset):
    def __init__(self, data_path, data_path_ref, transform=None, transform_ref=None, 
                 split='train', split_ref='train', ratio=1, ratio_ref=1):
        super().__init__(data_path, transform, split, ratio)
        
        self.data_path_ref = data_path_ref
        self.transform_ref = transform_ref

        with open(data_path_ref, 'rb') as f:
            self.all_data_ref = pickle.load(f)

        if isinstance(split_ref, str):
            self.split_ref_ = self.all_data_ref['splits'][split_ref]
        elif isinstance(split_ref, list):
            self.split_ref_ = [self.all_data_ref['splits'][s] for s in split_ref]
            self.split_ref_ = list(chain(*self.split_ref_))

        self.split_ref = self.split_ref_[:int(len(self.split_ref_) * ratio_ref)]

        self.data_ref = [self.all_data_ref['sequences'][i] for i in self.split_ref]
        self.seq_lens_ref = [len(seq['keypoints']) for seq in self.data_ref]
        self.len_ref = np.sum(self.seq_lens_ref)

    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        idx_ref = random.randint(0, self.len_ref - 1)
        seq_idx_ref = 0
        global_idx_ref = idx_ref
        while idx_ref >= self.seq_lens_ref[seq_idx_ref]:
            idx_ref -= self.seq_lens_ref[seq_idx_ref]
            seq_idx_ref += 1
        sample_ref = deepcopy(self.data_ref[seq_idx_ref])

        sample_ref['dataset_name'] = self.data_path_ref.split('/')[-1].split('.')[0]
        sample_ref['sequence_index'] = seq_idx_ref
        sample_ref['global_index'] = global_idx_ref
        sample_ref['index'] = idx_ref
        sample_ref['centroid'] = np.array([0.,0.,0.])
        sample_ref['radius'] = 1.
        sample_ref['scale'] = 1.
        sample_ref['translate'] = np.array([0.,0.,0.])
        sample_ref['rotation_matrix'] = np.eye(3)

        sample_ref = self.transform_ref(sample_ref)

        return sample, sample_ref
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        keys = ['point_clouds', 'keypoints', 'centroid', 'radius', 'sequence_index', 'index', 'global_index']
        keys_ref = keys.copy()
        
        for key in keys:
            batch_data[key] = torch.stack([sample[0][key] for sample in batch], dim=0)
        for key in keys_ref:
            batch_data[key+'_ref'] = torch.stack([sample[1][key] for sample in batch], dim=0)

        return batch_data
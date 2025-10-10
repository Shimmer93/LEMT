import torch
import pickle
import random
import numpy as np
from copy import deepcopy
from itertools import chain

from dataset.reference_dataset import ReferenceDataset

class LiDARAssistedPseudoLabelingDataset(ReferenceDataset):
    def __init__(self, data_path, data_path_ref, transform=None, transform_unsup=None, transform_ref=None, 
                 split='train', split_ref='train', ratio=1, ratio_ref=1, use_ref=False):
        super().__init__(data_path, data_path_ref, transform, transform_ref, split, split_ref, ratio, ratio_ref)

        self.use_ref = use_ref
        self.transform_unsup = transform_unsup

        if use_ref:
            self.split_unsup = self.split_ref_[int(len(self.split_ref_) * ratio_ref):]
            self.data_unsup = [self.all_data_ref['sequences'][i] for i in self.split_unsup]
        else:
            self.split_unsup = self.split_[int(len(self.split_) * ratio):]
            self.data_unsup = [self.all_data['sequences'][i] for i in self.split_unsup]
        self.seq_lens_unsup = [len(seq['keypoints']) for seq in self.data_unsup]
        self.len_unsup = int(np.sum(self.seq_lens_unsup))

    def __getitem__(self, idx):
        sample, sample_ref = super().__getitem__(idx)

        idx_unsup = random.randint(0, self.len_unsup - 1)
        seq_idx_unsup = 0
        global_idx_unsup = idx_unsup
        while idx_unsup >= self.seq_lens_unsup[seq_idx_unsup]:
            idx_unsup -= self.seq_lens_unsup[seq_idx_unsup]
            seq_idx_unsup += 1
        sample_unsup = deepcopy(self.data_unsup[seq_idx_unsup])

        sample_unsup['dataset_name'] = self.data_path_ref.split('/')[-1].split('.')[0] \
            if self.use_ref else self.data_path.split('/')[-1].split('.')[0]
        sample_unsup['sequence_index'] = seq_idx_unsup
        sample_unsup['global_index'] = global_idx_unsup
        sample_unsup['index'] = idx_unsup
        sample_unsup['centroid'] = np.array([0.,0.,0.])
        sample_unsup['radius'] = 1.
        sample_unsup['scale'] = 1.
        sample_unsup['translate'] = np.array([0.,0.,0.])
        sample_unsup['rotation_matrix'] = np.eye(3)

        sample_unsup = self.transform_unsup(sample_unsup)

        return sample, sample_unsup, sample_ref
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        keys = ['point_clouds', 'keypoints', 'centroid', 'radius', 'sequence_index', 'index', 'global_index']
        keys_unsup = keys.copy()
        keys_ref = keys.copy()
        
        for key in keys:
            batch_data[key] = torch.stack([sample[0][key] for sample in batch], dim=0)
        for key in keys_unsup:
            batch_data[key+'_unsup'] = torch.stack([sample[1][key] for sample in batch], dim=0)
        for key in keys_ref:
            batch_data[key+'_ref'] = torch.stack([sample[2][key] for sample in batch], dim=0)

        return batch_data
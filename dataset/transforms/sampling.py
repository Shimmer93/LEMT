import numpy as np
from copy import deepcopy

class UniformSample():
    def __init__(self, clip_len, pad_type='both', offset=0):
        self.clip_len = clip_len
        if pad_type not in ['both', 'start', 'end']:
            raise ValueError('pad_type must be "both" or "start" or "end"')
        if pad_type == 'both':
            assert clip_len % 2 == 1, 'num_frames must be odd'
            self.pad = (clip_len - 1) // 2
        else:
            self.pad = clip_len - 1
        self.pad_type = pad_type
        self.offset = offset

    def __call__(self, sample):
        if self.pad_type == 'both':
            for _ in range(self.pad):
                if 'point_clouds' in sample:
                    sample['point_clouds'].insert(0, sample['point_clouds'][0])
                    sample['point_clouds'].append(sample['point_clouds'][-1])
                if 'keypoints' in sample:
                    sample['keypoints'] = np.concatenate([sample['keypoints'][0][np.newaxis], sample['keypoints']], axis=0)
                    sample['keypoints'] = np.concatenate([sample['keypoints'], sample['keypoints'][-1][np.newaxis]], axis=0)

        elif self.pad_type == 'start':
            for _ in range(self.pad):
                if 'point_clouds' in sample:
                    sample['point_clouds'].insert(0, sample['point_clouds'][0])
                if 'keypoints' in sample:
                    sample['keypoints'] = np.concatenate([sample['keypoints'][0][np.newaxis], sample['keypoints']], axis=0)

        elif self.pad_type == 'end':
            for _ in range(self.pad):
                if 'point_clouds' in sample:
                    sample['point_clouds'].append(sample['point_clouds'][-1])
                if 'keypoints' in sample:
                    sample['keypoints'] = np.concatenate([sample['keypoints'], sample['keypoints'][-1][np.newaxis]], axis=0)

        for _ in range(self.offset):
            if 'point_clouds' in sample:
                sample['point_clouds'].append(sample['point_clouds'][-1])
            if 'keypoints' in sample:
                sample['keypoints'] = np.concatenate([sample['keypoints'], sample['keypoints'][-1][np.newaxis]], axis=0)

        start_idx = sample['index'] + self.offset
        if 'point_clouds' in sample:
            sample['point_clouds'] = sample['point_clouds'][start_idx:start_idx+self.clip_len]
        if 'keypoints' in sample:
            sample['keypoints'] = sample['keypoints'][start_idx:start_idx+self.clip_len]

        return sample
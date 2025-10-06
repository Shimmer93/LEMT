import numpy as np
import torch
from copy import deepcopy

class ConvertToMMWavePointCloud():
    def __init__(self, max_dist_threshold=0.1, add_std=0.1, default_num_points=32, num_noisy_points=32):
        self.max_dist_threshold = max_dist_threshold
        self.add_std = add_std
        self.default_num_points = default_num_points
        self.num_noisy_points = num_noisy_points

    def __call__(self, sample):
        if isinstance(sample['keypoints'], list):
            sample['keypoints'] = np.stack(sample['keypoints'])

        kps0 = sample['keypoints'][:-1]
        kps1 = sample['keypoints'][1:]
        kps_dist = np.linalg.norm(kps0 - kps1, axis=-1)
        dist_threshold = self.max_dist_threshold
        mask = kps_dist > dist_threshold

        new_pcs = []
        for i in range(len(sample['keypoints']) - 1):
            pc0 = sample['point_clouds'][i]
            
            num_points = 0
            new_pc = []
            for j in range(sample['keypoints'].shape[1]):
                pc0_j = pc0[sample['point_clouds'][i][..., -1] == j+1]
                if mask[i, j]:
                    new_pc.append(pc0_j)
                    num_points += len(pc0_j)

            if num_points == 0:
                random_idxs = np.random.choice(pc0.shape[0], self.default_num_points)
                new_pc.append(pc0[random_idxs])
            new_pc = np.concatenate(new_pc)
            new_pcs.append(new_pc)

        sample['point_clouds'] = new_pcs
        sample['keypoints'] = sample['keypoints'][:-1]
        return sample
    
class FlowBasedPointFiltering():
    def __init__(self, max_dist_threshold=0.1, min_dist_threshold=0.05, add_std=0.1, default_num_points=32, num_noisy_points=32):
        self.max_dist_threshold = max_dist_threshold
        self.min_dist_threshold = min_dist_threshold
        self.add_std = add_std
        self.default_num_points = default_num_points
        self.num_noisy_points = num_noisy_points

    def __call__(self, sample):
        if isinstance(sample['keypoints'], list):
            sample['keypoints'] = np.stack(sample['keypoints'])

        keypoints = sample['keypoints']
        point_clouds = sample['point_clouds']

        T, J, _ = keypoints.shape
        # random number between min_dist_threshold and max_dist_threshold
        flow_thres = np.random.uniform(self.min_dist_threshold, self.max_dist_threshold)

        # 1. Calculate keypoint flow (temporal displacement)
        keypoint_flow = keypoints[1:] - keypoints[:-1]  # (T-1, J, 3)

        new_pcs = []
        for t in range(T-1):
            pc = point_clouds[t][:, :3]
            kp = keypoints[t][:, :3]  # (J, 3)
            kpf = keypoint_flow[t]
            # print(np.linalg.norm(kpf, axis=-1))
            pc_expanded = pc[:, np.newaxis, :]  # (N, 1, 3)
            kp_expanded = kp[np.newaxis, :, :]  # (1, J, 3)
            kpf_expanded = kpf[np.newaxis, :, :]  # (1, J, 3)

            # Calculate pairwise distances between points and keypoints
            pairwise_distances = np.linalg.norm(pc_expanded - kp_expanded, axis=-1)  # (N, J)

            # 3. Calculate estimated point cloud flow
            # Use inverse distance weighting to estimate flow for each point
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            weights = 1.0 / (pairwise_distances + eps)  # (N, J)
            weights_normalized = weights / (weights.sum(axis=-1, keepdims=True) + eps)  # (N, J)

            # Calculate weighted average of keypoint flows for each point
            weights_expanded = weights_normalized[:, :, np.newaxis]  # (N, J, 1)
            pcf = (weights_expanded * kpf_expanded).sum(axis=1)  # (N, 3)
            pcf = np.linalg.norm(pcf, axis=-1)  # (N,)
            prob = np.clip(pcf / flow_thres, 0, 1)  # (N,)
            # print(f'Point cloud flow at time {t}: {np.mean(pcf):.4f} ± {np.std(pcf):.4f}, min: {np.min(pcf):.4f}, max: {np.max(pcf):.4f}')

            pcf[pcf < 0.01] = 0.0  # Set very small flows to zero
            pc = np.concatenate([pc, pcf[:, np.newaxis]], axis=-1)  # (N, 4)

            # 4. Filter points based on flow threshold
            mask = np.random.rand(pc.shape[0]) < prob  # (N,)
            if np.any(mask):
                new_pc = pc[mask]
            else:
                # If no points are below the threshold, randomly sample points
                random_idxs = np.random.choice(pc.shape[0], self.default_num_points)
                new_pc = pc[random_idxs]
            new_pcs.append(new_pc)
    
        sample['point_clouds'] = new_pcs
        sample['keypoints'] = sample['keypoints'][1:]
        return sample
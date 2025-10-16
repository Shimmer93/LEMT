import numpy as np
import torch
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
from miniball import get_bounding_ball

class AddNoisyPoints():
    def __init__(self, add_std=0.01, num_added=32, zero_centered=True, center_range=1.5):
        self.add_std = add_std
        self.num_added = num_added
        self.zero_centered = zero_centered
        self.center_range = center_range

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            if self.zero_centered:
                noise = np.random.normal(0, self.add_std, (self.num_added, sample['point_clouds'][i].shape[1]))
            else:
                noise_center = np.random.uniform(-self.center_range, self.center_range, sample['point_clouds'][i].shape[1])
                noise = np.random.normal(0, self.add_std, (self.num_added, sample['point_clouds'][i].shape[1])) + noise_center
            sample['point_clouds'][i] = np.concatenate([sample['point_clouds'][i], noise], axis=0)
        
        return sample
    
class AddPointsAroundJoint():
    def __init__(self, add_std=0.1, max_num2add=1, num_added=32):
        self.add_std = add_std
        self.max_num2add = max_num2add
        self.num_added = num_added

    def __call__(self, sample):
        num_joints = sample['keypoints'][0].shape[0]
        num2add = np.random.randint(1, self.max_num2add)
        idxs2add = np.random.choice(num_joints, num2add, replace=False)

        new_pcs = []
        for i in range(len(sample['keypoints'])):
            pc = sample['point_clouds'][i]
            for idx in idxs2add:
                add_point = sample['keypoints'][i][idx]
                if add_point.shape[-1] < pc.shape[-1]:
                    add_point = np.concatenate([add_point, np.zeros(pc.shape[-1]-add_point.shape[-1])], axis=-1)
                add_points = add_point[np.newaxis, :].repeat(self.num_added, axis=0) + np.random.normal(0, self.add_std, (self.num_added, sample['point_clouds'][-1].shape[-1]))
                pc = np.concatenate([pc, add_points], axis=0)
            new_pcs.append(pc)

        sample['point_clouds'] = new_pcs
        return sample
    
class RemoveOutliers():
    def __init__(self, outlier_type='statistical', num_neighbors=3, std_multiplier=1.0, radius=1.0, min_neighbors=2):
        self.outlier_type = outlier_type
        self.num_neighbors = num_neighbors
        self.std_multiplier = std_multiplier
        self.radius = radius
        self.min_neighbors = min_neighbors
        if outlier_type not in ['statistical', 'radius', 'cluster', 'box']:
            raise ValueError('outlier_type must be "statistical" or "radius" or "cluster" or "box"')

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):

            if self.outlier_type == 'statistical':
                neighbors = NearestNeighbors(n_neighbors=self.num_neighbors+1).fit(sample['point_clouds'][i][...,:3])
                distances, _ = neighbors.kneighbors(sample['point_clouds'][i][...,:3])
                mean_dist = np.mean(distances[:, 1:], axis=1)
                std_dist = np.std(distances[:, 1:], axis=1)
                dist_threshold = mean_dist + self.std_multiplier * std_dist
                inliers = np.where(distances[:, 1:] < dist_threshold[:, np.newaxis])

            elif self.outlier_type == 'radius':
                neighbors = NearestNeighbors(radius=self.radius).fit(sample['point_clouds'][i][...,:3])
                distances, _ = neighbors.radius_neighbors(sample['point_clouds'][i][...,:3], return_distance=True)
                inliers = np.where([len(d) >= self.min_neighbors for d in distances])

            elif self.outlier_type == 'cluster':
                clusterer = DBSCAN(min_samples=self.min_neighbors)
                inliers = clusterer.fit_predict(sample['point_clouds'][i][...,:3]) != -1
                if np.sum(inliers) == 0:
                    inliers[0] = True

            elif self.outlier_type == 'box':
                inliers = np.where(np.all(np.abs(sample['point_clouds'][i][...,:2] - np.array([[0, 1]])) < self.radius, axis=1))
            
            else:
                raise ValueError('You should never reach here!')
            
            if len(inliers[0]) == 0:
                sample['point_clouds'][i] = np.zeros((2, sample['point_clouds'][i].shape[1]))
            else:
                sample['point_clouds'][i] = sample['point_clouds'][i][inliers]

        return sample

class Pad():
    def __init__(self, max_len, pad_type='repeat'):
        self.max_len = max_len
        self.pad_type = pad_type
        if pad_type not in ['zero', 'repeat']:
            raise ValueError('pad_type must be "zero" or "repeat"')

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            cur_len = sample['point_clouds'][i].shape[0]
            if cur_len == 0:
                # add random points if the point cloud is empty
                sample['point_clouds'][i] = np.random.normal(0, 1, (self.max_len, sample['point_clouds'][i].shape[1]))
            elif cur_len >= self.max_len:
                indices = np.random.choice(cur_len, self.max_len, replace=False)
                sample['point_clouds'][i] = sample['point_clouds'][i][indices]
            else:
                if self.pad_type == 'zero':
                    sample['point_clouds'][i] = np.pad(sample['point_clouds'][i], ((0, self.max_len - sample['point_clouds'][i].shape[0]), (0, 0)), mode='constant')
                elif self.pad_type == 'repeat':
                    repeat = self.max_len // cur_len
                    residue = self.max_len % cur_len
                    indices = np.random.choice(cur_len, residue, replace=False)
                    sample['point_clouds'][i] = np.concatenate([sample['point_clouds'][i] for _ in range(repeat)] + [sample['point_clouds'][i][indices]], axis=0)
                else:
                    raise ValueError('You should never reach here! pad_type must be "zero" or "repeat"')
        sample['point_clouds'] = np.stack(sample['point_clouds'])
        return sample
    
class MultiFrameAggregate():
    def __init__(self, num_frames):
        self.num_frames = num_frames
        assert num_frames % 2 == 1, 'num_frames must be odd'
        self.offset = (num_frames - 1) // 2

    def __call__(self, sample):
        total_frames = len(sample['point_clouds'])
        if self.num_frames <= total_frames:
            sample['point_clouds'] = [np.concatenate(sample['point_clouds'][i-self.offset:i+self.offset]) for i in range(self.offset, total_frames-self.offset)]
            if 'keypoints' in sample:
                sample['keypoints'] = sample['keypoints'][self.offset:-self.offset]
        return sample

class MultiFrameAggregate():
    def __init__(self, num_frames):
        self.num_frames = num_frames
        assert num_frames % 2 == 1, 'num_frames must be odd'
        self.offset = (num_frames - 1) // 2

    def __call__(self, sample):
        total_frames = len(sample['point_clouds'])
        if self.num_frames <= total_frames:
            sample['point_clouds'] = [np.concatenate(sample['point_clouds'][i-self.offset:i+self.offset]) for i in range(self.offset, total_frames-self.offset)]
            # sample['point_clouds'] = [np.concatenate(sample['point_clouds'][np.maximum(0, i-self.offset):np.minimum(i+self.offset+1, total_frames-1)]) for i in range(total_frames)]
            if 'keypoints' in sample:
                sample['keypoints'] = sample['keypoints'][self.offset:-self.offset]
        # print('multi frame aggregate', len(sample['point_clouds']), len(sample['keypoints']))
        return sample

class RandomScale():
    def __init__(self, scale_min=0.9, scale_max=1.1):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, sample):
        scale = np.random.uniform(self.scale_min, self.scale_max)
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] *= scale
        if 'keypoints' in sample:
            sample['keypoints'] *= scale
        sample['scale'] = scale
        return sample
    
class RandomRotate():
    def __init__(self, angle_min=-np.pi, angle_max=np.pi, deg=False):
        self.angle_min = angle_min
        self.angle_max = angle_max

        if deg:
            angle_min = np.pi * angle_min / 180
            angle_max = np.pi * angle_max / 180

    def __call__(self, sample):
        angle_1 = np.random.uniform(self.angle_min, self.angle_max)
        angle_2 = np.random.uniform(self.angle_min, self.angle_max)
        rot_matrix = np.array([[np.cos(angle_1), -np.sin(angle_1), 0], [np.sin(angle_1), np.cos(angle_1), 0], [0, 0, 1]]) @ np.array([[np.cos(angle_2), 0, np.sin(angle_2)], [0, 1, 0], [-np.sin(angle_2), 0, np.cos(angle_2)]])
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] = sample['point_clouds'][i][...,:3] @ rot_matrix
        if 'keypoints' in sample:
            sample['keypoints'] = sample['keypoints'] @ rot_matrix
        sample['rotation_matrix'] = rot_matrix
        return sample
    
class RandomTranslate():
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, sample):
        translate = np.random.uniform(-self.translate_range, self.translate_range, 3)
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] += translate
        if 'keypoints' in sample:
            sample['keypoints'] += translate
        sample['translate'] = translate
        return sample

class RandomJitter():
    def __init__(self, jitter_std=0.01):
        self.jitter_std = jitter_std

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] += np.random.normal(0, self.jitter_std, sample['point_clouds'][i][...,:3].shape)
        return sample

class RandomJitterKeypoints():
    def __init__(self, jitter_std=0.01):
        self.jitter_std = jitter_std

    def __call__(self, sample):
        for i in range(len(sample['keypoints'])):
            sample['keypoints'][i][...,:3] += np.random.normal(0, self.jitter_std, sample['keypoints'][i][...,:3].shape)
        return sample
    
class RandomDrop():
    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            drop_indices = np.random.choice(sample['point_clouds'][i].shape[0], int(sample['point_clouds'][i].shape[0] * self.drop_prob), replace=False)
            sample['point_clouds'][i] = np.delete(sample['point_clouds'][i], drop_indices, axis=0)
        return sample
    
class GetCentroid():
    def __init__(self, centroid_type='minball'):
        self.centroid_type = centroid_type
        if centroid_type not in ['none', 'zonly', 'mean', 'median', 'minball', 'dataset_median', 'kps', 'xz']:
            raise ValueError('centroid_type must be "mean" or "minball"')
        
    def __call__(self, sample):
        pc_cat = np.concatenate(sample['point_clouds'], axis=0)
        pc_dedupe = np.unique(pc_cat[...,:3], axis=0)
        if self.centroid_type == 'none':
            centroid = np.zeros(3)
        elif self.centroid_type == 'zonly':
            centroid = np.zeros(3)
            centroid[2] = np.median(pc_dedupe[...,2])
        elif self.centroid_type == 'mean':
            centroid = np.mean(pc_dedupe[...,:3], axis=0)
        elif self.centroid_type == 'median':
            centroid = np.median(pc_dedupe[...,:3], axis=0)
        elif self.centroid_type == 'minball':
            try:
                centroid, _ = get_bounding_ball(pc_dedupe)
            except:
                print('Error in minball')
                centroid = np.mean(pc_dedupe[...,:3], axis=0)
        elif self.centroid_type == 'kps':
            kps_cat = np.concatenate(sample['keypoints'], axis=0)
            centroid = np.array([np.median(kps_cat[:, 0]), np.min(kps_cat[:, 1]), np.median(kps_cat[:, 2])])
        elif self.centroid_type == 'xz':
            centroid = np.array([np.median(pc_dedupe[:, 0]), 0, np.median(pc_dedupe[:, 2])])
        else:
            raise ValueError('You should never reach here! centroid_type must be "mean" or "minball"')
        sample['centroid'] = centroid

        return sample
    
class Normalize():
    def __init__(self, feat_scale=None):
        self.feat_scale = feat_scale

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] -= sample['centroid'][np.newaxis]
            if self.feat_scale:
                sample['point_clouds'][i][...,3:] /= np.array(self.feat_scale)[np.newaxis][np.newaxis]
                sample['feat_scale'] = self.feat_scale
        if 'keypoints' in sample:
            sample['keypoints'] -= sample['centroid'][np.newaxis][np.newaxis]
        return sample
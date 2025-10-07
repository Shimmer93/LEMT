import numpy as np
import pickle
from tqdm import tqdm
import argparse
import torch

def read_predictions(pred_path):
    predictions = torch.load(pred_path, weights_only=False)
    all_preds = []
    all_seq_indices = []
    all_global_indices = []
    for pred, seq_index, global_index in predictions:
        all_preds.append(pred[:, 0])
        all_seq_indices.append(seq_index.cpu().numpy()[:, 0])
        all_global_indices.append(global_index.cpu().numpy()[:, 0])
    all_preds = np.concatenate(all_preds, axis=0)
    all_seq_indices = np.concatenate(all_seq_indices, axis=0)
    all_global_indices = np.concatenate(all_global_indices, axis=0)
    
    # sort by index
    sort_idx = np.argsort(all_global_indices)
    all_preds = all_preds[sort_idx]
    all_seq_indices = all_seq_indices[sort_idx]
    all_global_indices = all_global_indices[sort_idx]
    return all_preds, all_seq_indices, all_global_indices

def read_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        all_data = pickle.load(f)
    return all_data

def smooth_predictions(preds, window_size=5):
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2
    padded_preds = np.concatenate([np.repeat(preds[0:1], pad, axis=0), preds, np.repeat(preds[-1:], pad, axis=0)])
    #kernel = np.ones(window_size) / window_size
    kernel = np.hanning(window_size) / np.sum(np.hanning(window_size))
    
    # padded_preds: (N + 2*pad, J, 3)
    # smoothed_preds: (N, J, 3)

    smoothed_preds = np.zeros_like(preds)
    for i in range(preds.shape[1]):
        for j in range(preds.shape[2]):
            # print(padded_preds[:, i, j].shape, kernel.shape)
            smoothed_preds[:, i, j] = np.convolve(padded_preds[:, i, j], kernel, mode='valid')

    return smoothed_preds

def convert(pred_path, dataset_path, save_path):
    all_preds, all_seq_indices, _ = read_predictions(pred_path)
    all_data = read_dataset(dataset_path)

    seq_list = all_data['splits']['train_rdn_p3'] + all_data['splits']['val_rdn_p3']
    for i in tqdm(range(len(seq_list))):
        seq_idx = seq_list[i]
        seq_data = all_data['sequences'][seq_idx]
        seq_len = len(seq_data['keypoints'])
        seq_preds = all_preds[all_seq_indices == i]
        if len(seq_preds) > 0:
            seq_preds = smooth_predictions(seq_preds, window_size=5)
        assert len(seq_preds) == seq_len, f"Length mismatch for sequence {i} (global index {seq_idx}): {len(seq_preds)} vs {seq_len}"
        all_data['sequences'][seq_idx]['keypoints'] = seq_preds

    with open(save_path, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"Saved converted dataset to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=True, help='Path to the predictions .pt file')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the original dataset .pkl file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the converted dataset .pkl file')
    args = parser.parse_args()

    convert(args.pred_path, args.dataset_path, args.save_path)
        
import numpy as np
import torch
import h5py

def normalize_camera(action_t, cam_mean, cam_std):
    """
    turns raw camera trajectories to z-normalized numbers for training stability
    returns updated action vector
    """
    action = action_t.clone()
    action[..., 8:] = ((action[..., 8:] - cam_mean) / cam_std).float()
    return action

def get_cam_mean_std(dataset: str):
    """fixed stats on the camera from training"""
    with h5py.File(dataset.h5_path, "r") as f:
        cam = torch.as_tensor(f["action"][:, 8:], dtype=torch.float32)
    
    mean = cam.mean(0, keepdim=True)
    std = cam.std(0, keepdim=True)

    return mean, std
def normalize_columns(dataset:str, col: int, target_col: str):
    """
    adapted from https://github.com/lucas-maes/le-wm/blob/main/utils.py
    normalize action and camera columns to exist in the same range
    """

    # needs to be a tensor to compute mean/std of data
    with h5py.File(dataset.h5_path, "r") as f:
        col_data_np = f[col][:]
    col_data_tensor = torch.as_tensor(col_data_np, dtype=torch.float32)

    # get the average and std of each column respectively
    # we are only interested in normalizing camera since 0/1 is accepted by tensor
    cam_mean, cam_std = get_cam_mean_std(dataset.h5_path)

    # HDF5Dataset needs a callable function as the normalizer
    def transform(sample):
        action = torch.as_tensor(sample[col], dtype=torch.float32)
        action = normalize_camera(action, cam_mean, cam_std)
        sample[target_col] = action # set the original column to the new normalized value
        return sample
    
    return transform
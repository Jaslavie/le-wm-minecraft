import numpy as np
import torch
import h5py
from PIL import Image
from torchvision import transforms

from lewm.models.lewm import LeWM


def recvall(sock, nbytes):
    """Receive exactly nbytes from a socket; None if the peer closed."""
    data = b""
    while len(data) < nbytes:
        chunk = sock.recv(nbytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def bytes_to_image(frame):
    """Raw 64x64x3 malmo frame bytes -> PIL image."""
    image = np.frombuffer(frame, dtype=np.uint8).reshape((64, 64, 3))
    return Image.fromarray(image)


def make_transform(image_size):
    """Resize -> tensor transform for the encoder."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def process_frame_pixels(transform, frame):
    """Raw malmo frame -> (1,3,H,W) tensor in 0-255 range (matches encoder training)."""
    return transform(bytes_to_image(frame)).unsqueeze(0) * 255.0

def normalize_camera(action_t, cam_mean, cam_std):
    """
    turns raw camera trajectories to z-normalized numbers for training stability
    returns updated action vector
    """
    action = action_t.clone()
    action[..., 8:] = ((action[..., 8:] - cam_mean) / cam_std).float()
    return action

def planner_output_to_actions(mu, cam_mean, cam_std):
    """
    Convert planner output to Malmo action sequences

    planner output: [forward, left, back, right, jump, sneak, sprint, attack, pitch, yaw]
    - dims 0-7 are binary {0, 1}
    - dims 8-9 are denormalized from camera mean/std to raw degrees
    - Malmo expects camera commands as [turn/yaw, pitch]
    """
    actions = np.asarray(mu, dtype=np.float32).copy()
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)

    # convert primary actions (0-7)
    # actions[:, :8] = (actions[:, :8] > binary_threshold).astype(np.float32)

    # Planner imagines camera; Malmo execution keeps fixed view.
    # mean = cam_mean.detach().cpu().numpy() if torch.is_tensor(cam_mean) else np.asarray(cam_mean)
    # std = cam_std.detach().cpu().numpy() if torch.is_tensor(cam_std) else np.asarray(cam_std)
    # camera = actions[:, 8:] * std + mean
    # actions[:, 8] = camera[:, 1]  # turn/yaw
    # actions[:, 9] = camera[:, 0]  # pitch
    actions[:, 8:10] = 0.0 # disable camera control

    return [actions[0]] if actions.shape[0] == 1 else actions

def get_cam_mean_std(dataset: str):
    """fixed stats on the camera from training"""
    with h5py.File(dataset, "r") as f:
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

def build_lewm(cfg, device=None):
    """Construct a LeWM with the architecture from cfg (weights random-initialized)."""
    model = LeWM(
        image_size=cfg.vit.image_size,
        patch_size=cfg.vit.patch_size,
        embedding_dim=cfg.vit.embedding_dim,
        num_channels=cfg.vit.num_channels,
        num_patches=cfg.vit.num_patches,
        vit_attention_heads=cfg.vit.attention_heads,
        vit_mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes,
        vit_transformer_blocks=cfg.vit.transformer_blocks,
        predictor_attention_heads=cfg.predictor.attention_heads,
        predictor_mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes,
        predictor_transformer_blocks=cfg.predictor.transformer_blocks,
        action_dim=cfg.action_dim,
        dropout=cfg.predictor.dropout,
        history_len=cfg.predictor.history_len,
        num_proj=cfg.sigreg.num_proj,
        factor=cfg.sigreg.factor,
        phi=cfg.sigreg.phi,
    )
    if device is not None:
        model = model.to(device)
    return model


def load_trained_lewm(cfg, checkpoint, device, strict=True):
    lewm_model = build_lewm(cfg)
    result = lewm_model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if not strict:
        allowed_missing = {
            "inverse_dynamics.0.weight",
            "inverse_dynamics.0.bias",
            "inverse_dynamics.2.weight",
            "inverse_dynamics.2.bias",
        }

    lewm_model.eval().to(device)

    return lewm_model

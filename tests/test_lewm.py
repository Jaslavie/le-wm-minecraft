# test suite for ViT model
import pytest
import random
import h5py
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from vit import tinyViT
from predictor import Predictor

#########
# Initialize models with configurations
#########
@pytest.fixture(scope="module")
def vit():
    # initialize params
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="lewm")

        model = tinyViT( 
            image_size=cfg.vit.image_size,
            patch_size=cfg.vit.patch_size,
            embedding_dim=cfg.vit.embedding_dim,
            num_channels=cfg.vit.num_channels,
            num_patches=cfg.vit.num_patches,
            attention_heads=cfg.vit.attention_heads,
            mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes,
            transformer_blocks=cfg.vit.transformer_blocks
        )
        model.eval()
        return model

@pytest.fixture(scope="module")
def predictor():
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="lewm")

        model = Predictor(
            embedding_dim=cfg.vit.embedding_dim,
            attention_heads=cfg.predictor.attention_heads,
            mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes,
            dropout=cfg.predictor.dropout,
            transformer_blocks=cfg.predictor.transformer_blocks,
            history_len=cfg.predictor.history_len,
        )
        model.eval()
        return model

#########
# Test suite
#########
def test_batch_embeddings_distribution(vit):
    """test that the vision transformer outputs evenly distributed"""
    # model.eval()

    with h5py.File("mineRL_training.h5", "r") as f:
        # get pixels (observations). shape: (453496, 64, 64, 3)
        # get first 50 observations
        pixels = f['pixels'][0:50]
        pixels_t = torch.tensor(pixels).float() / 255
    
    # run through model
    with torch.no_grad():
        z = vit(pixels_t)
    
    # check that dimensions are varied across images
    std = z.std(dim=0).mean()
    pdist = torch.pdist(z).mean()
    print(f"Standard deviation: {std}")
    print(f"Pairwise distance: {pdist}")
    
    assert torch.isfinite(z).all()
    assert std > 1e-6
    assert pdist > 1e-6

def test_predictor_output(predictor, vit):
    """test that the predictor outputs the correct next observation embedding"""
    # model.eval()

    with h5py.File("mineRL_training.h5", "r") as f:
        # test on 8 consecutive observations to match history length
        # include 1 extra representing the next observation to target for prediction
        pixels = f['pixels'][0:9]
        pixels_t = torch.tensor(pixels).float() / 255
        print("pixels_t shape", pixels_t.shape)

        # get corresponding actions for observations
        actions = f['action'][0:8]
        print(f"Actions at timestamps 1-8: {'\n'.join([f'at {x}: {actions[x]}' for x in range(1, 8)])}")

    with torch.no_grad():
        z = vit(pixels_t) # [8, 192]
    
    # normalize
    actions_t = torch.tensor(actions[:8]).float().unsqueeze(0) # [1, 8, 8]
    obs_t = z[:8].unsqueeze(0) # [1, 8, 192]
    print("embedding shape", obs_t.shape, "actions shape", actions_t.shape)

    # run through prediction
    obs_target = z[-1].unsqueeze(0) # 9th item
    obs_pred = predictor(obs_t, actions_t)
    
    print(f"Predicted obs embedding: {obs_pred[:10]}, Target obs embedding: {obs_target[:10]}")

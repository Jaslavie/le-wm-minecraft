# test suite for ViT model
import pytest
import random
from vit import tinyViT
import h5py
import torch
from hydra import compose, initialize
from omegaconf import DictConfig


@pytest.fixture(scope="module")
def model():
    # initialize params
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="lewm")

        vit = tinyViT( 
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            embedding_dim=cfg.embedding_dim,
            num_channels=cfg.num_channels,
            num_patches=cfg.num_patches,
            attention_heads=cfg.attention_heads,
            mlp_hidden_nodes=cfg.mlp_hidden_nodes,
            transformer_blocks=cfg.transformer_blocks
        )
        vit.eval()
        return vit

def test_batch_embeddings_distribution(model):
    """test that the vision transformer outputs evenly distributed"""
    # model.eval()

    with h5py.File("mineRL_training.h5", "r") as f:
        # get pixels (observations). shape: (453496, 64, 64, 3)
        # get first 50 observations
        pixels = f['pixels'][0:50]
        pixels_t = torch.tensor(pixels).float() / 255
    
    # run through model
    with torch.no_grad():
        z = model(pixels_t)
    
    # check that dimensions are varied across images
    std = z.std(dim=0).mean()
    pdist = torch.pdist(z).mean()
    print(f"Standard deviation: {std}")
    print(f"Pairwise distance: {pdist}")
    
    assert torch.isfinite(z).all()
    assert std > 1e-6
    assert pdist > 1e-6

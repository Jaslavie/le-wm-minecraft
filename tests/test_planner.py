from lewm.planning.planner import Planner
from lewm.data.utils import get_cam_mean_std
from lewm.models.lewm import LeWM
from lewm.paths import repo_path
import pytest
from hydra import initialize, compose
import torch
import stable_worldmodel as swm

@pytest.fixture(scope="module")
def planner_config():
    # initialize params
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="lewm")

        # create smaller configs
        planner = Planner(
            max_iter=1,
            n_samples=5,
            n_elites=2,
            planning_horizon=cfg.planner.planning_horizon,
            action_dim=cfg.action_dim,
        )
        return planner

@pytest.fixture(scope="module")
def lewm_config():
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="lewm")

        ckpt = repo_path(cfg.paths.best_model)
        lewm = LeWM(
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
            phi=cfg.sigreg.phi
        )
        lewm.load_state_dict(torch.load(ckpt)["model_state_dict"])
        lewm.eval()

        return lewm

def test_planner_returns_valid_plan(planner_config, lewm_config):
    # load data and trained model
    dataset = swm.data.HDF5Dataset("mineRL_training", cache_dir=str(repo_path("data")))

    # select test observation from dataset
    # we dont care about selecting an accurate target, just that the planner
    # can find a valid plan
    pixels = dataset[0]["pixels"] # (T, 64, 64, 3)
    obs = pixels[0].float().unsqueeze(0) / 255.0 # (B, T, 64, 64, 3)
    obs_goal = pixels[-1].float().unsqueeze(0) / 255.0 # (B, T, 64, 64, 3)
    
    # get camera params
    cam_mean, cam_std = get_cam_mean_std(str(repo_path("data/mineRL_training.h5")))
    
    # run planner
    mu, _ = planner_config.planner(lewm_config, obs, obs_goal, cam_mean, cam_std)

    # check
    print(f"mu shape: {mu.shape}")
    assert mu.shape == (planner_config.horizon, planner_config.action_dim)

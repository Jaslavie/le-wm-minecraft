from lewm.planning.planner import Planner
from lewm.utils import get_cam_mean_std, load_trained_lewm
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
        lewm = load_trained_lewm(cfg, torch.load(ckpt))

        return lewm

def test_planner_returns_valid_plan(planner_config, lewm_config):
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="lewm")
    dataset_h5 = repo_path(cfg.paths.data_dir, "mineRL_training.h5")

    # load data and trained model
    dataset = swm.data.HDF5Dataset(path=str(dataset_h5))

    # select test observation from dataset
    # we dont care about selecting an accurate target, just that the planner
    # can find a valid plan
    pixels = dataset[0]["pixels"] # (T, 64, 64, 3)
    obs = pixels[0].float().unsqueeze(0)
    obs_goal = pixels[-1].float().unsqueeze(0)
    
    # get camera params
    cam_mean, cam_std = get_cam_mean_std(str(dataset_h5))
    
    # run planner
    mu, _ = planner_config.planner(lewm_config, obs, obs_goal, cam_mean, cam_std)

    # check
    print(f"mu shape: {mu.shape}")
    assert mu.shape == (planner_config.horizon, planner_config.action_dim)

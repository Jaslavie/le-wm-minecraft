import stable_worldmodel as swm
from utils import normalize_columns
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="./config", config_name="lewm")
def train(cfg: DictConfig):
    # Load and normalize original dataset
    ds = swm.data.HDF5Dataset("mineRL_training", cache_dir=".")

    normalizer = normalize_columns(
        ds,
        col="action",
        target_col="action",
    )

    dataset = swm.data.HDF5Dataset(
        "mineRL_training",
        cache_dir=".",
        transform=normalizer,
    )

    # Encode images
    

if __name__=="__main__":
    train()
import stable_worldmodel as swm
from utils import normalize_columns


if __name__=="__main__":
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

    sample = dataset[0] # trigger normalizer
    print(sample["pixels"])
    print(sample["action"])
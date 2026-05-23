# LeWorldModel (LeWM) for Minecraft

Original paper: [https://arxiv.org/html/2603.19312v1](https://arxiv.org/html/2603.19312v1)

## Getting started

1. Install dependencies

```
-m pip install -e .
```

1. Download dataset into your local repo
2. Run the data_processing.ipynb file in full. This loads the processed data into the data folder. This is necesary for training
3. Running tests

```
python -m pytest
```

### Architecture

The **Vision Transformer** (Ti-ViT) is implemented with 3 attention heads and produces 192 Dim embedding vector (4x smaller than the base embedding).

## Training data

We use the MineRL dataset for training. This looksl like this: `data/MineRLTreechop-v0/<trajectory>/` with `recording.mp4`, `rendered.npz`, `metadata.json`, and processed `actions.npy`.

- **Size**: 210 trajectories, 453,496 total timesteps.
- **Video size**: 64 x 64 RGB

Each cleaned action trajectory is shaped `(timesteps, 10)`, where the x-axis columns are:

```
[ forward, left, back, right, jump, sneak, sprint, attack , camera, camera]
```

Below, the character is moving forward and left during the first timestamp.

```
timestamp 0 = [1, 1, 0, 0, 0, 0, 0, 0]
```

### Additional Reading

- [Neural Architectures for Vision > Transformers (MIT)](https://visionbook.mit.edu/transformers.html)
- [CLS Token in Vision Transformers](https://www.abhik.ai/concepts/attention/cls-token)

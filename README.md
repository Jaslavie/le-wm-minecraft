# LeWorldModel (LeWM) for Minecraft

Original paper: [https://arxiv.org/html/2603.19312v1](https://arxiv.org/html/2603.19312v1)

## Getting started

1. Install dependencies

```
-m pip install -e .
```

1. Download dataset into your local repo
2. Run the nn.ipynb file in full. This loads the processed data into the data folder

## Training data

Each MineRL trajectory lives under `data/MineRLTreechop-v0/<trajectory>/` with `recording.mp4`, `rendered.npz`, `metadata.json`, and processed `actions.npy`.

- **Size**: 210 trajectories, 453,496 total timesteps.
- **Video size**: 64 x 64 RGB

Each cleaned action trajectory is shaped `(timesteps, 8)`, where the x-axis columns are:

```
[ forward, left, back, right, jump, sneak, sprint, attack ]
```

Below, the character is moving forward and left during the first timestamp.

```
timestamp 0 = [1, 1, 0, 0, 0, 0, 0, 0]
```


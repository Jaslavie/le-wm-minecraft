# LeWorldModel (LeWM) for Minecraft

Implementing LeWorldModel (LeWM) from scratch in Minecraft within the Malmo sandbox environment using MineRL training datasets.

Training data: [https://zenodo.org/records/12659939](https://zenodo.org/records/12659939)

Original paper: [https://arxiv.org/html/2603.19312v1](https://arxiv.org/html/2603.19312v1)

## Getting started

1. Add API Keys to .env file
2. Install dependencies

```
python -m pip install -e .
```

1. Download dataset into your local repo
2. Run the data_processing.ipynb file in full. This loads the processed data into the data folder. This is necesary for training
3. Running tests

```
python -m pytest
```

1. Begin training

```
python train.py
```

Trained model checkpoints will be stored in the `checkpoints` folder

## Current Art

TBD

## Architecture

### Le World Model

First, we must teach the model how to create an appropriate internal representation of the environment. We borrow this idea from [human neurobiology](https://arxiv.org/html/2411.04383v1#S3): complex environments are simplified into abstract representations which we make predictions from. When we realize our predictions are wrong (i.e. by comparing against the actual future that occurred), we update our beliefs. Technically, we use a "latent embedding" to accomplish this.

The **Vision Transformer/Encoder** (Ti-ViT) transforms each RGB frame of the video into a 192D embedding. Each frame encodes based on semantic relationships across patches within the same frame (self-attention). ViT processes a batch of 128 clips, a total of 1024 frames at a time:

- **Input (o_t)**: 64x64 RGB images
- **Output (z_t)**: 192 Dim embedding vector, representing a semantic summary of each frame (4x smaller than the base embedding).

The **Predictor** guesses future observations by conditioning the 192D ViT embedding on the action embedding (Recall that the ViT only encodes semantic meaning of each frame independently). Using causal masking, the predictor can see 7 frame embeddings in the past to guess the 8th embedding, but it cannot view the future.

- **Input (z_t0 - z_t7)**: sequence of 7 observation embeddings, **(a_t)**: action embedding at each timestamp
- **Output (z_8)**: predicted next frame observation embedding (t+1, 192 Dim)

**LeWM** is a wrapper around the ViT and Predictor modules.

- **Input**: pixel frames (B, T, 3, 64, 64) and actions (B, T, 10)
- **Output**: list of predicted observation embeddings, actual observation embeddings

The **Loss function** combines Prediction loss (computes the difference between z_t+1 and the target next embedding z_target) and SigReg loss (forces the embeddings to follow a Gaussian distribution. More on this below)

$$
L_{\text{LeWM}} \triangleq L_{\text{pred}} + \lambda  \text{SIGReg}(Z)
$$

**SigReg** addresses the issue of representation collapse. This is when the embeddings oversimplify the observation so predicting the next frame becomes trivially easy. There is more detailed explanation on implementation in the paper.

### Planning Rollout

Next, we need to take actions on these predictions.

TBD

### Integration with Malmo

TBD

## Training data

We use the MineRL dataset for training. The initial dataset looks like this: `data/MineRLTreechop-v0/<trajectory>/` with subfiles `recording.mp4`, `rendered.npz`, `metadata.json`.

After pre-processing, we get `actions.npy`: a binary numpy array of actions activated at each timestamp.

- **Size**: 210 trajectories, 453,496 total timesteps.
- **Video frame dimensions**: 64 x 64 RGB
- **Frame clips**: (B=128, T=8, C=3, H=64, W=64)
  - Each clip is 8 RGB frames from a video sample. Each batch consists of 128 clips
- **Actions**: (B, T, 10)

Each cleaned action trajectory is shaped `(timesteps, 10)`, where the x-axis columns are:

```
[ forward, left, back, right, jump, sneak, sprint, attack , camera, camera]
```

Below, the character is moving forward and left during the first timestamp.

```
timestamp 0 = [1, 1, 0, 0, 0, 0, 0, 0]
```

## Additional Reading

- [Neural Architectures for Vision > Transformers (MIT)](https://visionbook.mit.edu/transformers.html)
- [CLS Token in Vision Transformers](https://www.abhik.ai/concepts/attention/cls-token)


# LeWorldModel (LeWM) for Minecraft

Original paper: [https://arxiv.org/html/2603.19312v1](https://arxiv.org/html/2603.19312v1)

## Getting started

1. Install dependencies

```
-m pip install -e .
```

1. Download dataset into your local repo
2. Run the nn.ipynb file in full. This loads the processed data into the data folder. This is necesary for training
3. Running tests

```
python -m pytest
```

### Current art

TBD

### Architecture

<add architecture diagram here>

First, we must teach the model how to create an appropriate internal representation of the environment. We borrow this idea from [human neurobiology](https://arxiv.org/html/2411.04383v1#S3): complex environments are simplified into abstract representations which we make predictions from. When we realize our predictions are wrong (i.e. by comparing against the actual future that occured), we update our beliefs. Technically, we use a "latent embedding" to accomplish this.

The **Vision Transformer/Encoder** (Ti-ViT) is implemented with 3 attention heads:

- **Input (o_t)**: 64x64 RGB images
- **Output (z_t)**: 192 Dim embedding vector, representing a semantic summary of each frame (4x smaller than the base embedding).

The **Predictor** guesses future actions based on action-conditioned version of the 192D ViT embedding

- **Input (z_t)**: current frame observation embedding (t, 192 Dim) 
- **Output (z_t+1)**: predicted next frame observation embedding (t+1, 192 Dim)

**SigReg** loss computes the difference between z_t+1 and the target next embedding z_target (this is the next timsetamp of the input z_t embedding)



Next, we need to take actions on these predictions.

## Training data

We use the MineRL dataset for training. The initial dataset looks like this: `data/MineRLTreechop-v0/<trajectory>/` with subfiles `recording.mp4`, `rendered.npz`, `metadata.json`. 

After pre-processing, we get `actions.npy`: a binary numpy array of actions activated at each timestamp.

- **Size**: 210 trajectories, 453,496 total timesteps.
- **Video frame dimensions**: 64 x 64 RGB

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

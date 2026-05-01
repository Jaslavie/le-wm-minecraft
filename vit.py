########
# tiny ViT implementation from scratch
# this will act as our encoder
########
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

class PatchEmbedding(nn.Module):
    """
    Images are divided into patches of Y x Y dimension. These are 
    then flattened into linear arrays. These are analogous to "tokens"
    in traditional sentence transformers.
    
    Dims
        Input: 64 x 64 image
            (8 patches per row) x (8 patches per col) = 64 total patches
            8 x 8 x 3 RGB dimensions = 192 flattened patch dimension
        Output: 1024
            project up the embedding dimension to capture more nuanced features per patch
    Args
        to_patch_embed: creates embeddings of each patch
    """
    def __init__(self, image_size, patch_size, embedding_dim, num_channels):
        super().__init__()
        self.embedding_dim = embedding_dim
        print(self.embedding_dim)
        # linear projection of flattened patches
        # we use a single convolution layer as this is mathematically the same but more efficient
        # as it does not require us to flatten the patch matrix first
        self.to_patch_embed = nn.Conv2d(num_channels, self.embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, input):
        """
        Input is the image frame. Each forward pass creates the patch embedding:
            Create embedding of img by passing through convolutional net
            Flatten and transpose into final dimensionality
        """
        # create patch embedding
        input_embed = self.to_patch_embed(input)
        # flatten and transpose into (batch_size, num_patches, embedding_dim)
        # for our case, this will be [1, 64, 1024]
        input_embed_ft = input_embed.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    pass
class Transformer(nn.Module):
    """
    Projected batches are passed through the transformer.
    """
    def __init__(self, attention_heads, embedding_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.attention_heads = nn.MultiheadAttention(embedding_dim, attention_heads)
        self.mlp
    def forward(self):
        pass

class tinyViT(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        assert image_size % patch_size == 0, 'image must be divisible by patch size'
    def forward(self, x):
        pass

@hydra.main(version_base=None, config_path="./config", config_name="lewm")
def main(cfg: DictConfig):
    p = PatchEmbedding(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        embedding_dim=cfg.embedding_dim,
        num_channels=cfg.num_channels,
    )
    img = Image.open('public/jeb_sheep_64x64.png')
    print(img.size) # 64 x 64
    
    convertor = transforms.ToTensor()
    img_t = convertor(img).unsqueeze(0) # add dim to beginning to represent batch
    embed_output = p.to_patch_embed(img_t)

    # transformer expects (batch_size, num_patches, embedding_dim)
    print(embed_output.shape) # [1024, 8, 8] (each patch is a 1024 dim vector)
    embed_output_f = embed_output.flatten(2).transpose(1, 2) # flatten 8 x 8 into one array
    print(embed_output_f.shape)

if __name__=="__main__":
    main()
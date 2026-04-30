########
# tiny ViT implementation from scratch
# this will act as our encoder
########
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class PatchEmbedding(nn.Module):
    """
    Images are divided into patches of Y x Y dimension. These are 
    then flattened into linear arrays. These are analogous to "tokens"
    in traditional sentence transformers.
    
    Args
        to_patch_embed: creates embeddings of each patch
    """
    def __init__(self, image_size, patch_size, embedding_dim=1028, num_channels=4):
        super().__init__()
        # kernel size defines the "moving window" that slides over image
        # stride defines the step size (in pixels)
        # both are the size of the patch to avoid overlap
        self.to_patch_embed = nn.Sequential(
            nn.Conv2d(num_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        )

    def forward(self, input):
        """Input is the image frame"""
        # create patch embedding
        input = self.to_patch_embed(input)
        # flatten embedding to 1d array
        input.flatten()

class Attention(nn.Module):
    pass
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        pass

class tinyViT(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        assert image_size % patch_size == 0, 'image must be divisible by patch size'
    def forward(self):
        pass

if __name__=="__main__":
    p = PatchEmbedding(image_size=64, patch_size=8)
    img = Image.open('public/jeb_sheep.png')
    convertor = transforms.ToTensor()
    img_t = convertor(img)
    print(p.to_patch_embed(img_t))
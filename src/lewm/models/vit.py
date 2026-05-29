"""
Ti-ViT (encoder) implementation from scratch
Expected input dimensions: [B, 3, 64, 64]
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import hydra
from omegaconf import DictConfig

from lewm.paths import repo_path

class PatchEmbedding(nn.Module):
    """
    Images are divided into patches of Y x Y dimension. These are the "tokens". 
    These are then flattened into linear arrays. These are analogous to "tokens"
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
        input_embed = self.to_patch_embed(input.float())
        # flatten and transpose into (batch_size, num_patches, embedding_dim)
        # for our case, this will be [1, 64, 1024]
        input_embed_ft = input_embed.flatten(2).transpose(1, 2)

        return input_embed_ft
class Transformer(nn.Module):
    """
    Projected patches are passed through transformer layers. Transformers operate
    off attention mechanisms which:
        Reduces connections needed across the network (more efficient)
        Highlights most salient features instead of localizing (a limitation of CNN)
        Understands global features of image (via self attention mechanism)

    Output: CLS token for a single frame (1, 64, 1024)
    """
    def __init__(self, attention_heads, embedding_dim, mlp_hidden_nodes):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.attention_heads = nn.MultiheadAttention(embedding_dim, attention_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_nodes),
            nn.GELU(),
            nn.Linear(mlp_hidden_nodes, embedding_dim),
        )
    def forward(self, input):
        """
        Takes patch embedding as input and produces and returns CLS token
        Residuals (skip connections) are used to pass along previous layers output
        """
        residual1 = input
        input = self.layer_norm1(input)
        # input is the key, query, and value in each parallel attention head
        # transformer output is stored in index 0
        input = self.attention_heads(input, input, input)[0]
        input = input + residual1

        residual2 = input
        input = self.layer_norm2(input)
        input = self.mlp(input)
        input = input + residual2

        # return processed cls token with same dimensions ([1, 64, 1024])
        return input 

class tinyViT(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embedding_dim: int,
        num_channels: int,
        num_patches: int,
        attention_heads: int,
        mlp_hidden_nodes: int,
        transformer_blocks: int,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_channels=num_channels,
        )
        # randomize token and embedding vectors
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim))
        
        # stack 12 transformer layers
        self.transformer_blocks = nn.Sequential(
            *[Transformer(attention_heads, embedding_dim, mlp_hidden_nodes) 
                for _ in range(transformer_blocks)]
        )
        
        # the following step is a property unique to the LeWM ViT
        # 1. projection head (1 layer MLP)
        #   maps CLS token into a latent space SIGReg can manipulate more easily
        #   batch norm allows model to see global distribution of images for SIGReg loss
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    def forward(self, input):
        curr_batch = input.size(0) # get the current batch size
        input = self.patch_embedding(input)

        # add cls token and positional embedding
        cls_tokens = self.cls_token.expand(curr_batch, -1, -1)
        input = torch.cat((cls_tokens, input), dim = 1) # prepend at the start of input
        input = input + self.position_embedding # add position embedding at the end

        # pass through 12 layers
        input = self.transformer_blocks(input)

        # extract cls token at index 0
        cls_out = input[:, 0]

        # apply projection head to derive final embedding
        z_t = self.projection_head(cls_out)

        return z_t

@hydra.main(version_base=None, config_path=str(repo_path("config")), config_name="lewm")
def main(cfg: DictConfig):
    vit = tinyViT( 
        image_size=cfg.vit.image_size,
        patch_size=cfg.vit.patch_size,
        embedding_dim=cfg.vit.embedding_dim,
        num_channels=cfg.vit.num_channels,
        num_patches=cfg.vit.num_patches,
        attention_heads=cfg.vit.attention_heads,
        mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes,
        transformer_blocks=cfg.vit.transformer_blocks
    )
    vit.eval()

    # img must be a tensor
    img = Image.open(repo_path("public", "jeb_sheep_64x64.png"))
    print(f"Image size: {img.size}") # 64 x 64
    # convert image from rgba -> rgb
    img = img.convert('RGB')
    convertor = transforms.ToTensor()
    img_t = convertor(img).unsqueeze(0) # add dim to beginning to represent batch
    print(img_t.shape)

    with torch.no_grad():
        embed_output = vit(img_t)
        
    print(f"Embedding output shape (CLS token): {embed_output.shape}")

if __name__=="__main__":
    main()

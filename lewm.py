"""
Wrapper for LeWM
"""
import torch.nn as nn
from vit import tinyViT
from predictor import Predictor
from modules import ActionEmbedder

import torch
import torch.nn.functional as F

class SIGReg(nn.Module):
    """
    The SIGReg error measure how well the embeddings follow a normal distribution
    by comparing the optimal gaussian frequency across all columns.
        input: history of latent observation embeddings (T, B, 192)
        output: mean error across all columns (single scalar)
    """
    def __init__(self, num_proj, factor, phi):
        super().__init__()
        self.num_proj = num_proj

        # store static numbers in gpu buffer since we do not train this
        self.register_buffer("factor", torch.tensor(factor))
        self.register_buffer("phi", torch.tensor(phi))
        
    def forward(self, obs):
        # create a matrix of random projections per dimension (192, 1024)
        # normalize each column to add up to 1 (unit vector property)
        A = torch.randn(obs.size(-1), self.num_proj).to(obs.device)
        A = A / A.norm(p=2, dim=0)

        # map projection to embeddings at each timestamp (T, B, 1024)
        H = obs @ A

        # epps pulley statistic checks if data follows normal distribution
        # compute cosine and sine frequency across all columns
        cos_avg = torch.cos(H).mean(dim=0)
        sin_avg = torch.sin(H).mean(dim=0)
        
        # compute error between perfect gaussian frequency and average across
        # all camera angles from projection matrix
        err = (cos_avg - self.phi).square() + sin_avg.square()
        return err.mean()
        


class LeWM(nn.Module):
    """
    obs: (B, T, C=3, H=64, W=64) raw pixels sequence
    actions: (B, T, A=10) action sequence
    lambd: (float) SIGReg loss weight
    """
    def __init__(self, 
        image_size, 
        patch_size, 
        embedding_dim, 
        num_channels, 
        num_patches, 
        attention_heads, 
        mlp_hidden_nodes, 
        transformer_blocks, 
        action_dim, 
        dropout, 
        num_proj, 
        factor, 
        phi,
    ):
        super().__init__()
        # 192 dim embedding of img
        self.encoder =  tinyViT( 
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_channels=num_channels,
            num_patches=num_patches,
            attention_heads=attention_heads,
            mlp_hidden_nodes=mlp_hidden_nodes,
            transformer_blocks=transformer_blocks
        )
        self.predictor = Predictor(
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
            mlp_hidden_nodes=mlp_hidden_nodes,
            dropout=dropout,
            transformer_blocks=transformer_blocks,
        )
        # 192 dim embedding of actions
        self.action_embedder = ActionEmbedder(
            action_dim=action_dim,
            embedding_dim=embedding_dim,
        )
        self.sigreg = SIGReg(num_proj, factor, phi)
    
    def forward(self, pixels, actions, lambd):
        B, T, C, H, W = pixels.shape # B, T, C, H, W

        # combine batch and time into one dimension for conv2d
        # reshape back to og dimension after converting to embedding
        emb = self.encoder(pixels.reshape(B * T, C, H, W))
        emb = emb.reshape(B, T, -1)
        action_emb = self.action_embedder(actions)

        # predict the next embedding for every non-final timestep.
        next_emb_pred = self.predictor(emb[:, :-1], action_emb[:, :-1])
        next_emb_target = emb[:, 1:] # actual next state

        # get overall loss
        pred_loss = F.mse_loss(next_emb_pred, next_emb_target)

        # step-wise sigreg (anti-collapse)
        sigreg_loss = self.sigreg(emb)

        total_loss = pred_loss + lambd * sigreg_loss
        
        return total_loss
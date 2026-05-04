"""
Wrapper for LeWM
"""
import torch.nn as nn
from vit import tinyViT
from predictor import Predictor
import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from statistics import mean

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
        A = torch.randn(obs.size(-1), self.num_proj)
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
        


@hydra.main(version_base=None, config_path="./config", config_name="lewm")
class LeWM(nn.Module):
    """
    obs: (B, T, C=3, H=64, W=64) raw pixels sequence
    actions: (B, T, A=10) action sequence
    lambd: (float) SIGReg loss weight
    """
    def __init__(self, encoder, predictor, cfg: DictConfig):
        super().__init__()
        self.encoder =  tinyViT( 
            image_size=cfg.vit.image_size,
            patch_size=cfg.vit.patch_size,
            embedding_dim=cfg.vit.embedding_dim,
            num_channels=cfg.vit.num_channels,
            num_patches=cfg.vit.num_patches,
            attention_heads=cfg.vit.attention_heads,
            mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes,
            transformer_blocks=cfg.vit.transformer_blocks
        )
        self.predictor = Predictor(
            action_dim=cfg.action_dim,
            embedding_dim=cfg.vit.embedding_dim,
            attention_heads=cfg.predictor.attention_heads,
            mlp_hidden_nodes=cfg.vit.mlp_hidden_nodes,
            dropout=cfg.predictor.dropout,
            history_len=cfg.predictor.history_len,
            transformer_blocks=cfg.predictor.transformer_blocks
        )
    def forward(self, pixels, actions, lambd):
        emb = self.encoder(pixels) # (B, T, D=192)
        next_emb = self.predictor(emb, actions) #(B, T, D=10)

        # next-embedding prediction loss
        pred_loss = F.mse_loss(emb[:, 1:] - next_emb[:, :-1])
        # step-wise sigreg (anti-collapse)
        sigreg_loss = mean(SIGReg(emb.transpose(0, 1)))
        
        return pred_loss + lambd * sigreg_loss
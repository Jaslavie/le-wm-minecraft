"""
Wrapper for LeWM
"""
import torch.nn as nn
# from lewm.models.vit import tinyViT
from lewm.models.resnet import ResNet
from lewm.models.predictor import Predictor
from lewm.models.modules import ActionEmbedder

import torch
import torch.nn.functional as F

class SIGReg(nn.Module):
    """
    The SIGReg error measure how well the embeddings follow a normal distribution
    by comparing the optimal gaussian frequency across all columns.
        input: history of latent observation embeddings (T, B, 192)
        output: mean error of all observations in the forward pass (single scalar)
    """
    def __init__(self, num_proj, factor, phi, knots=17):
        super().__init__()
        self.num_proj = num_proj

        # t are the points on the curve that we sample to check if it follows 
        # gaussian dist. Here, we sample 17 places (knots)
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        
        # we measure error as the area under the curve
        # get the size of each interval (size of curve / num of intervals)
        interval_width = (3.0) / (knots - 1)
        
        # use trapezoid rule to compute error at each knot
        # weights are used to determine how much to value each error
        weights = torch.zeros(knots, dtype=torch.float32) # init
        weights[0] = interval_width / 2.0
        weights[-1] = interval_width / 2.0
        weights[1:-1] = interval_width # all other weights are rectangular dt's

        # the t impacts how fast cos and sin change along the projection
        # rapid changes (i.e. higher t's) are more noisy so we weight them less
        window = torch.exp(-t.square() / 2.0) 
        weights = weights * window

        # store static numbers in gpu buffer since we do not train this
        self.register_buffer("t", t)
        self.register_buffer("weights", weights)
        self.register_buffer("factor", torch.tensor(factor))
        self.register_buffer("phi", window)
        
    def forward(self, obs):
        # create a matrix of 1024 random projections for the 192 dim vector
        A = torch.randn(obs.size(-1), self.num_proj).to(obs.device)
        A = A / A.norm(p=2, dim=0, keepdim=True) # each projection is a unit vector

        # turn 192D into 1024D by projecting embeddings onto each direction
        # Z (T, B, 192) @ A (D, 1024) = H (T, B, 1024)
        H = obs @ A
        x_t = (H).unsqueeze(-1) * self.t

        # average over the batch dimension
        cos_avg = torch.cos(x_t).mean(dim=1)
        sin_avg = torch.sin(x_t).mean(dim=1)

        # compute error between perfect gaussian frequency and average across
        # all camera angles from projection matrix
        err = (cos_avg - self.phi).square() + sin_avg.square()
        statistic = (err @ self.weights) * obs.size(-2)

        return statistic.mean()
        


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
        vit_attention_heads, 
        vit_mlp_hidden_nodes, 
        vit_transformer_blocks, 
        predictor_attention_heads,
        predictor_mlp_hidden_nodes,
        predictor_transformer_blocks,
        action_dim, 
        dropout, 
        history_len,
        num_proj, 
        factor, 
        phi,
    ):
        super().__init__()
        # 192 dim embedding of img
        # self.encoder =  tinyViT( 
        #     image_size=image_size,
        #     patch_size=patch_size,
        #     embedding_dim=embedding_dim,
        #     num_channels=num_channels,
        #     num_patches=num_patches,
        #     attention_heads=vit_attention_heads,
        #     mlp_hidden_nodes=vit_mlp_hidden_nodes,
        #     transformer_blocks=vit_transformer_blocks,
        # )
        self.encoder = ResNet(
            embedding_dim=embedding_dim,
        )
        self.predictor = Predictor(
            embedding_dim=embedding_dim,
            attention_heads=predictor_attention_heads,
            mlp_hidden_nodes=predictor_mlp_hidden_nodes,
            dropout=dropout,
            transformer_blocks=predictor_transformer_blocks,
            history_len=history_len,
        )
        # 192 dim embedding of actions
        self.action_embedder = ActionEmbedder(
            action_dim=action_dim,
            embedding_dim=embedding_dim,
        )
        self.sigreg = SIGReg(num_proj, factor, phi)

        # inverse dynamics head recovers action a_t from (z_t, z_{t+1}).
        # fixes identity collapse problem by forcing the predictor's output to
        # depend on physics-grounded actions.
        self.action_dim = action_dim
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, action_dim),
        )
    
    def forward(self, pixels, actions):
        B, T, C, H, W = pixels.shape

        # combine batch and time into one dimension for conv2d
        # reshape back to og dimension after converting to embedding
        emb = self.encoder(pixels.reshape(B * T, C, H, W))
        emb = emb.reshape(B, T, -1)
        action_emb = self.action_embedder(actions)

        # predict the next embedding for every non-final timestep.
        next_emb_pred = self.predictor(emb[:, :-1], action_emb[:, :-1])
        next_emb_target = emb[:, 1:] # actual next state

        # recover predicted action based on curr embedding
        action_pred = self.inverse_dynamics(
            torch.cat([emb[:, :-1], next_emb_pred], dim=-1)
        )

        return next_emb_pred, next_emb_target, emb, action_pred

def compute_loss(next_emb_pred, next_emb_target, emb, sigreg, lambd,
                 action_pred=None, action_target=None, lambd_inv=0.0):
    # get overall loss
    pred_loss = F.mse_loss(next_emb_pred, next_emb_target)

    # step-wise sigreg (anti-collapse)
    sigreg_loss = sigreg(emb.transpose(0, 1))

    # inverse-dynamics loss pushes predictor to produce predicted latents with
    # recoverable actions
    inv_loss = next_emb_pred.new_tensor(0.0)
    if action_pred is not None and action_target is not None and lambd_inv > 0:
        # bce loss for binary actions
        bin_loss = F.binary_cross_entropy_with_logits(
            action_pred[..., :8], action_target[..., :8].float()
        )
        # TODO: mse loss for camera actions (disabled for now)
        cam_loss = F.mse_loss(action_pred[..., 8:], action_target[..., 8:].float())
        inv_loss = bin_loss + cam_loss

    # total loss is the sum of the three losses (prediction, sigreg, inverse dynamics)
    total = pred_loss + (lambd * sigreg_loss) + (lambd_inv * inv_loss)
    
    return total, pred_loss, sigreg_loss, inv_loss

"""
Predictor (Decoder): Autoregressively predicts the next observation embedding z_t+1
given prior obs embeddings and actions. Importantly, it does not recommend the next action.
    Observation prediction: Uses temporal causal masking during self attention
    Action prediction: AdaLn normalization conditions each observation on prior actions
"""
import torch.nn as nn
import torch

class Transformer(nn.Module):
    """
    Transformer predicts next observation embedding z_t+1 based on the current obs + previous actions
        Input: observation embedding z_t, action a_t, 
        Output: predicted observation embedding z_t+1
    """
    def __init__(self, attention_heads, embedding_dim, mlp_hidden_nodes, dropout, history_len):
        super().__init__()

        # create temporal mask over future observations (actions will be paired via adaln)
        # the size of the matrix represents the sliding window of observations we attend to 
        self.register_buffer('causal_mask', torch.triu(torch.ones(history_len, history_len), diagonal=1).bool())
        self.attention_heads = nn.MultiheadAttention(embedding_dim, attention_heads, batch_first=True)

        self.adaln1 = AdaLn(embedding_dim)
        self.adaln2 = AdaLn(embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_nodes),
            nn.GELU(),
            nn.Dropout(dropout), # dropout during training avoids overfitting
            nn.Linear(mlp_hidden_nodes, embedding_dim),
            nn.Dropout(dropout),
        )
    def forward(self, obs, action):
        # self attention with masking
        # Mask future timesteps; its size must match the current input sequence length.
        T = obs.size(1)

        # create temporal mask over future observations (actions will be paired via adaln)
        # the size of the matrix represents the sliding window of observations we attend to 
        causal_mask = torch.triu(torch.ones(T, T, device=obs.device), diagonal=1).bool()

        residual1 = obs
        obs = self.adaln1(obs, action) # condition embeddings on action
        obs = self.attention_heads(obs, obs, obs, attn_mask=causal_mask)[0]
        obs = obs + residual1

        residual2 = obs
        obs = self.adaln2(obs, action)
        obs = self.mlp(obs)
        obs = obs + residual2

        return obs 

class AdaLn(nn.Module):
    """
    AdaLN introduced action conditioned normalization. It learns a scale and shift factor
    dynamically at inference time. Normally, this is fixed as a model weight.
        input: embedding of condition vector (prior action states)
        output: each observation embedding token is now conditioned on prior states
    """
    def __init__(self, embedding_dim):
        super().__init__()
        # only apply normalization on observations and don't learn parameters
        self.layer_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

        # project previous actions into gamma, beta onto embedding
        # this makes both values derivable/optimizable during training 
        self.cond_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2), # 192 dim * 2 
            nn.SELU(),
        )
    def forward(self, obs_embedding, action_condition):
        # normalize observations
        obs_n = self.layer_norm(obs_embedding)

        # compute shifts/scale of observations conditioned on actions
        # split output of network into 2 pieces
        gamma, beta = self.cond_proj(action_condition).chunk(2, dim=-1)

        # return normalized obs conditioned on past actions
        return (1 + gamma) * obs_n + beta

class Predictor(nn.Module):
    """
    The predictor applies the transformer blocks to autoregressively predict the next observation
    embedding conditioned on previous actions. Each transformer block evolves the prediction of the next state.
        Input: observation embedding z_t, action vector at timestamp t
        Output: predicted observation embedding z_t+1
    """
    def __init__(self, embedding_dim, attention_heads, mlp_hidden_nodes, dropout, transformer_blocks, history_len):
        super().__init__()
        # 6 transformer layers
        self.transformer_blocks = nn.Sequential(
            *[Transformer(
                attention_heads=attention_heads,
                embedding_dim=embedding_dim,
                mlp_hidden_nodes=mlp_hidden_nodes,
                dropout=dropout,
                history_len=history_len,
            )
                for _ in range(transformer_blocks)]
        )
        # a final projection head is applied after transformer blocks (same as encoder)
        # this allows the model to see the global distribution of embeddings for SIGReg loss
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    def forward(self, obs, actions):
        # pass output of each transformer layer manually to the next layer
        for block in self.transformer_blocks:
            obs = block(obs, actions)

        # apply projection per timestep; BatchNorm1d expects features in dim 1.
        B, T, D = obs.shape
        obs_t_next = self.projection_head(obs.reshape(B * T, D))
        obs_t_next = obs_t_next.reshape(B, T, D)

        # next predicted obs embedding for all timestamps (except last)
        return obs_t_next

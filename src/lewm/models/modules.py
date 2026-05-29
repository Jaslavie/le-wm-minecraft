import torch.nn as nn

class ActionEmbedder(nn.Module):
    """
    Embed action vector into embedding so we condition
    observation embeddings on actions through AdaLn
        input: (B, T, 10)
        output: (B, T, 192)
    """
    def __init__(self, action_dim, embedding_dim):
        super().__init__()
        self.proj = nn.Linear(action_dim, embedding_dim)
    def forward(self, input):
        return self.proj(input.float())

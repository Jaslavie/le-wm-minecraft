"""Pre-trained ResNet-18 model as CNN-based encoder."""
import torchvision.models as models
import torch.nn as nn
import torch

# Load a pre-trained ResNet-18 model
class ResNet(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 192 dim output (fc is the final layer)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)

        # normalize input to ImageNet statistics
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
    def forward(self, x):
        # since MineRL frames have 0-255 integers per color channel, we
        # must normalize them to 0-1 for stability
        return self.model(((x.float() / 255.0) - self.mean) / self.std)
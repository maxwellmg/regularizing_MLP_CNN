import timm
import torch
from torch import nn, optim

'''def __init__(self, in_channels, num_classes):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2)

    # Infer flattened size dynamically
    with torch.no_grad():
        dummy_input = torch.zeros(1, in_channels, 224, 224)
        x = self.pool(torch.relu(self.conv1(dummy_input)))
        x = self.pool(torch.relu(self.conv2(x)))
        self.feature_size = x.view(1, -1).shape[1]

    self.fc1 = nn.Linear(self.feature_size, 128)
    self.head = nn.Linear(128, num_classes)'''

class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x, return_embedding=False):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        embedding = torch.relu(self.fc2(x))
        logits = self.head(embedding)
        if return_embedding:
            return logits, embedding
        return logits
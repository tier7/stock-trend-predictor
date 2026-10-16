import os
import torch
from torch import nn
from torch.utils.data import DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    model = NeuralNetwork(input_size=13)
    x = torch.randn((4, 13))
    out = model(x)
    print(out.shape)
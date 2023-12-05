import torch.nn as nn
from torch import sigmoid


class SmokeNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, hidden_size_2: int = 90):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size)
        )

    def forward(self, x):
        out = sigmoid(self.net(x))
        return out

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device='cpu', dropout=0.0):
        super(Encoder, self).__init__()
        self.device = device

        self.mlp = nn.Sequential(nn.Linear(input_size, hidden_size, device=device),
                                 nn.SiLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_size, output_size, device=device)).to(device)

    def forward(self, y):
        return self.mlp(y)

import torch
from torch import nn, optim


class Sine(nn.Module):
    '''
    A sine activation function.
    '''
    def forward(self, x):
        return torch.sin(x)


class SIREN(nn.Sequential):
    '''
    A sinusoidal representation network.
    '''
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        assert n_layers > 0

        layers = []
        for i in range(n_layers):
            if i + 1 < n_layers:
                layers.append(nn.Linear(n_input, n_hidden))
                layers.append(Sine())
            else:
                layers.append(nn.Linear(n_input, n_output))

        super().__init__(layers)

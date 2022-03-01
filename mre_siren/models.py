import math
import torch
from torch import nn, optim


class Sine(nn.Module):
    '''
    A sine activation function.
    '''
    def __init__(self, w0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def __repr__(self):
        return f'{type(self).__name__}(w0={self.w0})'


class SIRENLayer(nn.Module):

    def __init__(self, n_input, n_output, has_sine=True, w0=30):
        super().__init__()
        self.linear = nn.Linear(n_input, n_output)
        self.has_sine = has_sine
        if has_sine:
            self.sine = Sine(w0)

    def forward(self, x):
        x = self.linear(x)
        if self.has_sine:
            x = self.sine(x)
        return x

    def init_weights(self, layer_idx, w0, c):
        n_input = self.linear.weight.shape[-1]
        if layer_idx == 0:
            w_std = 1 / n_input
        else:
            w_std = math.sqrt(c / n_input) / w0
        with torch.no_grad():
            self.linear.weight.uniform_(-w_std, w_std)


class SIREN(nn.Sequential):
    '''
    A sinusoidal representation network.

    https://github.com/vsitzmann/siren
    '''
    def __init__(self, n_input, n_output, n_hidden, n_layers, w0=30, c=6):
        assert n_layers > 0
        modules = []
        for i in range(n_layers):
            is_first = (i == 0)
            is_last = (i + 1 == n_layers)
            modules.append(SIRENLayer(
                n_input=n_input if is_first else n_hidden,
                n_output=n_output if is_last else n_hidden,
                has_sine=not is_last, w0=w0 if is_first else 1
            ))
        super().__init__(*modules)
        self.init_weights(w0, c)

    def init_weights(self, w0, c):
        for i, m in enumerate(self.children()):
            m.init_weights(i, w0, c)

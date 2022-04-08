import math
import torch
from torch import nn, optim


class Sine(nn.Module):
    '''
    A sine activation function.
    '''
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, x):
        return torch.sin(self.w * x)

    def __repr__(self):
        return f'{type(self).__name__}(w={self.w})'


class SIRENLayer(nn.Module):

    def __init__(self, n_input, n_output, has_sine=True, w=1):
        super().__init__()
        self.linear = nn.Linear(n_input, n_output)
        self.has_sine = has_sine
        if has_sine:
            self.sine = Sine(w)

    def forward(self, x):
        x = self.linear(x)
        if self.has_sine:
            x = self.sine(x)
        return x


class SIREN(nn.Sequential):
    '''
    A sinusoidal representation network.

    https://github.com/vsitzmann/siren
    '''
    def __init__(self, n_input, out_shape, n_hidden, n_layers, w0=30):
        assert n_layers > 0

        n_output = torch.prod(torch.as_tensor(out_shape))
        self.out_shape = out_shape

        modules = []
        for i in range(n_layers):
            is_first_layer = (i == 0)
            is_last_layer = (i + 1 == n_layers)
            modules.append(SIRENLayer(
                n_input=n_input if is_first_layer else n_hidden,
                n_output=n_output if is_last_layer else n_hidden,
                has_sine=not is_last_layer,
                w=w0 if is_first_layer else 1
            ))
        super().__init__(*modules)

    def init_weights(self, c=6, input_scale=1, output_scale=1, output_loc=0):
        for i, m in enumerate(self.children()):
            n_input = m.linear.weight.shape[-1]

            if i == 0:
                w_std = 1 / n_input
            else:
                w_std = math.sqrt(c / n_input)

            with torch.no_grad():
                m.linear.weight.uniform_(-w_std, w_std)

                if i == 0: # map from centered input to [-1, 1]
                    m.linear.weight /= torch.as_tensor(
                        input_scale, device=m.linear.weight.device
                    ).unsqueeze(0)

                if i + 1 == len(self): # map from standard normal to output
                    m.linear.weight *= torch.as_tensor(
                        output_scale, device=m.linear.weight.device
                    ).unsqueeze(1)
                    m.linear.bias[...] = output_loc

    def forward(self, x):
        x = super().forward(x)
        return x.reshape(-1, *self.out_shape)


class NullModel(nn.Module):
    '''
    An intercept-only model.
    '''
    def __init__(self, out_shape, *args, **kwargs):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(out_shape))

    def forward(self, x):
        shape = x.shape[0:1] + self.param.shape
        return self.param.unsqueeze(0).expand(shape)

    def init_weights(self, *args, **kwargs):
        pass

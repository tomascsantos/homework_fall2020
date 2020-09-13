from typing import Union
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    layers = [nn.Linear(input_size, size), activation]
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(size, size))
        layers.append(activation)
    layers += [nn.Linear(size, output_size), output_activation]
    old_net = nn.Sequential(*layers)
    net = Net(input_size, output_size, size)
    print(net)
    print("old net: ", old_net)
    return old_net


class Net(nn.Module):

    def __init__(self, input_size, output_size, size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, output_size)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

def from_numpy(array):
    return torch.from_numpy(array.astype(np.float32))

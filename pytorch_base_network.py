from collections import namedtuple

import torch
import torch.nn.functional as tnn_functional
from torch import nn


def is_iterable(element):
    try:
        iterator = iter(element)
    except TypeError:
        return False
    else:
        return True


def squeeze_iterable(element):
    """if input is an iterable with a single element, return its value. """
    in_iterable = is_iterable(element)
    out_value = element
    out_iterable = in_iterable
    if in_iterable:
        if len(element) == 1:
            # this is the case in which i have a single element iterable.
            out_iterable = False
            out_value = element[0]
    return [out_value, out_iterable]


class PyTorchBaseNetwork(nn.Module):
    """
    Basic PyTorch Network composed of Convolutional and linear layers.
    The Convolutional linear layers are positioned before the linear layer by default. The network only supports linear
    layers for the moment.
    """

    version = (0, 5)

    def __init__(self, input_shape=(784,), conv_layers=None, lin_layers=(128, 64), output_shape=(10,), dropout_p=0):
        """ Network architecture initialization according to linear and convolutional layers features """
        super().__init__()

        if isinstance(input_shape, str):
            ckp=input_shape
            saved = torch.load(ckp)
            if saved["version"] != self.version:
                raise ImportError(
                    "PyTorchBaseNetwork is now at version " + self.version + " but model was saved at version " + saved[
                        'version'])
            print("loading network " + saved["description"])

            self.initialise(input_shape=saved["input_shape"], conv_layers=saved["conv_layers"],
                            lin_layers=saved["lin_layers"], output_shape=saved["output_shape"],
                            dropout_p=saved["dropout_p"])
            self.load_state_dict(saved["state_dict"])
        else:
            self.initialise(input_shape=input_shape, conv_layers=conv_layers,
                            lin_layers=lin_layers, output_shape=output_shape, dropout_p=dropout_p)

    def initialise(self, input_shape=(784,), conv_layers=None, lin_layers=(128, 64), output_shape=(10,), dropout_p=0):
        self.input_shape=input_shape
        self.conv_layers=conv_layers
        self.lin_layers=lin_layers
        self.output_shape=output_shape
        self.dropout_p=dropout_p
        [input_shape, iterable_input] = squeeze_iterable(input_shape)
        [output_shape, _] = squeeze_iterable(output_shape)
        self.all_linear_network = not iterable_input
        self.dropout_p = dropout_p
        if self.all_linear_network:
            assert (conv_layers is None)
            prev_layer_n = input_shape
            self.fc_layers = nn.ModuleList()
            for curr_layer_n in lin_layers:
                self.fc_layers.append(nn.Linear(prev_layer_n, curr_layer_n))
                prev_layer_n = curr_layer_n
            self.out_layer = nn.Linear(prev_layer_n, output_shape)
        else:
            assert (conv_layers is not None)
            print("Convolutional Layers not supported for the moment")

    def forward(self, x):
        """ Forward pass through the network, returns the output logits

        :param x(torch.FloatTensor): input or set of inputs to be processed by the network
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a Tensor but is of type " + str(type(x)))

        if not isinstance(x, torch.FloatTensor):
            raise TypeError("x should be a Tensor of Float but is of type " + str(x.type()))

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = tnn_functional.relu(x)
            if self.dropout_p > 0:
                x = tnn_functional.dropout(x, p=self.dropout_p)
        x = self.out_layer(x)
        return x



    def save_model(self, checkpoint_file, description=None):

        tosave = {"version": self.version, "input_shape": self.input_shape, "conv_layers": self.conv_layers,
                  "lin_layers": self.lin_layers, "output_shape": self.output_shape, "dropout_p": self.dropout_p,
                  "state_dict": self.state_dict(), "description": description}
        torch.save(tosave, checkpoint_file)




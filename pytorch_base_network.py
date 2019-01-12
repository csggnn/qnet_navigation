from torch import nn
from torch import optim
import torch.nn.functional as F

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
    out_value= element
    out_iterable=in_iterable
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
    def __init__(self, input_shape = (784,), conv_layers = None, lin_layers = [128,64], output_shape=(10,)):
        """ Network architecture initialization according to linear and convolutional layers features """
        super().__init__()
        [input_shape, iterable_input] = squeeze_iterable(input_shape)
        [output_shape, dump] = squeeze_iterable(output_shape)
        self.all_linear_network = not iterable_input

        if self.all_linear_network:
            assert (conv_layers is None)
            prev_layer_n = input_shape
            self.fc_layers=nn.ModuleList()
            for curr_layer_n in lin_layers:
                self.fc_layers.append(nn.Linear(prev_layer_n, curr_layer_n))
                prev_layer_n = curr_layer_n
            self.out_layer = nn.Linear(prev_layer_n,output_shape)
        else:
            assert (conv_layers is not None)
            print("Convolutional Layers not supported for the moment")

    def forward(self, x):
        """ Forward pass through the network, returns the output logits """
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = F.relu(x)
        x = self.out_layer(x)
        #x = F.softmax(x, dim=1)
        return x








from parameters import Parameters

import torch
import torch.nn as nn

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(*Parameters.layers.copy())
        # Set eval 
        self.model.train()

    def forward(self, x):
        output = self.model(x)
        return output 

class TargetNetwork(nn.Module):
    def __init__(self):
        super(TargetNetwork, self).__init__()
        self.model = nn.Sequential(*Parameters.layers.copy())
        self.model.eval()

    def forward(self, x):
        output = self.model(x)
        return output
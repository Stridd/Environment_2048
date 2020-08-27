import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class Reinforce_Policy(nn.Module):
    def __init__(self, input_size, output_size, history):
        super(Reinforce_Policy, self).__init__()

        self.layer_1 = nn.Linear(input_size, 128)
        self.layer_2 = nn.Linear(128, output_size)

        self.history = history

        self.reset_policy()
        self.train()

    def reset_policy(self):
        self.log_probablities = []
        self.rewards = []

    def forward(self, x):

        copy_x = x.clone()

        output_1     = self.layer_1(copy_x)
        activation_1 = F.relu(output_1)

        output_2     = self.layer_2(activation_1)

        return output_2

    def act(self, state):
        # Convert the state from numpy to further process
        x = torch.from_numpy(state.astype(np.float32))

        # Get the "estimations" from the neural network
        prob_distr_params = self.forward(x)

        # Detach to store. We only take the first element as we only process for a single state
        output = prob_distr_params.detach().numpy()[0]

        # Store output
        self.history.store_network_output_for_current_episode(output)

        # Use a categorical distribution since we have 4 actions
        distribution = Categorical(logits = prob_distr_params)

        # Store entropy
        entropy = distribution.entropy()
        self.history.store_entropy_for_current_episode(entropy.item())

        # Sample one of the actions
        action = distribution.sample()

        # Get the log probability
        log_probability = distribution.log_prob(action)

        # Store it
        self.log_probablities.append(log_probability)

        return action.item()
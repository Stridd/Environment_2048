import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from Parameters import Parameters


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

        copy_x = x.clone().to(Parameters.device)

        output_1     = self.layer_1(copy_x)
        activation_1 = F.relu(output_1)

        output_2     = self.layer_2(activation_1)

        return output_2

    def get_action(self, state, available_actions):

        # Convert the state from numpy to further process
        x = torch.from_numpy(state.astype(np.float32))

        # Get the "estimations" from the neural network
        prob_distr_params = self.forward(x)

        # Detach to store. We only take the first element as we only process for a single state
        output = prob_distr_params.cpu().detach().numpy()[0]

        # Store output
        self.history.store_network_output_for_current_episode(output)

        # Zero-out the logits for the invalid actions so they won't be selected in the sampling step
        for i in range(len(prob_distr_params[0])):
            if i not in available_actions:
                prob_distr_params[0][i] = float('-inf')

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

        return action.detach().item()

    def store_reward(self, reward):
        self.rewards.append(reward)
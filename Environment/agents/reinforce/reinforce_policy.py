import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical

from parameters import Parameters
from utilities import Utility

class Reinforce_Policy(nn.Module):
    def __init__(self, history):
        super(Reinforce_Policy, self).__init__()

        self.model = nn.Sequential(*Parameters.layers.copy()).to(Parameters.device) 

        if Parameters.weight_init != None:
            self.initialize_weights()

        self.history = history

        self.reset_policy()
        self.train()

    def initialize_weights(self):
        self.model.apply(Utility.get_initialization_function)

    def reset_policy(self):
        self.log_probablities = []
        self.rewards = []
        
    def forward(self, x):

        output = self.model(x)
        return output
        
    def get_action(self, state, available_actions):

        # Convert the state from numpy to further process
        x = torch.from_numpy(state.astype(np.float32)).to(Parameters.device)

        # Get the "estimations" from the neural network
        prob_distr_params = self.forward(x)

        for i in range(len(prob_distr_params)):
            if i not in available_actions:
                prob_distr_params[i] = float('-Inf')
        
        softmax_params = F.softmax(prob_distr_params, dim = 0)

        # Detach to store. We only take the first element as we only process for a single state
        output = prob_distr_params.cpu().detach().numpy()

        # Store output
        self.history.store_network_output_for_current_episode(output)

        # Zero-out the logits for the invalid actions so they won't be selected in the sampling step
        

        # Use a categorical distribution since we have 4 actions
        distribution = Categorical(softmax_params)

        # Store entropy
        with torch.no_grad():
            entropy = distribution.entropy()
            self.history.store_entropy_for_current_episode(entropy.item())

        # Sample one of the actions
        action = distribution.sample()

        # Get the log probability
        log_probability = distribution.log_prob(action)

        # Store it
        self.log_probablities.append(log_probability)

        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)
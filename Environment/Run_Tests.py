#!/usr/bin/env python
# coding: utf-8

# # To-DO:
# - Add pretty print for episode (Game length, maximum cell, maximum cell apparition, end of game score, etc)
# - Add method for setting the random seed explicitly
# - Add method for resetting the generator
# - Add create trainer (max cell, cell distribution(both graphical and numerical), steps taken(episode length), average reward
#   custom_reward_function)
# - Create python trainer(Loss, actions taken, distribution of actions, etc etc)
# - Create transition class (maybe c++)
# - Create neural network for training using different algorithms:
#     - Double Deep Q-Network
#     - AAC
#     - A3C
#     - Bayesian network
#     - Random agent
#     - No Exploration/Max exploration
#     - Other literature approaches

# ## For training we need the following parameters:
# - Epsilon(Exploration rate)
# - Gamma(Discount rate)
# - Learning rate of optimizer
# - Batch size of transitions processed per step
# - Epochs

from Environment_2048 import Environment_2048
from Reinforce_Policy import Reinforce_Policy

policy = Reinforce_Policy(256, 4)

print('Everything compiled fine')
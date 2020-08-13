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

from Reinforce_Agent import Reinforce_Agent
from Utility import Utility
#agent = Reinforce_Agent(256, 4, 0.99, None)
#agent.learn(200)

dictionary = {}
dictionary[4] = 2
dictionary[2] = 6

# Expected 4 * 2 + 2 * 6
print(Utility.get_reward_from_dictionary(dictionary))

print('Everything compiled fine')
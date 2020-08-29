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
from Parameters import Parameters
import numpy as np

#for i in range(5):
agent = Reinforce_Agent()
agent.learn()
agent.write_game_info()

#agent.plot_rewards()
#agent.plot_episode_lengths()
#agent.plot_losses()
#agent.plot_max_cell_evolution()
#agent.plot_max_cell_distribution()

print('Done')

from Reinforce_Policy import Reinforce_Policy
from Environment_2048 import Environment_2048

from History import History
from Logger  import Logger 
from Utility import Utility
from Parameters import Parameters
from Plotter import Plotter

import torch.optim as optim
import numpy as np
import torch
import matplotlib.pyplot as plt
import os 

class Reinforce_Agent():
    def __init__(self):
        self.setup_logger()

        self.history = History()

        self.policy = Reinforce_Policy(Parameters.input_size, Parameters.output_size, self.history)

        self.game =  Environment_2048(4)

        self.gamma  = Parameters.gamma

        self.optimizer = optim.Adam(self.policy.parameters(), lr=Parameters.lr)

        self.penalty = Parameters.penalty

        
    def setup_logger(self):
        current_directory = os.path.dirname(__file__)

        self.logger = Logger(current_directory)


    def train(self):
        episode_length = len(self.policy.rewards)
        returns = np.empty(episode_length, dtype = np.float32)

        # Initialize this for better processing of future rewards
        future_returns = 0.0 

        for t in reversed(range(episode_length)):
            future_returns = self.policy.rewards[t] + self.gamma * future_returns
            returns[t] = future_returns

        returns = torch.tensor(returns)
        log_probabilities = torch.stack(self.policy.log_probablities).flatten()

        loss = -log_probabilities * returns
        # Loss needs to be an item, not an tensor.
        loss = loss[0]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def learn(self):


        for episode in range(Parameters.episodes):
            print('Processing episode {}'.format(episode))

            game_is_done = False

            min_reward = None
            max_reward = None

            steps = 0

            self.game.resetGame()

            while not game_is_done and steps < Parameters.max_episode_duration:
                
                game_board = self.game.getBoard()
                state = np.array(game_board).reshape(1, -1)

                action = self.policy.act(state)

                available_actions = self.game.getAvailableMoves(game_board, len(game_board))

                reward = 0

                if len(available_actions) == 0:

                    self.game.setFinishedIfNoActionIsAvailable()
                elif action in available_actions:

                    self.game.takeAction(action)
                    reward = Utility.get_reward_from_dictionary(self.game.getMergedCellsAfterMove())
                else:
                    reward = self.penalty

                # Check if game ended prematurely

                min_reward = reward if min_reward is None else min(reward, min_reward)
                max_reward = reward if max_reward is None else max(reward, max_reward) 

                self.policy.rewards.append(reward)

                self.history.store_state_action_reward_for_current_episode([game_board, action, reward])

                game_is_done = self.game.isFinished()

                steps += 1

            loss = self.train()

            total_rewards = np.sum(self.policy.rewards)

            max_cell, max_cell_count = Utility.get_max_cell_value_and_count_from_board(self.game.getBoard())

            self.history.store_episode_reward(total_rewards)
            self.history.store_episode_length(steps)
            self.history.store_loss(loss.item())
            self.history.store_min_reward(min_reward)
            self.history.store_max_reward(max_reward)
            self.history.store_max_cell(max_cell)
            self.history.store_max_cell_count(max_cell_count)

            self.history.increment_episode()

            self.policy.reset_policy()

    def write_game_info(self):
        self.logger.write_statistics(self.history)

        max_cell   = max(self.history.max_cell)
        max_length = max(self.history.episode_lengths)
        max_reward = max(self.history.episode_rewards)

        print('MAX CELL: {} | MAX LENGTH: {} | MAX REWARD: {} | '.format(max_cell, max_length, max_reward))

    def plot_rewards(self):
        Plotter.plot_moving_average_of_reward_using_history(self.history, 10)

    def plot_episode_lengths(self):
        Plotter.plot_moving_average_of_episode_lengths_using_history(self.history, 10)

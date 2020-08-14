from Reinforce_Policy import Reinforce_Policy
from Environment_2048 import Environment_2048
from Utility import Utility
import torch.optim as optim
import numpy as np
import torch
import matplotlib.pyplot as plt 

class Reinforce_Agent():
    def __init__(self, input_size, output_size, gamma):
        self.policy = Reinforce_Policy(input_size, output_size)
        self.gamma  = gamma
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)
        self.game =  Environment_2048(4)
        self.penalty = -100
        self.episode_rewards = []

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

    def learn(self, episodes):
        max_steps = 1000

        for episode in range(episodes):

            game_is_done = False

            steps = 0

            self.game.resetGame()

            while not game_is_done and steps < max_steps:
                
                game_board = self.game.getBoard()
                state = np.array(game_board).reshape(1, -1)

                action = self.policy.act(state)
                available_actions = self.game.getAvailableMoves(game_board, len(game_board))
                reward = 0

                if action == -1:
                    self.game.setFinishedIfNoActionIsAvailable()
                elif action in available_actions:
                    self.game.takeAction(action)
                    reward = Utility.get_reward_from_dictionary(self.game.getMergedCellsAfterMove())
                else:
                    reward = self.penalty

                self.policy.rewards.append(reward)

                game_is_done = self.game.isFinished()

                steps += 1

            loss = self.train()
            total_reward = sum(self.policy.rewards)
            self.episode_rewards.append(total_reward)
            self.policy.reset_policy()
            print('| Episode: {} | Loss: {} | Total reward: {} |'.format(episode, loss, total_reward))

    def print_statistics(self):
        plt.plot([i for i in range(len(self.episode_rewards))], self.episode_rewards)
        plt.show()

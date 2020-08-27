import matplotlib.pyplot as plt
import numpy as np

class Plotter():

    @staticmethod
    def plot_moving_average_of_reward_using_history(history, number_episodes):

        reward_history = history.episode_rewards

        # Decided to use convolution. Reason is here: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
        moving_average = np.convolve(reward_history, 
                                    np.ones((number_episodes,))/number_episodes, 
                                    mode='valid')

        Plotter.plot_ma_for(moving_average,'reward')

    @staticmethod
    def plot_moving_average_of_episode_lengths_using_history(history, number_episodes):
        episode_lengths = history.episode_lengths

        # Decided to use convolution. Reason is here: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
        moving_average = np.convolve(episode_lengths, 
                                    np.ones((number_episodes,))/number_episodes, 
                                    mode='valid')

        Plotter.plot_ma_for(moving_average,'episode length')

    @staticmethod
    def plot_ma_for(moving_average, label):
        plt.figure()
        plt.plot([i for i in range(len(moving_average))],moving_average)
        plt.xlabel('Episode')
        plt.ylabel('Moving average of ' + label)
        plt.title(label)
        plt.show()

    @staticmethod
    def plot_losses_using_history(history):
        losses = history.losses
        return 

        

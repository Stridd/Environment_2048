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

        plt.figure()
        plt.plot([i for i in range(len(losses))], losses)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Loss evolution')
        plt.show()

    @staticmethod
    def plot_max_cell_using_history(history):
        max_cell = history.max_cell

        plt.figure()
        plt.plot([i for i in range(len(max_cell))], max_cell)
        plt.xlabel('Episode')
        plt.ylabel('Max cell')
        plt.title('Max cell evolution')
        plt.show()

    @staticmethod
    def plot_max_cell_bins_using_history(history):
        max_cell = history.max_cell
        max_cell_count = history.max_cell_count

        cells_and_occurences = {}

        for i in range(len(max_cell)):
            if max_cell[i] in cells_and_occurences.keys():
                cells_and_occurences[max_cell[i]] += max_cell_count[i]
            else:
                cells_and_occurences[max_cell[i]] = max_cell_count[i]

        # Sort and change the type to print in ascending order
        cells_and_occurences = dict(sorted(cells_and_occurences.items()))
        cells_and_occurences = {str(k):v for k,v in cells_and_occurences.items()}

        # Taken from here: https://stackoverflow.com/questions/16010869/plot-a-bar-using-matplotlib-using-a-dictionary 
        plt.bar(*zip(*cells_and_occurences.items()))
        plt.show()
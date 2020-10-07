from utilities import Utility, DataUtility
from parameters import Parameters
import matplotlib.pyplot as plt
import os

class Plotter():
    def __init__(self, time_of_experiment):

        plotter_directory = os.path.dirname(__file__)

        folder_for_plots = plotter_directory + '\\' + Parameters.plots_folder_name

        Utility.make_folder_if_not_exist(folder_for_plots)
        
        self.folder_for_plots = folder_for_plots + '\\' + time_of_experiment
        
        Utility.make_folder_if_not_exist(self.folder_for_plots)

    def generate_and_save_plots_from_history(self, history):
        self.plot_ma_of_reward_from_history(history)
        self.plot_ma_of_episode_length_from_history(history)
        self.plot_ma_of_loss_from_history(history)
        self.plot_ma_of_max_cell_from_history(history)
        self.plot_max_cell_distribution_from_history(history)

    def plot_ma_of_reward_from_history(self, history):

        reward_history = history.episode_rewards
        
        moving_average = DataUtility.calculate_moving_average_for(reward_history)

        title  = 'Reward evolution'
        ylabel = 'Moving average of reward'
        xlabel = 'Episode'
        data = moving_average
        file_name = 'reward.png'

        concatenated_data = (xlabel, ylabel, title, file_name, data)

        self.plot_and_save_figure(concatenated_data)

    def plot_ma_of_episode_length_from_history(self, history):

        episode_lengths = history.episode_lengths

        moving_average = DataUtility.calculate_moving_average_for(episode_lengths)

        title  = 'Episode length evolution'
        ylabel = 'Moving average of length'
        xlabel = 'Episode'
        data = moving_average
        file_name = 'episode_length.png'

        concatenated_data = (xlabel, ylabel, title, file_name, data)

        self.plot_and_save_figure(concatenated_data)
        
    def plot_ma_of_loss_from_history(self, history):

        losses = history.losses

        moving_average = DataUtility.calculate_moving_average_for(losses)

        title  = 'Loss evolution'
        ylabel = 'Loss'
        xlabel = 'Episode'
        data = moving_average
        file_name = 'loss.png'

        concatenated_data = (xlabel, ylabel, title, file_name, data)

        self.plot_and_save_figure(concatenated_data)

    def plot_max_cell_distribution_from_history(self, history):

        cells_and_occurences = DataUtility.build_and_sort_max_cell_distribution_from_history(history)
    
        # Taken from here: https://stackoverflow.com/questions/16010869/plot-a-bar-using-matplotlib-using-a-dictionary
        plt.figure() 
        plt.bar(*zip(*cells_and_occurences.items()))
        plt.title('Max cell distribution')
        plt.savefig(self.folder_for_plots + '\\' + 'cell_distribution.png', bbox_inches='tight')

    def plot_ma_of_max_cell_from_history(self, history):
        max_cells = history.max_cell
        moving_average = DataUtility.calculate_moving_average_for(max_cells)

        title  = 'Max cell evolution'
        ylabel = 'Moving average of max cell'
        xlabel = 'Episode'
        data = moving_average
        file_name = 'max_cells.png'

        concatenated_data = (xlabel, ylabel, title, file_name, data)
        self.plot_and_save_figure(concatenated_data)

    def plot_and_save_figure(self, data):

        xlabel    = data[0]
        ylabel    = data[1]
        title     = data[2]
        file_name = data[3]
        plot_data = data[4]

        plt.figure()
        plt.plot([i for i in range(len(plot_data))], plot_data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(self.folder_for_plots + '\\' + file_name, bbox_inches='tight')
        plt.close()

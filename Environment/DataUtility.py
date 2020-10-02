from abc import ABC
from Parameters import Parameters
from Enums import Optimizers, RewardFunctions

import numpy as np 
class DataUtility(ABC):

    @staticmethod
    def calculate_moving_average_for(data):
        moving_average_coefficient = len(data) // 10

        if moving_average_coefficient != 0:
            # Decided to use convolution. Reason is here: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
            moving_average = np.convolve(data, 
                                        np.ones((moving_average_coefficient,))/moving_average_coefficient, 
                                        mode='valid')
        else:
            moving_average = data

        return moving_average

    @staticmethod
    def build_and_sort_max_cell_distribution_from_history(history):
        max_cell       = history.max_cell
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

        return cells_and_occurences

    @staticmethod
    def get_max_cell_value_and_count_from_board(board):
        max_cell = None
        max_cell_count = 0
        for i in range(len(board)):
            for j in range(len(board)):
                if max_cell is None or max_cell < board[i][j]: 
                    max_cell = board[i][j]
                    max_cell_count = 1
                elif max_cell == board[i][j]:
                    max_cell_count +=1

        return max_cell, max_cell_count

    @staticmethod
    def get_parameters_class_as_json():
        json_content = {}
        json_content['Episodes']        = Parameters.episodes
        json_content['Learning rate']   = Parameters.lr
        json_content['Momentum']        = Parameters.momentum
        json_content['Gamma']           = Parameters.gamma
        json_content['Board size']      = Parameters.board_size
        json_content['Input size']      = Parameters.input_size
        json_content['Output size']     = Parameters.output_size
        json_content['Optimizer']       = str(Optimizers(Parameters.optimizer).name)
        json_content['Reward function'] = str(RewardFunctions(Parameters.reward_type).name)
        json_content['Model']           = repr(Parameters.layers)
        return json_content

    @staticmethod
    def export_history_to_csv(history, file_name):

        losses          = history.losses
        rewards         = history.episode_rewards
        episode_lengths = history.episode_lengths
        min_rewards     = history.min_rewards
        max_rewards     = history.max_rewards
        max_cells       = history.max_cell
        max_cells_count = history.max_cell_count

        dataFrame = pd.DataFrame()
        headers = ['Loss','Total_Reward','Length','Min_Reward','Max_Reward','Max_Cell','Max_Cell_Count']
        dataFrame.loc[:, 0] = losses
        dataFrame.loc[:, 1] = rewards
        dataFrame.loc[:, 2] = episode_lengths
        dataFrame.loc[:, 3] = min_rewards
        dataFrame.loc[:, 4] = max_rewards
        dataFrame.loc[:, 5] = max_cells
        dataFrame.loc[:, 6] = max_cells_count

        directory_to_save = os.path.dirname(__file__)
        dataFrame.to_csv(directory_to_save + '\\' + file_name, header = headers)
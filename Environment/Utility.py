from abc import ABC
import numpy as np
import pandas as pd
import os 
from Enums import RewardFunctions


class Utility(ABC):

    func_map = None

    @staticmethod
    def get_reward_from_dictionary(cells_dictionary):
        reward = 0
        for cell, times_merged in cells_dictionary.items():
            reward += cell * times_merged

        return reward 

    @staticmethod
    def get_reward(data = None):
        functions_map = get_functions_map()

    @staticmethod
    def get_functions_map():

        if func_map == None:
            build_func_map()
        
        return func_map

    def build_func_map():
        func_map = {}
        func_map[RewardFunctions.cells_merged]          = get_reward_from_dictionary
        func_map[RewardFunctions.distance_to_2048]      = get_reward_by_distance_to_target_cell
        func_map[RewardFunctions.high_cell_high_reward] = get_high_cell_high_reward

    @staticmethod
    def get_reward_by_distance_to_target_cell(cells_dictionary, target_cell):
        raise NotImplementedError

    @staticmethod
    def get_high_cell_high_reward(cells_dictionary):
        raise NotImplementedError

    @staticmethod
    def pretty_print_game_board(board):
        for i in range(len(board)):
            for j in range(len(board)):
                print(board[i][j], end = ' ')
            print('\n', end = '')

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
    def process_state_using_log2_and_factor(state, factor):
        work_state = np.array(state.copy(), dtype = np.float32)
        # Set 0 values to 1 to avoid -inf when doing log2
        work_state[work_state == 0] = 1

        work_state = np.log2(work_state)
        work_state /= factor
        return work_state

    @staticmethod
    def min_max_normalize_state(state):
        work_state = np.array(state.copy(), dtype = np.float32)

        min_cell = np.min(work_state)
        max_cell = np.max(work_state)

        work_state = (work_state - min_cell) / (max_cell - min_cell)

        return work_state

    @staticmethod
    def standardize_state(state):
        work_state = np.array(state.copy(), dtype = np.float32)

        mean_state = np.mean(work_state)
        variance_state = np.var(work_state)

        work_state = (work_state - mean_state) / variance_state
        return work_state

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

    @staticmethod
    def transform_board_into_state(game_board):
        state = np.array(game_board).reshape(1, -1)
        return state 
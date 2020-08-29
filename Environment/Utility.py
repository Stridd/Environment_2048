from abc import ABC
import numpy as np
import pandas as pd 

class Utility(ABC):

    @staticmethod
    def get_reward_from_dictionary(cells_dictionary):
        reward = 0
        for cell, times_merged in cells_dictionary.items():
            reward += cell * times_merged

        return reward 

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
    def export_history_to_csv(history):

        losses          = agent_history.losses
        rewards         = agent_history.episode_rewards
        episode_lengths = agent_history.episode_lengths
        min_rewards     = agent_history.min_rewards
        max_rewards     = agent_history.max_rewards
        max_cells       = agent_history.max_cell
        max_cells_count = agent_history.max_cell_count

        dataFrame = pd.DataFrame()
        headers = ['Loss','Total_Reward','Length','Min_Reward','Max_Reward','Max_Cell','Max_Cell_Count']
        dataFrame[0][0:6] = headers
        dataFrame[1:][0] = rewards
        dataFrame[1:][1] = episode_lengths
        dataFrame[1:][2] = episode_lengths
        dataFrame[1:][3]= min_rewards
        dataFrame[1:][4] = max_rewards
        dataFrame[1:][5] = max_cells
        dataFrame[1:][6] = max_cells_count

        dataFrame.to_csv('C:\\agent_data.csv')


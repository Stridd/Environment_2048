from Parameters import Parameters
from abc import ABC
import numpy as np 

class Utility(ABC):

    @staticmethod
    def get_reward_from_dictionary(cells_dictionary):
        reward = 0
        for cell, times_merged in cells_dictionary.items():
            reward += cell * times_merged

        return reward 

    @staticmethod
    def pretty_print_game_board(board):
        for i in range(Parameters.board_size):
            for j in range(Parameters.board_size):
                print(board[i][j], end = ' ')
            print('\n', end = '')

    @staticmethod
    def get_max_cell_value_and_count_from_board(board):
        max_cell = None
        max_cell_count = 0
        for i in range(Parameters.board_size):
            for j in range(Parameters.board_size):
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
    def standardize_state(state):
        work_state = np.array(state.copy(), dtype = np.float32)
        mean = np.mean(work_state)
        variance = np.var(work_state)
        work_state -= mean
        work_state /= variance
        return work_state
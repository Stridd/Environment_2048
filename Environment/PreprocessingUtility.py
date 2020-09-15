from abc import ABC
import numpy as np
   
class PreprocessingUtility(ABC):

    @staticmethod
    def transform_board_into_state(game_board):
        state = PreprocessingUtility.process_state_using_log2_and_factor(game_board, 11)
        state = np.array(state).reshape(1, -1)
        return state 

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
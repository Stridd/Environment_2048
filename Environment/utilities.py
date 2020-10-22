from abc import ABC
from datetime import datetime 
from pstats import SortKey

import glob
import os
import numpy as np 
import cProfile, pstats, io
import torch.optim as optim
import torch

from enums import WeightInit, Optimizers,RewardFunctions

class Utility(ABC):

    @staticmethod
    def clean_folder(path):
        folder_path = path + '\\' + '*'
        old_files = glob.glob(folder_path)
        for old_file in old_files:
            os.remove(old_file)

    @staticmethod
    def make_folder_if_not_exist(path):
        if not os.path.isdir(path):
            os.mkdir(path)

    @staticmethod
    def get_time_of_experiment():
        time_of_experiment = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        return time_of_experiment
    
    
    @staticmethod
    def profile_function(function, params_folder_name):

        time_of_profiling = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

        pr = cProfile.Profile()
        pr.enable()

        function()

        pr.disable()
        s = io.StringIO()
        sortby = SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)

        current_directory = Utility.get_absolute_path_from_file_name(__file__)

        profiler_folder = current_directory + '\\' + params_folder_name

        Utility.make_folder_if_not_exist(profiler_folder)

        profile_file_name = profiler_folder + '\\' + time_of_profiling + '.stats'

        ps.dump_stats(profile_file_name)

    @staticmethod
    def get_absolute_path_from_file_name(file_name):
        return os.path.dirname(file_name)

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
               cells_and_occurences[max_cell[i]] += 1
            else:
                cells_and_occurences[max_cell[i]] = 1
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

class RewardUtility(ABC):

    func_map = None

    @staticmethod
    def get_reward(type_of_reward, data):
        functions_map = RewardUtility.get_functions_map()
        return functions_map[type_of_reward](data)

    @staticmethod
    def get_functions_map():

        if RewardUtility.func_map == None:
            RewardUtility.build_func_map()
        
        return RewardUtility.func_map

    @staticmethod
    def build_func_map():
        RewardUtility.func_map = {}
        RewardUtility.func_map[RewardFunctions.cells_merged]          = RewardUtility.get_reward_from_dictionary
        RewardUtility.func_map[RewardFunctions.distance_to_2048]      = RewardUtility.get_reward_by_distance_to_2048
        RewardUtility.func_map[RewardFunctions.high_cell_high_reward] = RewardUtility.get_high_cell_high_reward

    @staticmethod
    def get_reward_from_dictionary(cells_dictionary):
        reward = 0
        for cell, times_merged in cells_dictionary.items():
            reward += cell * times_merged

        return reward 

    @staticmethod
    def get_reward_by_distance_to_2048(cells_dictionary):

        target_cell = 2048
        reward = -2048

        if cells_dictionary != {}:

            keys = cells_dictionary.keys()

            max_value_cells_merged = max(keys)

            reward =  max_value_cells_merged - target_cell

        return reward

    @staticmethod
    def get_high_cell_high_reward(cells_dictionary):

        reward = 0
        if cells_dictionary != {}:
            keys = cells_dictionary.keys()
            reward = max(keys)

        return reward

class PreprocessingUtility(ABC):

    @staticmethod
    def transform_board_into_state(game_board):
        state = PreprocessingUtility.process_state_using_log2_and_factor(game_board, 11)
        state = np.array(state, dtype = np.float32).flatten()
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

class OptimizerUtility(ABC):

    PARAMS = None

    @staticmethod
    def set_params(parameters):
        OptimizerUtility.PARAMS = parameters

    @staticmethod
    def get_optimizer_map():
        optimizers = {}
        optimizers[Optimizers.ADAM]     = OptimizerUtility.get_adam
        optimizers[Optimizers.RMSPROP]  = OptimizerUtility.get_RMSPROP
        optimizers[Optimizers.SGD]      = OptimizerUtility.get_SGD
        optimizers[Optimizers.ADAGRAD]  = OptimizerUtility.get_ADAGRAD
        optimizers[Optimizers.ADADELTA] = OptimizerUtility.get_ADADELTA
        return optimizers

    @staticmethod
    def get_adam(network_parameters):
        return optim.Adam(network_parameters, lr = OptimizerUtility.PARAMS.LR)
    
    @staticmethod
    def get_RMSPROP(network_parameters):
        return optim.RMSprop(network_parameters, 
                             lr = OptimizerUtility.PARAMS.LR, 
                             momentum = OptimizerUtility.PARAMS.MOMENTUM)

    @staticmethod
    def get_SGD(network_parameters):
        return optim.SGD(network_parameters, 
                         lr = OptimizerUtility.PARAMS.LR, 
                         momentum = OptimizerUtility.PARAMS.MOMENTUM)
    
    @staticmethod
    def get_ADAGRAD(network_parameters):
        return optim.Adagrad(network_parameters, lr = OptimizerUtility.PARAMS.LR)

    @staticmethod
    def get_ADADELTA(network_parameters):
        return optim.Adadelta(network_parameters, lr = OptimizerUtility.PARAMS.LR)

    @staticmethod
    def get_optimizer_for_parameters(network_parameters):

        optimizer_map = OptimizerUtility.get_optimizer_map()
        optimizer = optimizer_map[OptimizerUtility.PARAMS.OPTIMIZER](network_parameters)

        del optimizer_map

        return optimizer

class WeightInitUtility(ABC):

    PARAMS = None

    @staticmethod
    def set_params(parameters):
        WeightInitUtility.PARAMS = parameters

    @staticmethod
    def get_weight_initialization_map():

        weight_initialization = {}
        weight_initialization[WeightInit.EYE]               = torch.nn.init.eye_
        weight_initialization[WeightInit.UNIFORM]           = torch.nn.init.uniform_
        weight_initialization[WeightInit.NORMAL]            = torch.nn.init.normal_
        weight_initialization[WeightInit.XAVIER_NORMAL]     = torch.nn.init.xavier_normal_
        weight_initialization[WeightInit.XAVIER_UNIFORM]    = torch.nn.init.xavier_uniform_
        weight_initialization[WeightInit.KAIMING_NORMAL]    = torch.nn.init.kaiming_normal_
        weight_initialization[WeightInit.KAIMING_UNIFORM]   = torch.nn.init.kaiming_uniform_

        return weight_initialization

    @staticmethod
    def get_initialization_function(nn_layer):

        if type(nn_layer) == torch.nn.Linear:
            weight_init_map = WeightInitUtility.get_weight_initialization_map()
            weight_init_function = weight_init_map[WeightInitUtility.PARAMS.WEIGHT_INIT]
            weight_init_function(nn_layer.weight)
            nn_layer.bias.data.fill_(0.01)
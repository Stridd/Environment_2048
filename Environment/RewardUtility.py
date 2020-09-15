from abc import ABC
from Parameters import Parameters
from Enums import RewardFunctions

class RewardUtility(ABC):

    func_map = None

    @staticmethod
    def get_reward(data = None):
        functions_map = RewardUtility.get_functions_map()
        return functions_map[Parameters.reward_type](data)

    @staticmethod
    def get_functions_map():

        if RewardUtility.func_map == None:
            RewardUtility.build_func_map()
        
        return RewardUtility.func_map

    @staticmethod
    def build_func_map():
        RewardUtility.func_map = {}
        RewardUtility.func_map[RewardFunctions.cells_merged]          = RewardUtility.get_reward_from_dictionary
        RewardUtility.func_map[RewardFunctions.distance_to_2048]      = RewardUtility.get_reward_by_distance_to_target_cell
        RewardUtility.func_map[RewardFunctions.high_cell_high_reward] = RewardUtility.get_high_cell_high_reward

    @staticmethod
    def get_reward_from_dictionary(cells_dictionary):
        reward = 0
        for cell, times_merged in cells_dictionary.items():
            reward += cell * times_merged

        return reward 

    @staticmethod
    def get_reward_by_distance_to_target_cell(cells_dictionary, target_cell):
        raise NotImplementedError

    @staticmethod
    def get_high_cell_high_reward(cells_dictionary):
        raise NotImplementedError
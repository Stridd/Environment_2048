from utilities import DataUtility

class Data_Helper():
    def __init__(self):
        self.clear_current_data()

    def store_min_max_reward(self, reward):
        self.min_reward = reward if self.min_reward is None else min(reward, self.min_reward)
        self.max_reward = reward if self.max_reward is None else max(reward, self.max_reward) 

    def store_max_cell_statistics(self):
        self.max_cell, self.max_cell_count = DataUtility.get_max_cell_value_and_count_from_board(self.game_board)

    def clear_current_data(self):
        self.steps = 0
        self.available_actions = None
        self.min_reward        = None
        self.max_reward        = None
        self.game_board        = None
        self.loss              = None
        self.max_cell          = None
        self.max_cell_count    = None
        self.total_reward      = None 
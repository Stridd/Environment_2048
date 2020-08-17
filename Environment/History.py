class History():
    def __init__(self):
        self.episode_rewards    = []
        self.losses             = []
        self.episode_lengths    = []
        self.min_rewards        = []
        self.max_rewards        = []
        self.max_cells_on_board  = []
        self.count_max_cell_on_board = []    
    
    def add_episode_reward(self, reward):
        self.episode_rewards.append(reward)

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_episode_length(self, episode_length):
        self.episode_lengths.append(episode_length)

    def add_min_reward(self, reward):
        self.min_rewards.append(reward)

    def add_max_reward(self, reward):
        self.max_rewards.append(reward) 

    def add_max_cell(self, cell_value):
        self.max_cells_on_board.append(cell_value)

    def add_max_cell_count(self, cell_count):
        self.count_max_cell_on_board.append(cell_count)

    def get_losses(self):
        return self.losses

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_min_rewards(self):
        return self.min_rewards

    def get_max_rewards(self):
        return self.max_rewards

    def get_max_cells(self):
        return self.max_cells_on_board

    def get_max_cells_count(self):
        return self.count_max_cell_on_board

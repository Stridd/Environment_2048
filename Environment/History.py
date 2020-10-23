class History():
    def __init__(self):

        self.current_episode = 0

        self.episode_rewards    = []
        self.episode_lengths    = []

        self.losses             = []

        self.min_rewards        = []
        self.max_rewards        = []

        self.max_cell           = []
        self.max_cell_count     = []
        
        self.clear_current_episode_data()

    def increment_episode(self):
        self.current_episode += 1

    def store_episode_reward(self, reward):
        self.episode_rewards.append(reward)
    
    def store_loss(self, loss):
        self.losses.append(loss)
    
    def store_episode_length(self, length):
        self.episode_lengths.append(length)

    def store_min_reward(self, reward):
        self.min_rewards.append(reward)

    def store_max_reward(self, reward):
        self.max_rewards.append(reward)

    def store_max_cell(self, max_cell):
        self.max_cell.append(max_cell)

    def store_max_cell_count(self, max_cell_count):
        self.max_cell_count.append(max_cell_count)
    
    def store_state_action_reward_for_current_episode(self, data):

        state, action, reward = data
 
        self.state_evolution_current_episode.append(state)
        self.actions_current_episode.append(action)
        self.rewards_current_episode.append(reward)

    def store_network_output_for_current_episode(self, output):
        self.network_output.append(output)

    def store_entropy_for_current_episode(self, entropy):
        self.entropy.append(entropy)

    def clear_current_episode_data(self):
        self.state_evolution_current_episode         = []
        self.rewards_current_episode                 = []
        self.actions_current_episode                 = []

        self.network_output                          = []
        self.entropy                                 = []

    def add_data_helper_info(self, data):
        self.store_episode_reward(data.total_reward)
        self.store_episode_length(data.steps)
        self.store_loss(data.loss.item())
        self.store_min_reward(data.min_reward)
        self.store_max_reward(data.max_reward)
        self.store_max_cell(data.max_cell)
        self.store_max_cell_count(data.max_cell_count)

class DQNHistory(History):
    def __init__(self):
        super().__init__()

        self.average_rewards = []

    def clear_current_episode_data(self):
        self.state_evolution_current_episode           = []
        self.rewards_current_episode                   = []
        self.actions_current_episode                   = []

        self.network_outputs                           = []
        self.losses                                    = []
        # Can be either explore or exploit
        self.action_type                               = []

    def add_data_helper_info(self, data):
        self.store_episode_length(data.steps)
        self.store_min_reward(data.min_reward)
        self.store_max_reward(data.max_reward)
        self.store_max_cell(data.max_cell)
        self.store_max_cell_count(data.max_cell_count)
        self.store_total_reward(data.rewards)
        self.store_average_reward(data.rewards)
    
    def store_total_reward(self, rewards):
        self.episode_rewards.append(sum(rewards))

    def store_average_reward(self, rewards):
        self.average_rewards.append(sum(rewards) / len(rewards))

    def store_net_q_values(self, q_values):
        self.network_outputs.append(q_values)

    def store_action_type(self, action_type):
        self.action_type.append(action_type)
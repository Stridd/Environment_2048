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
        
        self.state_evolution_per_episode   = {}
        self.rewards_on_action_per_episode = {}
        self.actions_taken_per_episode     = {}

        self.network_output                = {}
        self.entropy_per_episode           = {}

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

        state = data[0]
        action = data[1]
        reward = data[2]    

        if self.current_episode in self.state_evolution_per_episode.keys():
            self.state_evolution_per_episode[self.current_episode].append(state)
            self.actions_taken_per_episode[self.current_episode].append(action)
            self.rewards_on_action_per_episode[self.current_episode].append(reward)
        else:
            self.state_evolution_per_episode[self.current_episode]   = [state]
            self.actions_taken_per_episode[self.current_episode]     = [action]
            self.rewards_on_action_per_episode[self.current_episode] = [reward]

    def store_reward_for_current_episode(self, reward):
        if self.current_episode in self.state_evolution_per_episode.keys():
            self.state_evolution_per_episode[self.current_episode].append(state)
        else:
            self.state_evolution_per_episode[self.current_episode] = [state]

    def store_network_output_for_current_episode(self, output):
        if self.current_episode in self.network_output.keys():
            self.network_output[self.current_episode].append(output)
        else:
            self.network_output[self.current_episode] = [output]

    def store_entropy_for_current_episode(self, entropy):
        if self.current_episode in self.entropy_per_episode.keys():
            self.entropy_per_episode[self.current_episode].append(entropy)
        else:
            self.entropy_per_episode[self.current_episode] = [entropy]
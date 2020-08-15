class History():
    def __init__(self):
        self.episode_rewards = []
        self.losses          = []
        self.episode_lengths  = [] 
    
    def add_episode_reward(self, reward):
        self.episode_rewards.append(reward)

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_episode_length(self, episode_length):
        self.episode_lengths.append(episode_length)

    def get_losses(self):
        return self.losses

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

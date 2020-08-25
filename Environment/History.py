class History():
    def __init__(self):
        self.episode_rewards    = []
        self.losses             = []
        self.episode_lengths    = []
        self.min_rewards        = []
        self.max_rewards        = []
        self.max_cell           = []
        self.max_cell_count     = []
        
        self.state_evolution_per_episode   = {}
        self.rewards_on_action_per_episode = {}
        self.actions_taken_per_episode     = {}
        self.entropy_per_episode           = {}
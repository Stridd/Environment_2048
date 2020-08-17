class Parameters:
    
    def __init__(self):
        self.gamma = 0.9
        
        self.lr = 0.01

        self.episodes = 200
        self.board_size = 4
        
        self.penalty = -2

        self.max_episode_duration = 50000

        self.input_size = 16
        self.output_size = 4
        self.device = 'cuda'

        self.episode_data_file_name = 'episode_data.txt'
        self.general_data_file_name = 'general_data.txt'
        self.logs_folder_name = 'Logs'



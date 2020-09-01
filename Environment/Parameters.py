from Enums import RewardFunctions
class Parameters:
    gamma = 0.9
    
    lr = 0.001

    episodes = 100
    board_size = 4
    
    input_size = 16
    output_size = 4
    
    device = 'cuda'

    episode_data_file_name = 'episode_data.txt'
    logs_folder_name = 'Logs'

    reward_type = RewardFunctions.cells_merged
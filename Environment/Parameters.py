from Enums import RewardFunctions

import torch.nn as nn

class Parameters:
    gamma = 0.9
    
    lr = 0.001

    episodes = 10
    board_size = 4
    
    input_size = 16
    output_size = 4
    
    device = 'cuda'

    experiment_data_file_name = 'experiment_data.txt'
    logs_folder_name = 'Logs'
    plots_folder_name = 'Plots'

    reward_type = RewardFunctions.cells_merged

    layers = \
    [
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, output_size)
    ]

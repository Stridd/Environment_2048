from Enums import RewardFunctions, Optimizers

import torch.nn as nn

class Parameters:
    gamma = 0.9
    
    optimizer = Optimizers.ADAM

    lr = 0.001
    momentum = 0.9
    
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

    load_model = False 
    model_path = r'D:\Projects\1. Environment_2048\Environment\Logs\14-09-2020_06-46-00\model.pt'

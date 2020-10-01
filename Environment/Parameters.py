from Enums import RewardFunctions, Optimizers

import torch
import torch.nn as nn

class Parameters:

    # Learning hyperparameters
    gamma = 0.99
    lr = 0.01
    momentum = 0.9
    
    optimizer = Optimizers.ADAM

    episodes = 5
    board_size = 4
    
    input_size = 16
    output_size = 4
    
    device = 'cpu'

    experiment_data_file_name = 'experiment_data.txt'
    logs_folder_name     = 'Logs'
    plots_folder_name    = 'Plots'
    profiles_folder_name = 'Profiles'

    reward_type = RewardFunctions.cells_merged

    seed = 0

    # Must set manual seed before layer initialization
    if seed is not None:
        torch.manual_seed(0)

    layers = \
    [
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, output_size)
    ]

    load_model = False 
    model_path = r'D:\Projects\1. Environment_2048\Environment\Logs\16-09-2020_06-51-30\model.pt'

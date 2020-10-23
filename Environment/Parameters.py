from enums import RewardFunctions, Optimizers, WeightInit

import torch
import torch.nn as nn

class Parameters:
    # Learning hyperparameters
    
    LR = 0.001
    MOMENTUM = 0.9
    
    EPISODES = 10
    BOARD_SIZE = 4

    INPUT_SIZE  = 16
    OUTPUT_SIZE = 4

    EXPERIMENT_FILE             = 'experiment_data.txt'
    LOG_FOLDER_NAME             = 'logs'
    PLOT_FOLDER_NAME            = 'plots'
    PROFILE_FOLDER_NAME         = 'profiles'
    PARAMETERS_FILE             = 'parameters.json'
    OBTAINED_CELLS_FILE         = 'obtained_cells.json'
    MODEL_NAME                  = 'model.pt'

    OPTIMIZER       = Optimizers.ADAM
    REWARD_FUNCTION = RewardFunctions.cells_merged
    WEIGHT_INIT     = WeightInit.XAVIER_UNIFORM

    SEED = None

    # Must set manual seed before layer initialization
    if SEED is not None:
        torch.manual_seed(SEED)

    NN_LAYERS = None

    PROFILE    = False 
    LOAD_MODEL = False
    MODEL_PATH = None

    def get_constants_as_json():
        json_content = {}
        json_content['episodes']              = Parameters.EPISODES
        json_content['learning rate']         = Parameters.LR
        json_content['momentum']              = Parameters.MOMENTUM
        json_content['board_size']            = Parameters.BOARD_SIZE
        json_content['input_size']            = Parameters.INPUT_SIZE
        json_content['output_size']           = Parameters.OUTPUT_SIZE
        json_content['optimizer']             = str(Optimizers(Parameters.OPTIMIZER).name)
        json_content['reward function']       = str(RewardFunctions(Parameters.REWARD_FUNCTION).name)
        json_content['weight initialization'] = str(WeightInit(Parameters.WEIGHT_INIT).name)
        json_content['seed']                  = 'None' if Parameters.SEED == None else Parameters.SEED
        return json_content

class REINFORCEParameters():

    LR                          = Parameters.LR
    MOMENTUM                    = Parameters.MOMENTUM
    
    EPISODES                    = Parameters.EPISODES
    BOARD_SIZE                  = Parameters.BOARD_SIZE

    INPUT_SIZE                  = Parameters.INPUT_SIZE
    OUTPUT_SIZE                 = Parameters.OUTPUT_SIZE

    EXPERIMENT_FILE             = Parameters.EXPERIMENT_FILE
    LOG_FOLDER_NAME             = Parameters.LOG_FOLDER_NAME
    PLOT_FOLDER_NAME            = Parameters.PLOT_FOLDER_NAME
    PROFILE_FOLDER_NAME         = Parameters.PROFILE_FOLDER_NAME
    PARAMETERS_FILE             = Parameters.PARAMETERS_FILE
    OBTAINED_CELLS_FILE         = Parameters.OBTAINED_CELLS_FILE
    MODEL_NAME                  = Parameters.MODEL_NAME

    OPTIMIZER                   = Parameters.OPTIMIZER
    REWARD_FUNCTION             = Parameters.REWARD_FUNCTION
    WEIGHT_INIT                 = Parameters.WEIGHT_INIT
    LOAD_MODEL                  = Parameters.LOAD_MODEL
    MODEL_PATH                  = Parameters.MODEL_PATH

    SEED = Parameters.SEED

    # Must set manual seed before layer initialization
    if SEED is not None:
        torch.manual_seed(SEED)

    GAMMA = 0.99
    DEVICE = torch.device('cpu')

    NN_LAYERS = \
    [
        nn.Linear(INPUT_SIZE, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, OUTPUT_SIZE)
    ]

    def get_constants_as_json():

        json_content = Parameters.get_constants_as_json()
        json_content['gamma']   = REINFORCEParameters.GAMMA
        json_content['model']   = repr(REINFORCEParameters.NN_LAYERS)
        return json_content

class DDQNParameters():

    LR                          = Parameters.LR
    MOMENTUM                    = Parameters.MOMENTUM
    
    EPISODES                    = 25
    BOARD_SIZE                  = Parameters.BOARD_SIZE

    INPUT_SIZE                  = Parameters.INPUT_SIZE
    OUTPUT_SIZE                 = Parameters.OUTPUT_SIZE

    EXPERIMENT_FILE             = Parameters.EXPERIMENT_FILE
    LOG_FOLDER_NAME             = Parameters.LOG_FOLDER_NAME
    PLOT_FOLDER_NAME            = Parameters.PLOT_FOLDER_NAME
    PROFILE_FOLDER_NAME         = Parameters.PROFILE_FOLDER_NAME
    PARAMETERS_FILE             = Parameters.PARAMETERS_FILE
    OBTAINED_CELLS_FILE         = Parameters.OBTAINED_CELLS_FILE
    MODEL_NAME                  = Parameters.MODEL_NAME

    OPTIMIZER                   = Parameters.OPTIMIZER
    REWARD_FUNCTION             = Parameters.REWARD_FUNCTION
    WEIGHT_INIT                 = Parameters.WEIGHT_INIT
    LOAD_MODEL                  = Parameters.LOAD_MODEL
    MODEL_PATH                  = Parameters.MODEL_PATH

    SEED = 0

    # Must set manual seed before layer initialization
    if SEED is not None:
        torch.manual_seed(SEED)

    # DDQN ONLY
    GAMMA                 = 0.99
    MEMORY_SIZE           = 15000
    EPSILON               = 0.1
    BATCH_SIZE            = 256
    UPDATE_FREQUENCY      = 25
    DEVICE                = torch.device('cuda')

    NN_LAYERS = \
    [
        nn.Linear(INPUT_SIZE, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, OUTPUT_SIZE)
    ]

    def get_constants_as_json():
        json_content = Parameters.get_constants_as_json()
        json_content['gamma']                     = DDQNParameters.GAMMA
        json_content['model']                     = repr(DDQNParameters.LAYERS)
        json_content['memory_size']               = DDQNParameters.MEMORY_SIZE
        json_content['batch_size']                = DDQNParameters.BATCH_SIZE
        json_content['update_frequency']          = DDQNParameters.UPDATE_FREQUENCY
        return json_content
from enums import RewardFunctions, Optimizers, WeightInit

import torch
import torch.nn as nn

class REINFORCEParameters():

    LR = 0.001
    MOMENTUM = 0.9
    
    EPISODES = 15
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

    PROFILE    = False 
    LOAD_MODEL = False
    MODEL_PATH = None

    SEED = 0

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
        json_content = {}
        json_content['episodes']                 = REINFORCEParameters.EPISODES
        json_content['learning rate']            = REINFORCEParameters.LR
        json_content['momentum']                 = REINFORCEParameters.MOMENTUM
        json_content['board_size']               = REINFORCEParameters.BOARD_SIZE
        json_content['input_size']               = REINFORCEParameters.INPUT_SIZE
        json_content['output_size']              = REINFORCEParameters.OUTPUT_SIZE
        json_content['optimizer']                = str(Optimizers(REINFORCEParameters.OPTIMIZER).name)
        json_content['reward function']          = str(RewardFunctions(REINFORCEParameters.REWARD_FUNCTION).name)
        json_content['weight initialization']    = str(WeightInit(REINFORCEParameters.WEIGHT_INIT).name)
        json_content['seed']                     = 'None' if REINFORCEParameters.SEED == None else REINFORCEParameters.SEED
        json_content['gamma']                    = REINFORCEParameters.GAMMA
        json_content['model']                    = repr(REINFORCEParameters.NN_LAYERS)
        return json_content

class DDQNParameters():

    LR = 0.001
    MOMENTUM = 0.9
    
    EPISODES = 1000
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

    PROFILE    = False 
    LOAD_MODEL = False
    MODEL_PATH = None

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
        json_content = {}
        json_content['episodes']                  = DDQNParameters.EPISODES
        json_content['learning rate']             = DDQNParameters.LR
        json_content['momentum']                  = DDQNParameters.MOMENTUM
        json_content['board_size']                = DDQNParameters.BOARD_SIZE
        json_content['input_size']                = DDQNParameters.INPUT_SIZE
        json_content['output_size']               = DDQNParameters.OUTPUT_SIZE
        json_content['optimizer']                 = str(Optimizers(DDQNParameters.OPTIMIZER).name)
        json_content['reward function']           = str(RewardFunctions(DDQNParameters.REWARD_FUNCTION).name)
        json_content['weight initialization']     = str(WeightInit(DDQNParameters.WEIGHT_INIT).name)
        json_content['seed']                      = 'None' if DDQNParameters.SEED == None else DDQNParameters.SEED
        json_content['gamma']                     = DDQNParameters.GAMMA
        json_content['exploration_rate']          = DDQNParameters.EPSILON     
        json_content['model']                     = repr(DDQNParameters.NN_LAYERS)
        json_content['memory_size']               = DDQNParameters.MEMORY_SIZE
        json_content['batch_size']                = DDQNParameters.BATCH_SIZE
        json_content['update_frequency']          = DDQNParameters.UPDATE_FREQUENCY
        return json_content
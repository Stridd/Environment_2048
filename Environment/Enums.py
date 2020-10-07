import enum

class Optimizers(enum.Enum):
    ADAM     = 0
    SGD      = 1
    RMSPROP  = 2
    ADAGRAD  = 3
    ADADELTA = 4
		
class RewardFunctions(enum.Enum):
    cells_merged          = 0
    distance_to_2048      = 1
    high_cell_high_reward = 2
	
class WeightInit(enum.Enum):
    UNIFORM         = 0
    NORMAL          = 1
    EYE             = 2
    XAVIER_UNIFORM  = 3
    XAVIER_NORMAL   = 4
    KAIMING_UNIFORM = 5
    KAIMING_NORMAL  = 6
	
	
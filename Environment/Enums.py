import enum
class RewardFunctions(enum.Enum):
    cells_merged          = 0
    distance_to_2048      = 1
    high_cell_high_reward = 2
    

class Optimizers(enum.Enum):
    ADAM    = 1
    SGD     = 2
    RMSPROP = 3

import enum
class RewardFunctions(enum.Enum):
    cells_merged          = 1
    distance_to_2048      = 2
    high_cell_high_reward = 3
    
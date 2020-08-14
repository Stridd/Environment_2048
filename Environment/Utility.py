from abc import ABC
class Utility(ABC):

    @staticmethod
    def get_reward_from_dictionary(cells_dictionary):
        reward = 0
        for cell, times_merged in cells_dictionary.items():
            reward += cell * times_merged

        return reward 

    @staticmethod
    def pretty_print_game_board(board):
        for i in range(len(board)):
            for j in range(len(board)):
                print(board[i][j], end = ' ')
            print('\n', end = '')

    @staticmethod
    def write_to_file(item):
        with open('Log.txt','w+') as f:
            f.write(item)

import numpy as np
import torch

from .networks import Network, TargetNetwork

from Environment_2048 import Environment_2048

from history            import History
from data_helper        import Data_Helper
from logger             import Logger
from plotter            import Plotter
from utilities          import Utility,RewardUtility,PreprocessingUtility
from memory.memory      import Memory
from parameters         import Parameters


class DDQN():
    def __init__(self):

        self.net           = Network().to(Parameters.device)
        self.target_net    = TargetNetwork().to(Parameters.device)

        self.replay_memory = Memory()
        self.epsilon       = Parameters.epsilon
        self.game          = Environment_2048(Parameters.board_size)

        self.game.resetGame()

        self.time_of_experiment = Utility.get_time_of_experiment()
        
        self.logger = Logger(self.time_of_experiment)
        self.plotter = Plotter(self.time_of_experiment)
        
        self.history = History()
        self.data_helper = Data_Helper()

        self.optimizer = Utility.get_optimizer_for_parameters(self.net.parameters())

    def train(self):

        for episode in range(Parameters.episodes):
            print('Processing episode: {}'.format(episode))
            self.play_until_end_of_game()

    def play_until_end_of_game(self):
        game_is_done = False

        self.game.resetGame()

        available_actions = None

        while not game_is_done:

            game_board = self.game.getBoard()

            # Need available actions to 0-out the probabilities of the invalid moves
            # Also to check if the game is finished
            if available_actions == None:
                available_actions = self.game.getAvailableMoves(game_board, len(game_board))

            #self.data_helper.game_board        = game_board
            #self.data_helper.available_actions = available_actions

            state = PreprocessingUtility.transform_board_into_state(game_board)

            action = self.get_action(state, available_actions)

            self.game.takeAction(action)

            reward = RewardUtility.get_reward(self.game.getMergedCellsAfterMove()) 

            #self.data_helper.store_min_max_reward(reward)

            next_game_board = self.game.getBoard()
            next_state = PreprocessingUtility.transform_board_into_state(next_game_board)

            # TO-DO: There is some inneficiency here, as we compute the available actions twice
            available_actions = self.game.getAvailableMoves(next_game_board, len(next_game_board))

            if len(available_actions) == 0:
                self.game.setFinishedIfNoActionIsAvailable()

            game_is_done = self.game.isFinished()

            self.replay_memory.store_experience( (state,action,reward,next_state,game_is_done) ) 

            self.learn()
            #self.history.store_state_action_reward_for_current_episode([data_helper.game_board, action, reward])
            #self.data_helper.steps += 1

            #self.perform_action_and_store_data(self.data_helper)

            #game_is_done = self.game.isFinished()

    def get_action(self, state, available_actions):
        # Sample a random
        # If less than epsilon, then choose randomly, otherwise, use network to select action
        sample = np.random.random()
        action = None

        if sample < self.epsilon:
            action = np.random.choice(available_actions)
        else:
 
            state = torch.tensor(state, device = Parameters.device)

            # Set all actions as invalid (with return -Inf)
            q_values = np.ones(shape = (4,), dtype = np.float32) * np.float32('-Inf')

            predicted_values = self.net.forward(state)

            # Correct the values for the available actions
            predicted_q_values = predicted_values[0].detach()

            q_values[available_actions] = predicted_q_values[available_actions]

            # See link here: https://stackoverflow.com/questions/42071597/numpy-argmax-random-tie-breaking
            action = np.random.choice(np.where(q_values == q_values.max())[0])
        
        return action

    def learn(self):

        if self.replay_memory.get_size() < Parameters.batch_size:
            return 

        experiences = self.replay_memory.sample_experiences(Parameters.batch_size)

        states, actions, rewards, next_states, dones = [exp[0],exp[1],exp[2],exp[3],exp[4] for exp in experiences]